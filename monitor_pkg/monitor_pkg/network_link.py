"""Networking part of the link monitor.

Contains:
  - LossWindow:     sliding-window packet-loss estimator (shared by both nodes)
  - LinkStats:      snapshot of latency/loss handed to the controller
  - NetworkServer:  echo node (runs on the remote machine)
  - NetworkMonitor: probing node (runs next to the controller)

The controller does NOT need to know anything about topics or QoS: it just
passes a callback to NetworkMonitor and receives a LinkStats every packet.

Directions (tx/rx are from the controller's point of view):
  uplink   (tx): controller -> responder, topic `network_uplink`
  downlink (rx): responder  -> controller, topic `network_downlink`

Loss is measured per direction with INDEPENDENT sequence numbers:
  - uplink loss   = gaps in the probe seq, measured by the responder
  - downlink loss = gaps in the echo seq (the responder's own counter),
                    measured by the controller
Using the probe seq for both would count uplink losses twice (a probe lost on
the uplink is never echoed, so it would show up as a downlink gap too).
"""
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from link_monitor_msg.msg import LinkState


def make_qos(node: Node) -> QoSProfile:
    """QoS selected by the `best_effort` ROS parameter (must match on both ends).

    best_effort:=true  -> real packet loss is visible (recommended for measuring)
    best_effort:=false -> reliable, the middleware retransmits and hides loss
    """
    node.declare_parameter('best_effort', False)
    if node.get_parameter('best_effort').value:
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class LossWindow:
    """Sliding-window loss estimator based on sequence numbers.

    - the first packet is the baseline: nothing is counted as lost before it
    - gaps are counted as losses (capped at `size`, so a huge jump is O(size))
    - duplicate / late packets are ignored
    - a sender restart (seq going back to 0 or far back) resets the window
    """

    def __init__(self, size: int = 100, detect_restart: bool = True):
        self.size = size
        self.detect_restart = detect_restart
        self.expected = None          # unknown until the first packet arrives
        self.resets = 0
        self.received = deque(maxlen=size)  # 1 = received, 0 = lost

    def reset(self):
        self.received.clear()
        self.expected = None
        self.resets += 1

    def update(self, seq: int) -> bool:
        """Register a received seq. Returns True if it was counted."""
        if self.expected is None:
            self.expected = seq
        elif seq < self.expected:
            restarted = self.detect_restart and (
                seq == 0 or (self.expected - seq) > self.size
            )
            if not restarted:
                return False  # duplicate / late packet
            self.reset()
            self.expected = seq

        gap = seq - self.expected
        self.received.extend([0] * min(gap, self.size))
        self.received.append(1)
        self.expected = seq + 1
        return True

    @property
    def loss(self) -> float:
        n = len(self.received)
        return 1.0 - (sum(self.received) / n) if n else 0.0


@dataclass(frozen=True)
class LinkStats:
    tau_tx: float   # ms, median uplink latency
    tau_rx: float   # ms, median downlink latency
    tau_e2e: float  # ms
    p_tx: float     # uplink loss probability
    p_rx: float     # downlink loss probability
    p_e2e: float


class NetworkServer(Node):
    """Echoes every uplink packet back, adding uplink latency and loss."""

    def __init__(self):
        super().__init__('network_monitor_server')

        self.declare_parameter('window', 100)
        window = int(self.get_parameter('window').value)
        qos = make_qos(self)

        self.sub = self.create_subscription(LinkState, 'network_uplink', self._on_uplink, qos)
        self.pub = self.create_publisher(LinkState, 'network_downlink', qos)

        self.up_loss = LossWindow(window)
        self.dn_seq = 0  # echo counter: lets the controller measure downlink loss alone

    def _on_uplink(self, msg: LinkState):
        # One-way latency: only meaningful if both machines' clocks are synchronized
        # (chrony / PTP). Otherwise it includes the clock offset.
        up_lat_ms = (time.time() - msg.ts_sent) * 1000.0

        resets = self.up_loss.resets
        self.up_loss.update(msg.seq)
        if self.up_loss.resets != resets:
            self.get_logger().warn(f'Client restart detected (seq={msg.seq}): loss window reset')

        res = LinkState()
        res.ts_sent = time.time()
        res.seq = self.dn_seq   # NOT msg.seq, see module docstring
        self.dn_seq += 1
        res.up_lat = up_lat_ms
        res.up_loss = float(self.up_loss.loss)
        self.pub.publish(res)


class NetworkMonitor(Node):
    """Sends probe packets and computes latency/loss statistics from the echoes.

    `on_update(LinkStats)` is called for every echoed packet received.
    """

    def __init__(
        self,
        on_update: Callable[[LinkStats], None],
        send_period: float = 0.1,
        window: int = 100,
        log_period: float = 2.0,
    ):
        super().__init__('network_monitor_client')

        qos = make_qos(self)
        self.pub = self.create_publisher(LinkState, 'network_uplink', qos)
        self.sub = self.create_subscription(LinkState, 'network_downlink', self._on_downlink, qos)
        self.create_timer(send_period, self._send_packet)

        self.on_update = on_update
        self.log_period = log_period
        self.seq = 0

        self.dn_loss = LossWindow(window)
        self.history_up_lat = deque(maxlen=window)
        self.history_dn_lat = deque(maxlen=window)

    def _send_packet(self):
        msg = LinkState()
        msg.ts_sent = time.time()
        msg.seq = self.seq
        self.pub.publish(msg)
        self.seq += 1

    def _on_downlink(self, msg: LinkState):
        now = time.time()

        self.dn_loss.update(msg.seq)

        # One-way latencies need synchronized clocks (chrony / PTP). With an offset
        # theta: up = true_up + theta, dn = true_dn - theta. The sum (e2e) is still
        # correct, the individual values are not. Do NOT clamp: it breaks the sum.
        up_lat = msg.up_lat
        dn_lat = (now - msg.ts_sent) * 1000.0
        if up_lat < 0 or dn_lat < 0:
            self.get_logger().warn(
                'Negative one-way latency: clocks are not synchronized. '
                'e2e is still valid, up/down individually are not.',
                throttle_duration_sec=10.0,
            )
        self.history_up_lat.append(up_lat)
        self.history_dn_lat.append(dn_lat)

        tau_up = statistics.median(self.history_up_lat)
        tau_dn = statistics.median(self.history_dn_lat)

        p_up = msg.up_loss
        p_dn = self.dn_loss.loss
        p_e2e = p_up + p_dn - (p_up * p_dn)

        # One status line every few seconds instead of prints per packet
        self.get_logger().info(
            f'lat up/dn/e2e {tau_up:.1f}/{tau_dn:.1f}/{tau_up + tau_dn:.1f} ms | '
            f'loss up/dn/e2e {p_up * 100:.1f}/{p_dn * 100:.1f}/{p_e2e * 100:.1f} %',
            throttle_duration_sec=self.log_period,
        )

        self.on_update(LinkStats(
            tau_tx=tau_up, tau_rx=tau_dn, tau_e2e=tau_up + tau_dn,
            p_tx=p_up, p_rx=p_dn, p_e2e=p_e2e,
        ))