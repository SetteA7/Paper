"""Link monitor: latency and packet loss in both directions.

Both ends run the same logic (Peer):
  - every `period` s, publish a heartbeat on their own topic (seq + send time)
  - receive the other end's heartbeats and measure that incoming direction:
      loss    = missing seq numbers in a sliding window. If nothing arrives for
                `timeout` s, one packet per period is counted as lost. This is
                what makes it work over TCP too: there packets are not dropped,
                the link stalls and they arrive late (or never).
      latency = receive time - send time (one-way, needs synced clocks)
  - put their measurement of the incoming direction (up_lat / up_loss fields)
    in their own heartbeat. That's how the controller learns the uplink values.

Directions (from the controller's point of view):
  tx = uplink   controller -> bot, topic `network_uplink`   (measured by the bot)
  rx = downlink bot -> controller, topic `network_downlink` (measured by the controller)

Heartbeats are independent in each direction (no echo), so a dead uplink does
not show up as downlink loss.

Clocks: one-way latencies are only correct if both machines are synced
(chrony). With an offset theta, tx = true_tx + theta and rx = true_rx - theta,
so tau_e2e = tx + rx is always correct.

`period` and `window` must be the same on both ends.
"""
import socket
import statistics
import struct
import time
import zlib
from collections import deque
from dataclasses import dataclass
from typing import Callable

from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import UInt8MultiArray

from link_monitor_msg.msg import LinkState

# --- Noise (see noise.py): every Peer is always a sink for the other machine's noise.
# Message header: host id, stream id, seq, send time (20 bytes), then random bytes.
NOISE_TOPIC = 'network_noise'
NOISE_HEADER = struct.Struct('<IIId')
NOISE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=10)
HOST_ID = zlib.crc32(socket.gethostname().encode())   # noise from this machine is ignored
NOISE_PERIOD = 2.0                                     # s, rate/loss averaging window


@dataclass(frozen=True)
class LinkStats:
    tau_tx: float   # ms, median uplink latency
    tau_rx: float   # ms, median downlink latency
    tau_e2e: float  # ms
    p_tx: float     # uplink loss probability
    p_rx: float     # downlink loss probability
    p_e2e: float
    noise_kbps: float = 0.0   # noise received from the other machine
    noise_loss: float = 0.0   # its loss (0-1)


class NoiseSink:
    """Counts the noise received from other machines: rate and loss per stream."""

    def __init__(self):
        self.kbps = 0.0
        self.loss = 0.0
        self._bytes = self._msgs = self._lost = 0
        self._expected = {}           # (host, stream) -> next seq
        self._t = time.monotonic()

    def on_msg(self, msg):
        data = msg.data
        if len(data) < NOISE_HEADER.size:
            return
        host, stream, seq, _ = NOISE_HEADER.unpack_from(data, 0)
        if host == HOST_ID:
            return                    # sent from this machine, did not cross the link
        key = (host, stream)
        exp = self._expected.get(key)
        if exp is None or seq < exp - 100:      # new or restarted sender
            exp = seq
        if seq < exp:
            return                    # late duplicate, already counted lost
        self._lost += seq - exp
        self._expected[key] = seq + 1
        self._msgs += 1
        self._bytes += len(data)

    def update(self):
        """Every NOISE_PERIOD: rate and loss over the last window."""
        now = time.monotonic()
        self.kbps = self._bytes * 8 / 1000 / (now - self._t)
        total = self._msgs + self._lost
        self.loss = self._lost / total if total else 0.0
        self._bytes = self._msgs = self._lost = 0
        self._t = now

    def __str__(self):
        return f'noise rx {self.kbps:.0f} kbps (loss {self.loss * 100:.1f} %)'


class Direction:
    """Loss and latency of one incoming direction, over the last `window` packets."""

    def __init__(self, window: int, timeout: float):
        self.window = window
        self.timeout = timeout
        self.received = deque(maxlen=window)  # 1 = received, 0 = lost
        self.latencies = deque(maxlen=window)  # ms
        self.expected = None                   # next seq we expect
        self.last_rx = time.time()

    def reset(self):
        """Forget all history (used when the other end restarts)."""
        self.received.clear()
        self.latencies.clear()
        self.expected = None

    def on_packet(self, seq: int, ts_sent: float, now: float) -> bool:
        """Returns True if the sender restarted (stats were reset)."""
        restarted = self.expected is not None and seq < self.expected and (
            seq == 0 or self.expected - seq > self.window)
        if restarted:
            # New session: drop the losses counted while the sender was down,
            # so stats start from zero instead of from ~100% loss.
            self.reset()
        if self.expected is None:
            self.expected = seq                # first packet / sender restarted
        elif seq < self.expected:
            return False                       # late or duplicate: already counted lost
        self.received.extend([0] * min(seq - self.expected, self.window))
        self.received.append(1)
        self.expected = seq + 1
        self.last_rx = now
        self.latencies.append((now - ts_sent) * 1000.0)
        return restarted

    def on_tick(self, now: float):
        """Call once per period: counts a lost packet while the link is silent.
        Nothing is counted before the first packet (no peer yet, nothing to lose)."""
        if self.expected is not None and now - self.last_rx > self.timeout:
            self.received.append(0)
            self.expected += 1

    @property
    def loss(self) -> float:
        n = len(self.received)
        return 1.0 - sum(self.received) / n if n else 1.0

    @property
    def latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0


class Peer(Node):
    """Publishes heartbeats on `tx_topic`, measures the heartbeats on `rx_topic`."""

    def __init__(self, name, tx_topic, rx_topic, period=0.1, window=100, timeout=0.5):
        super().__init__(name)
        period = float(self.declare_parameter('period', period).value)
        window = int(self.declare_parameter('window', window).value)
        timeout = float(self.declare_parameter('timeout', timeout).value)

        # best_effort:=true (default) shows the real loss; reliable QoS makes the
        # middleware retransmit, so loss turns into latency. Must match on both ends.
        if self.declare_parameter('best_effort', True).value:
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        else:
            qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pub = self.create_publisher(LinkState, tx_topic, qos)
        self.create_subscription(LinkState, rx_topic, self._on_msg, qos)
        self.create_timer(period, self._on_timer)

        self.incoming = Direction(window, timeout)
        self.last_msg = None  # last heartbeat from the other end
        self.seq = 0

        # Always-on sink for noise.py running on the other machine (DDS/Zenoh only
        # send the noise over the link if someone there subscribes).
        self.noise = NoiseSink()
        self.create_subscription(UInt8MultiArray, NOISE_TOPIC, self.noise.on_msg, NOISE_QOS)
        self.create_timer(NOISE_PERIOD, self.noise.update)

    def _on_msg(self, msg: LinkState):
        if self.incoming.on_packet(msg.seq, msg.ts_sent, time.time()):
            self.get_logger().info('Other end restarted: link stats reset')
        self.last_msg = msg

    def _on_timer(self):
        self.incoming.on_tick(time.time())
        msg = LinkState()
        msg.seq = self.seq
        msg.up_lat = float(self.incoming.latency)  # what *we* measure on our rx
        msg.up_loss = float(self.incoming.loss)
        msg.ts_sent = time.time()
        self.pub.publish(msg)
        self.seq += 1
        self.on_period()

    def on_period(self):
        pass


class NetworkServer(Peer):
    """Bot side: measures the uplink and reports it back in its heartbeats."""

    def __init__(self):
        super().__init__('network_monitor_server', 'network_downlink', 'network_uplink')

    def on_period(self):
        self.get_logger().info(
            f'uplink: lat {self.incoming.latency:.1f} ms | loss {self.incoming.loss * 100:.1f} % | '
            f'{self.noise}',
            throttle_duration_sec=2.0,
        )


class NetworkMonitor(Peer):
    """Controller side: measures the downlink, gets the uplink from the bot.

    `on_update(LinkStats)` is called every period once the bot has been heard.
    `log_period=None` disables this node's own logging (the caller logs instead).
    """

    def __init__(self, on_update: Callable[[LinkStats], None], send_period: float = 0.1,
                 window: int = 100, log_period: float | None = 2.0, timeout: float = 0.5):
        super().__init__('network_monitor_client', 'network_uplink', 'network_downlink',
                         send_period, window, timeout)
        self.on_update = on_update
        self.log_period = log_period

    def on_period(self):
        if self.last_msg is None:
            if self.log_period:
                self.get_logger().warn('No heartbeat from the bot yet', throttle_duration_sec=5.0)
            return
        tau_tx, p_tx = self.last_msg.up_lat, self.last_msg.up_loss
        tau_rx, p_rx = self.incoming.latency, self.incoming.loss
        p_e2e = p_tx + p_rx - p_tx * p_rx

        if self.log_period:
            self.get_logger().info(
                f'lat tx/rx/e2e {tau_tx:.1f}/{tau_rx:.1f}/{tau_tx + tau_rx:.1f} ms | '
                f'loss tx/rx/e2e {p_tx * 100:.1f}/{p_rx * 100:.1f}/{p_e2e * 100:.1f} % | '
                f'{self.noise}',
                throttle_duration_sec=self.log_period,
            )
        self.on_update(LinkStats(
            tau_tx=tau_tx, tau_rx=tau_rx, tau_e2e=tau_tx + tau_rx,
            p_tx=p_tx, p_rx=p_rx, p_e2e=p_e2e,
            noise_kbps=self.noise.kbps, noise_loss=self.noise.loss,
        ))
