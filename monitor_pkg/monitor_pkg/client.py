import time
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from link_monitor_msg.msg import LinkState


class NetworkServer(Node):
    def __init__(self):
        super().__init__('network_monitor_server')

        # Parameters (override with: --ros-args -p best_effort:=true -p window:=200)
        self.declare_parameter('best_effort', False)
        self.declare_parameter('window', 100)

        self.win = int(self.get_parameter('window').value)
        best_effort = bool(self.get_parameter('best_effort').value)

        # NOTE: the client must use a compatible QoS (best_effort <-> best_effort),
        # otherwise the connection may silently fail to match in one direction.
        if best_effort:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
        else:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )

        self.sub = self.create_subscription(LinkState, 'network_uplink', self.callback, qos)
        self.pub = self.create_publisher(LinkState, 'network_downlink', qos)

        self.expected_up_seq = 0
        # 1 = received, 0 = lost
        self.history_received = deque(maxlen=self.win)

        self.get_logger().info(
            f'network_monitor_server up (best_effort={best_effort}, window={self.win})'
        )

    def _update_loss_window(self, seq: int) -> bool:
        """Update the sliding window with the incoming seq.

        Returns True if the packet is "new" and was counted,
        False if it is a duplicate / out-of-order packet (ignored).
        """
        if seq >= self.expected_up_seq:
            gap = seq - self.expected_up_seq
            # More than `win` zeros would just overwrite the whole window anyway,
            # so cap them: avoids a huge loop on large seq jumps.
            self.history_received.extend([0] * min(gap, self.win))
            self.history_received.append(1)
            self.expected_up_seq = seq + 1
            return True

        # seq < expected: duplicate, late packet, or client restart.
        restarted = seq == 0 or (self.expected_up_seq - seq) > self.win
        if restarted:
            self.get_logger().warn(
                f'Sequence reset detected (got {seq}, expected {self.expected_up_seq}): '
                'clearing loss window'
            )
            self.history_received.clear()
            self.history_received.append(1)
            self.expected_up_seq = seq + 1
            return True

        # Late / duplicate packet that was already accounted for: ignore for loss stats.
        return False

    def callback(self, msg: LinkState):
        now = time.time()
        # One-way latency: only meaningful if the two machines' clocks are synchronized
        # (same host, or chrony/PTP). Otherwise it includes the clock offset.
        up_lat_ms = (now - msg.ts_sent) * 1000.0

        self._update_loss_window(msg.seq)

        n = len(self.history_received)
        p_up = 1.0 - (sum(self.history_received) / n) if n else 0.0

        res = LinkState()
        res.ts_sent = time.time()
        res.seq = msg.seq
        res.up_lat = up_lat_ms
        res.up_loss = float(p_up)
        self.pub.publish(res)


def main(args=None):
    rclpy.init(args=args)
    node = NetworkServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()