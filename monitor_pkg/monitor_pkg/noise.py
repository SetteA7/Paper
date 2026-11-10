"""Network noise generator: loads the link going OUT of the machine it runs on.

DDS/Zenoh only put data on the network if a subscriber on the OTHER machine is
matched. A sender is also a sink, so:

  noise in one direction:   sender:  ros2 run monitor_pkg noise kbps=1024
                            other:   ros2 run monitor_pkg noise role=sink
  noise in both directions: run `noise kbps=...` on both machines (no extra sink)

Options (key=value, any order):
  kbps=<n>     data rate in kbit/s (1 kbit = 1000 bit), default 1024
  hz=<n>       messages per second, default 100. Message size = kbps*1000/8/hz bytes
               (1024 kbps @ 100 Hz = 1280 B per message, fits a single UDP packet).
               Many small packets = steady load; low hz = big bursty messages
               (DDS fragments anything > ~64 KB).
  role=send|sink   send = send + receive (default), sink = receive only
  topic=<name> default network_noise (must be the same on both machines)

Payload is random bytes (cannot be compressed away) with sender id + seq + send
time in the first 16 bytes; a node ignores its own messages. QoS is best effort:
lost noise is not retransmitted, like real cross traffic. Every 2 s the node logs
what it sent and what it received from the other machine(s), with loss
(missing sequence numbers, per sender).
"""
import array
import os
import random
import struct
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args

from std_msgs.msg import UInt8MultiArray

HEADER = struct.Struct('<IId')  # sender id, seq (uint32), send time (double) = 16 bytes
LOG_PERIOD = 2.0                # s
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


class NoiseNode(Node):
    """Receives noise from the other machine(s); if kbps is given, also sends it."""

    def __init__(self, topic: str, kbps: float | None = None, hz: float = 100.0):
        super().__init__('noise_sender' if kbps else 'noise_sink')
        self.my_id = random.getrandbits(32)
        self.create_subscription(UInt8MultiArray, topic, self.on_msg, QOS)
        self.rx_bytes = self.rx_msgs = self.rx_lost = 0
        self.expected = {}              # sender id -> next expected seq
        self.pub = None
        self.tx_bytes = 0
        if kbps:
            size = max(int(kbps * 1000 / 8 / hz), HEADER.size)
            self.msg = UInt8MultiArray()
            self.msg.data = array.array('B', os.urandom(size))  # reused, header changes
            self.pub = self.create_publisher(UInt8MultiArray, topic, QOS)
            self.seq = 0
            self.create_timer(1.0 / hz, self.send)
            self.get_logger().info(
                f'SEND + RECEIVE on /{topic}: {kbps:g} kbps = {size} B x {hz:g} Hz '
                f'(real rate {size * 8 * hz / 1000:.0f} kbps)')
        else:
            self.get_logger().info(f'RECEIVE only on /{topic}')
        self.t_log = time.monotonic()
        self.create_timer(LOG_PERIOD, self.log)

    def send(self):
        HEADER.pack_into(self.msg.data, 0, self.my_id, self.seq & 0xFFFFFFFF, time.time())
        self.pub.publish(self.msg)
        self.seq += 1
        self.tx_bytes += len(self.msg.data)

    def on_msg(self, msg: UInt8MultiArray):
        if len(msg.data) < HEADER.size:
            return
        sender, seq, _ = HEADER.unpack_from(msg.data, 0)
        if sender == self.my_id:
            return                      # our own noise, delivered locally
        exp = self.expected.get(sender)
        if exp is not None and seq > exp:
            self.rx_lost += seq - exp
        if exp is None or seq >= exp:
            self.expected[sender] = seq + 1
        self.rx_msgs += 1
        self.rx_bytes += len(msg.data)

    def log(self):
        now = time.monotonic()
        dt = now - self.t_log
        total = self.rx_msgs + self.rx_lost
        loss = 100.0 * self.rx_lost / total if total else 0.0
        line = (f'received {self.rx_bytes * 8 / 1000 / dt:8.1f} kbps | '
                f'{self.rx_msgs} msgs | loss {loss:4.1f} %')
        if self.pub is not None:
            # our own subscription is matched too: remote receivers = count - 1
            remote = max(self.pub.get_subscription_count() - 1, 0)
            line = (f'sent {self.tx_bytes * 8 / 1000 / dt:8.1f} kbps (receivers {remote}) | '
                    + line)
            if remote == 0:
                line += '  <- no receiver: nothing goes on the network, start noise there'
        self.get_logger().info(line)
        self.tx_bytes = self.rx_bytes = self.rx_msgs = self.rx_lost = 0
        self.t_log = now


def parse_args(argv):
    opts = {'kbps': '1024', 'hz': '100', 'role': 'send', 'topic': 'network_noise'}
    for arg in argv:
        key, sep, value = arg.partition('=')
        if not sep or key.lower() not in opts:
            raise SystemExit(f"Bad option '{arg}'. Use: kbps=<n> hz=<n> role=send|sink "
                             'topic=<name>')
        opts[key.lower()] = value
    try:
        kbps, hz = float(opts['kbps']), float(opts['hz'])
    except ValueError:
        raise SystemExit('kbps and hz must be numbers')
    if kbps <= 0 or hz <= 0:
        raise SystemExit('kbps and hz must be > 0')
    if opts['role'] not in ('send', 'sink'):
        raise SystemExit('role must be send or sink')
    return kbps, hz, opts['role'], opts['topic']


def main(args=None):
    kbps, hz, role, topic = parse_args(remove_ros_args(args=sys.argv)[1:])
    rclpy.init(args=args)
    node = NoiseNode(topic, kbps, hz) if role == 'send' else NoiseNode(topic)
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
