"""Network noise generator: loads the link going OUT of the machine it runs on.

DDS/Zenoh only put data on the network if a subscriber on the OTHER machine is
matched, so run the sink there:

  machine that should send (uplink noise):  python3 noise.py kbps=1024
  the other machine:                        python3 noise.py role=sink

Options (key=value, any order):
  kbps=<n>     data rate in kbit/s (1 kbit = 1000 bit), default 1024
  hz=<n>       messages per second, default 100. Message size = kbps*1000/8/hz bytes
               (1024 kbps @ 100 Hz = 1280 B per message, fits a single UDP packet).
               Many small packets = steady load; low hz = big bursty messages
               (DDS fragments anything > ~64 KB).
  role=send|sink
  topic=<name> default network_noise. Use different topics for noise in both
               directions at once (e.g. topic=noise_ul and topic=noise_dl).

Payload is random bytes (cannot be compressed away) with seq + send time in the
first 12 bytes. QoS is best effort: lost noise is not retransmitted, like real
cross traffic. Both roles log the measured rate every 2 s; the sink also logs loss
(missing sequence numbers).
"""
import array
import os
import struct
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args

from std_msgs.msg import UInt8MultiArray

HEADER = struct.Struct('<Id')   # seq (uint32), send time (double) = 12 bytes
LOG_PERIOD = 2.0                # s
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


class NoiseSender(Node):
    def __init__(self, kbps: float, hz: float, topic: str):
        super().__init__('noise_sender')
        size = max(int(kbps * 1000 / 8 / hz), HEADER.size)
        self.msg = UInt8MultiArray()
        self.msg.data = array.array('B', os.urandom(size))   # reused, only the header changes
        self.pub = self.create_publisher(UInt8MultiArray, topic, QOS)
        self.seq = 0
        self.sent_bytes = 0
        self.t_log = time.monotonic()
        self.create_timer(1.0 / hz, self.send)
        self.create_timer(LOG_PERIOD, self.log)
        self.get_logger().info(
            f'SEND {kbps:g} kbps on /{topic}: {size} B x {hz:g} Hz '
            f'(real rate = {size * 8 * hz / 1000:.0f} kbps)')

    def send(self):
        HEADER.pack_into(self.msg.data, 0, self.seq & 0xFFFFFFFF, time.time())
        self.pub.publish(self.msg)
        self.seq += 1
        self.sent_bytes += len(self.msg.data)

    def log(self):
        now = time.monotonic()
        kbps = self.sent_bytes * 8 / 1000 / (now - self.t_log)
        subs = self.pub.get_subscription_count()
        self.get_logger().info(
            f'sent {kbps:8.1f} kbps | subscribers {subs}'
            + ('  <- no receiver: nothing goes on the network, start role=sink there'
               if subs == 0 else ''))
        self.sent_bytes, self.t_log = 0, now


class NoiseSink(Node):
    def __init__(self, topic: str):
        super().__init__('noise_sink')
        self.create_subscription(UInt8MultiArray, topic, self.on_msg, QOS)
        self.bytes = self.received = self.lost = 0
        self.expected = None
        self.t_log = time.monotonic()
        self.create_timer(LOG_PERIOD, self.log)
        self.get_logger().info(f'SINK on /{topic}')

    def on_msg(self, msg: UInt8MultiArray):
        seq, _ = HEADER.unpack_from(msg.data, 0)
        if self.expected is not None and seq > self.expected:
            self.lost += seq - self.expected
        if self.expected is None or seq >= self.expected or seq < 100:  # (re)start
            self.expected = seq + 1
        self.received += 1
        self.bytes += len(msg.data)

    def log(self):
        now = time.monotonic()
        kbps = self.bytes * 8 / 1000 / (now - self.t_log)
        total = self.received + self.lost
        loss = 100.0 * self.lost / total if total else 0.0
        self.get_logger().info(f'received {kbps:8.1f} kbps | {self.received} msgs | '
                               f'loss {loss:4.1f} %')
        self.bytes = self.received = self.lost = 0
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
    node = NoiseSender(kbps, hz, topic) if role == 'send' else NoiseSink(topic)
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
