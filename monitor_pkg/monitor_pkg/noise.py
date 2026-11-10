"""Network noise generator: loads the link going OUT of the machine it runs on.

  ros2 run monitor_pkg noise kbps=1024

The noise travels as ROS messages (through the Zenoh bridge, like the real
traffic). It is received on the other machine by the link monitor, which is always
a sink: bot_monitor on the bot, controller_monitor on the controller. They log
`noise rx <kbps> (loss %)`, which is what actually crossed the link. Noise from the
same machine is ignored there.

Options (key=value, any order):
  kbps=<n>     data rate in kbit/s (1 kbit = 1000 bit), default 1024
  hz=<n>       messages per second, default 100. Message size = kbps*1000/8/hz bytes
               (1024 kbps @ 100 Hz = 1280 B per message, fits a single UDP packet).
               Many small packets = steady load; low hz = big bursty messages.

Payload is random bytes (cannot be compressed away) with host id, stream id, seq and
send time in the first 20 bytes. QoS is best effort: lost noise is not
retransmitted, like real cross traffic.
"""
import array
import os
import random
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from std_msgs.msg import UInt8MultiArray

from monitor_pkg.network_link import HOST_ID, NOISE_HEADER, NOISE_QOS, NOISE_TOPIC

LOG_PERIOD = 2.0                # s


class NoiseSender(Node):
    def __init__(self, kbps: float, hz: float):
        super().__init__('noise_sender')
        size = max(int(kbps * 1000 / 8 / hz), NOISE_HEADER.size)
        self.msg = UInt8MultiArray()
        self.msg.data = array.array('B', os.urandom(size))   # reused, only the header changes
        self.pub = self.create_publisher(UInt8MultiArray, NOISE_TOPIC, NOISE_QOS)
        self.stream = random.getrandbits(32)                  # tells restarts apart
        self.seq = 0
        self.sent_bytes = 0
        self.t_log = time.monotonic()
        self.create_timer(1.0 / hz, self.send)
        self.create_timer(LOG_PERIOD, self.log)
        self.get_logger().info(
            f'SEND {kbps:g} kbps on /{NOISE_TOPIC}: {size} B x {hz:g} Hz '
            f'(real rate {size * 8 * hz / 1000:.0f} kbps). Check what arrives in the '
            '"noise rx" of bot_monitor / controller_monitor on the other machine.')

    def send(self):
        NOISE_HEADER.pack_into(self.msg.data, 0, HOST_ID, self.stream,
                               self.seq & 0xFFFFFFFF, time.time())
        self.pub.publish(self.msg)
        self.seq += 1
        self.sent_bytes += len(self.msg.data)

    def log(self):
        now = time.monotonic()
        kbps = self.sent_bytes * 8 / 1000 / (now - self.t_log)
        line = f'sent {kbps:8.1f} kbps'
        if self.pub.get_subscription_count() == 0:
            line += ('  <- no subscriber: nothing leaves this machine '
                     '(is bot_monitor / controller_monitor and the bridge running?)')
        self.get_logger().info(line)
        self.sent_bytes, self.t_log = 0, now

    def stop(self):
        # after Ctrl+C the ROS context is already shut down: rosout is gone
        print(f'[noise_sender] STOP after {self.seq} messages', flush=True)


def parse_args(argv):
    opts = {'kbps': '1024', 'hz': '100'}
    for arg in argv:
        key, sep, value = arg.partition('=')
        if key.lower() == 'role':
            raise SystemExit('role= is gone: noise only sends now; the receiver is built '
                             'into bot_monitor / controller_monitor on the other machine.')
        if not sep or key.lower() not in opts:
            raise SystemExit(f"Bad option '{arg}'. Use: kbps=<n> hz=<n>")
        opts[key.lower()] = value
    try:
        kbps, hz = float(opts['kbps']), float(opts['hz'])
    except ValueError:
        raise SystemExit('kbps and hz must be numbers')
    if kbps <= 0 or hz <= 0:
        raise SystemExit('kbps and hz must be > 0')
    return kbps, hz


def main(args=None):
    kbps, hz = parse_args(remove_ros_args(args=sys.argv)[1:])
    rclpy.init(args=args)
    node = NoiseSender(kbps, hz)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Same clean shutdown as the main scripts: the node and its publisher are
        # destroyed, so they disappear from the ROS graph (and from the bridge).
        node.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
