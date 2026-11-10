# Bot side: run with `ros2 run monitor_pkg bot_monitor`.
#   - link monitor (heartbeats, latency/loss measurement)
#   - velocity relay: holds the last velocity sent by the controller (cmd_vel_remote)
#     and publishes it on /cmd_vel with the bot's own timer, every CMD_PERIOD.
#     The bot's control period is then constant, not shaped by network jitter/latency.
import time

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped

from monitor_pkg.network_link import NetworkServer

CMD_TOPIC = 'cmd_vel_remote'    # from the controller (same name as in controller.py)
CMD_PERIOD = 0.1                # s, control period T on the bot (ROS param cmd_period)
CMD_TIMEOUT = 1.0               # s without commands -> stop (ROS param cmd_timeout, 0 = never)


class VelocityRelay(Node):
    def __init__(self):
        super().__init__('velocity_relay')
        period = float(self.declare_parameter('cmd_period', CMD_PERIOD).value)
        self.timeout = float(self.declare_parameter('cmd_timeout', CMD_TIMEOUT).value)

        self.last_cmd = Twist()         # last velocity received from the controller
        self.last_rx = None             # time.monotonic() of that command
        self.active = False             # publishing on /cmd_vel (after the first command)

        self._out = TwistStamped()
        self._out.header.frame_id = 'base_link'
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(TwistStamped, CMD_TOPIC, self.on_cmd, 10)
        self.create_timer(period, self.tick)
        self.get_logger().info(
            f'relay /{CMD_TOPIC} -> /cmd_vel every {period * 1000:.0f} ms '
            f'(stop after {self.timeout:g} s without commands)' if self.timeout > 0 else
            f'relay /{CMD_TOPIC} -> /cmd_vel every {period * 1000:.0f} ms (no timeout)')

    def on_cmd(self, msg: TwistStamped):
        # log only when the motion state changes (forward/stop/reverse, turning or not)
        state = lambda t: ((t.linear.x > 0) - (t.linear.x < 0), (t.angular.z > 0) - (t.angular.z < 0))
        changed = state(msg.twist) != state(self.last_cmd)
        self.last_cmd = msg.twist
        self.last_rx = time.monotonic()
        if not self.active:
            self.active = True
            self.get_logger().info('first command from the controller: relay active')
        if changed:
            self.get_logger().info(f'new velocity: linear {msg.twist.linear.x:+.2f} m/s, '
                                   f'angular {msg.twist.angular.z:+.2f} rad/s')

    def tick(self):
        """Fixed-period control step: publish the last commanded velocity."""
        if not self.active:
            return                      # never commanded: leave /cmd_vel to others
        if self.timeout > 0 and time.monotonic() - self.last_rx > self.timeout:
            # Controller gone (or link dead): stop once, then go quiet. The base
            # driver's own cmd_vel timeout keeps the bot stopped.
            self.active = False
            self.last_cmd = Twist()
            self._publish(self.last_cmd)
            self.get_logger().warn(f'no command for {self.timeout:g} s: STOP, relay idle')
            return
        self._publish(self.last_cmd)

    def _publish(self, twist: Twist):
        self._out.header.stamp = self.get_clock().now().to_msg()
        self._out.twist = twist
        self.pub.publish(self._out)

    def stop(self):
        self._publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    monitor = NetworkServer()
    relay = VelocityRelay()
    executor = SingleThreadedExecutor()
    executor.add_node(monitor)
    executor.add_node(relay)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            try:
                relay.stop()
                executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                print(f'[bot_monitor] Could not send stop command: {e}')
        executor.shutdown()
        relay.destroy_node()
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
