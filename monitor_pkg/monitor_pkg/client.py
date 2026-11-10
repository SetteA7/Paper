# Bot side: run with `ros2 run monitor_pkg bot_monitor`.
#   - link monitor (heartbeats, latency/loss measurement)
#   - brake monitor: for each STOP, logs the braking distance measured by the LiDAR
from collections import deque

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan

from monitor_pkg.network_link import NetworkServer


class BrakeMonitor(Node):
    """d_brake = d(first STOP) - d(standstill), front LiDAR distance, bot clock only.

    first STOP   first /cmd_vel with linear.x <= 0 after forward commands, i.e. the moment
                 the STOP reaches the bot
    d(STOP)      the last scans before it are up to ~100 ms old, so a line is fitted to the
                 readings of the last FIT_S before the STOP (bot at cruise speed) and
                 evaluated at the STOP time. Each reading is timed at its own sample:
                 header.stamp + index * time_increment
    standstill   STILL_SCANS consecutive readings within STILL_SPAN_CM; d = their median
    """

    K = 10                          # samples at each end of the array (front, as controller.py)
    FIT_S = 0.35                    # s of readings before the STOP used for d(STOP)
    STILL_SCANS = 5                 # consecutive readings ...
    STILL_SPAN_CM = 0.5             # ... within this span: standstill
    TIMEOUT_S = 3.0                 # no standstill within this time after the STOP: give up

    def __init__(self):
        super().__init__('brake_monitor')
        self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.readings = deque(maxlen=50)    # (sample time s, front distance cm)
        self.forward = False                # last command was forward
        self.stop_t = None                  # arrival time (s) of the STOP being measured
        self.get_logger().info('brake monitor: d_brake = d(first STOP) - d(standstill), LiDAR')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg: TwistStamped):
        if self.forward and msg.twist.linear.x <= 0.0 and self.stop_t is None:
            self.stop_t = self._now()           # first STOP reached the bot
        self.forward = msg.twist.linear.x > 0.0

    def on_scan(self, msg: LaserScan):
        r = np.asarray(msg.ranges, dtype=np.float32)
        n = r.size
        idx = np.r_[0:self.K, n - self.K:n]
        zone = r[idx]
        ok = (zone > msg.range_min) & (zone < msg.range_max) & np.isfinite(zone)
        if not ok.any():
            return
        j = int(np.argmin(np.where(ok, zone, np.inf)))
        t = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
             + int(idx[j]) * msg.time_increment)
        self.readings.append((t, float(zone[j]) * 100.0))
        if self.stop_t is not None:
            self._check_standstill()

    def _check_standstill(self):
        after = [d for t, d in self.readings if t > self.stop_t]
        if len(after) >= self.STILL_SCANS:
            last = after[-self.STILL_SCANS:]
            if max(last) - min(last) <= self.STILL_SPAN_CM:
                self._report(float(np.median(last)))
                self.stop_t = None
                return
        if self._now() - self.stop_t > self.TIMEOUT_S:
            self.get_logger().warn('BRAKE | no standstill within '
                                   f'{self.TIMEOUT_S:g} s after the STOP')
            self.stop_t = None

    def _report(self, d_still: float):
        before = [(t, d) for t, d in self.readings
                  if self.stop_t - self.FIT_S <= t <= self.stop_t]
        if len(before) >= 2:
            t, d = np.array(before).T
            slope, icpt = np.polyfit(t - self.stop_t, d, 1)
            d_stop = float(icpt)                    # line evaluated at the STOP time
            how = (f'fit of {len(before)} scans, last one '
                   f'{(self.stop_t - t[-1]) * 1000:.0f} ms before, {-slope:.1f} cm/s')
        elif before:
            d_stop = before[-1][1]
            how = 'last scan only'
        else:
            self.get_logger().warn('BRAKE | no LiDAR reading before the STOP')
            return
        self.get_logger().info(f'BRAKE | d(STOP) {d_stop:.2f} cm ({how}) | '
                               f'd(standstill) {d_still:.2f} cm | '
                               f'd_brake {d_stop - d_still:.2f} cm')


def main(args=None):
    rclpy.init(args=args)
    monitor = NetworkServer()
    brake = BrakeMonitor()
    executor = SingleThreadedExecutor()
    executor.add_node(monitor)
    executor.add_node(brake)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        brake.destroy_node()
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
