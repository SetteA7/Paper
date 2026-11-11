# Bot side: run with `ros2 run monitor_pkg bot_monitor`.
#   - link monitor (heartbeats, latency/loss measurement)
#   - brake monitor: for each STOP, measures the braking distance with the LiDAR, logs it
#     and publishes it on BRAKE_TOPIC (the controller uses it as D_BRAKE of the next run)
from collections import deque

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from monitor_pkg.network_link import NetworkServer

BRAKE_TOPIC = '/brake_distance'     # std_msgs/Float32, d_brake in cm, one message per STOP
# One message per brake: reliable, so it is retransmitted instead of lost on a lossy link
BRAKE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=10)


class BrakeMonitor(Node):
    """d_brake = d(first STOP) - d(standstill), front LiDAR distance, bot clock only.

    first STOP   first /cmd_vel with linear.x <= 0 after forward commands, i.e. the moment
                 the STOP reaches the bot
    d(STOP)      the last scans before it are up to ~100 ms old, so a line is fitted to the
                 readings of the last FIT_S before the STOP (bot at cruise speed) and
                 evaluated at the STOP time. Each reading is timed at its own sample:
                 header.stamp + (order of the ray in time) * time_increment, see RAYS_REVERSED
    standstill   STILL_SCANS consecutive readings within STILL_SPAN_CM; d = their median
    """

    K = 10                          # samples at each end of the array (front, as controller.py)
    # sllidar_ros2 with angle_compensate and inverted: false stores the i-th measured ray in
    # ranges[n-1-i] (sllidar_node.cpp, publish_scan, reverse_data) but still reports
    # header.stamp = first measured ray and time_increment > 0. So ranges[k] was measured at
    # stamp + (n-1-k) * time_increment: index 0 is the NEWEST ray, not the oldest.
    # Timing it as stamp + k * time_increment put the front rays ~scan_time too early and
    # made d_brake ~v * scan_time (~4.3 cm) too small. False: driver follows the LaserScan
    # convention (ranges[k] at stamp + k * time_increment).
    RAYS_REVERSED = True
    FIT_S = 0.35                    # s of readings before the STOP used for d(STOP)
    STILL_SCANS = 5                 # consecutive readings ...
    STILL_SPAN_CM = 0.5             # ... within this span: standstill
    TIMEOUT_S = 3.0                 # no standstill within this time after the STOP: give up

    def __init__(self):
        super().__init__('brake_monitor')
        self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.pub = self.create_publisher(Float32, BRAKE_TOPIC, BRAKE_QOS)
        self.readings = deque(maxlen=50)    # (sample time s, front distance cm)
        self.forward = False                # last command was forward
        self.stop_t = None                  # arrival time (s) of the STOP being measured
        self.get_logger().info('brake monitor: d_brake = d(first STOP) - d(standstill), LiDAR')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg: TwistStamped):
        if self.forward and msg.twist.linear.x <= 0.0 and self.stop_t is None:
            self.stop_t = self._now()           # first STOP reached the bot
        elif msg.twist.linear.x > 0.0 and self.stop_t is not None:
            # moving again before standing still (e.g. a pause on a stale scan): that
            # STOP is not a full brake, drop it so the next STOP is measured on its own
            self.get_logger().warn('BRAKE | forward again before standstill: STOP not measured')
            self.stop_t = None
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
        k = int(idx[j])
        order = n - 1 - k if self.RAYS_REVERSED else k     # position of the ray in time
        t = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
             + order * msg.time_increment)
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
        d_brake = d_stop - d_still
        self.get_logger().info(f'BRAKE | d(STOP) {d_stop:.2f} cm ({how}) | '
                               f'd(standstill) {d_still:.2f} cm | '
                               f'd_brake {d_brake:.2f} cm -> {BRAKE_TOPIC}')
        self.pub.publish(Float32(data=float(d_brake)))


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
