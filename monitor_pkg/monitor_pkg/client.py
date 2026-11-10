# Bot side: run with `ros2 run monitor_pkg bot_monitor`.
#   - link monitor (heartbeats, latency/loss measurement)
#   - brake monitor: when a STOP reaches the bot while it drives forward, detects when
#     the wheels actually start to decelerate and logs the braking distance
import math
from collections import deque

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, LaserScan

from monitor_pkg.network_link import NetworkServer


class BrakeMonitor(Node):
    """Measures each brake on the bot itself (bot clock only, no sync needed).

    Times:
      STOP arrival  first /cmd_vel with linear.x <= 0 after forward commands, while moving
      brake onset   wheels start to decelerate: a line is fitted to the wheel speed
                    while it falls from 85% to 25% of the cruise speed and extrapolated
                    back to the cruise speed (finer than the 10 ms joint_states period)
      standstill    wheel speed below STILL_SPEED for STILL_S
    Distances (wheel odometry, /joint_states at 100 Hz):
      braking distance  travelled from brake onset to standstill (pure mechanical d_brake)
      from STOP         travelled from STOP arrival to standstill (d_brake + v * actuation)
    Wheel speed = WHEEL_RADIUS * mean |wheel rate|: for straight motion all four wheels
    turn equally, so this does not depend on the joint sign conventions.
    LiDAR check: the run's displacement from the LiDAR (front distance at rest before the
    GO minus at rest after the stop, so no timing involved) over the wheel displacement
    gives the wheel-odometry scale (slip, radius); the braking distance is also given
    scaled by it.
    """

    WHEEL_RADIUS = 0.05 * 0.98      # m, mecanum_drive_controller wheel_radius * multiplier
    K = 10                          # LiDAR samples at each end of the array (front, as controller.py)
    MOVING_SPEED = 0.05             # m/s, above this the bot is moving
    STILL_SPEED = 0.005             # m/s, below this the wheels are still
    STILL_S = 0.3                   # s still -> standstill
    REST_SCANS = 5                  # scans at standstill for the LiDAR rest distance
    CRUISE_WINDOW_S = 0.2           # s before the STOP used for the cruise speed
    SPEED_SPAN = 2                  # speed = position difference over +-2 samples (40 ms)
    HISTORY_S = 3.0                 # s of wheel samples kept

    def __init__(self):
        super().__init__('brake_monitor')
        self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joints, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)

        self.samples = deque()          # (t s, path m): wheel path length, always increasing
        self.path = 0.0
        self.last_pos = None            # np.array of wheel angles (rad), joint order fixed below
        self.joint_idx = None
        self.cmd_forward = False        # last command was a forward one
        self.stop_t = None              # STOP arrival (s) of the brake being measured
        self.still_since = None
        self.moving = False
        self.front = deque(maxlen=self.REST_SCANS)  # recent front distances (cm)
        self.rest = None                # (lidar cm, path m) at the last standstill
        self.run_start = None           # rest snapshot taken at the GO
        self.pending = None             # brake result waiting for the LiDAR rest distance
        self.get_logger().info(f'brake monitor: /cmd_vel + /joint_states + /scan '
                               f'(wheel radius {self.WHEEL_RADIUS * 100:.1f} cm)')

    # ------------------------------------------------------------- inputs
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg: TwistStamped):
        forward = msg.twist.linear.x > 0.0
        if forward and not self.cmd_forward and self.rest is not None and not self.moving:
            self.run_start = self.rest          # GO from standstill: start of a run
        if self.cmd_forward and msg.twist.linear.x <= 0.0 and self.moving \
                and self.stop_t is None:
            self.stop_t = self._now()           # STOP reached the bot
            self.pending = None
        self.cmd_forward = forward

    def on_joints(self, msg: JointState):
        if self.joint_idx is None:
            names = ['fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint']
            if not all(n in msg.name for n in names):
                return
            self.joint_idx = [list(msg.name).index(n) for n in names]
        pos = np.array([msg.position[i] for i in self.joint_idx])
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_pos is not None:
            self.path += self.WHEEL_RADIUS * float(np.mean(np.abs(pos - self.last_pos)))
        self.last_pos = pos
        self.samples.append((t, self.path))
        while self.samples and t - self.samples[0][0] > self.HISTORY_S:
            self.samples.popleft()
        self._update_motion(t)

    def on_scan(self, msg: LaserScan):
        r = np.asarray(msg.ranges, dtype=np.float32)
        zone = np.concatenate((r[:self.K], r[-self.K:]))
        valid = zone[(zone > msg.range_min) & (zone < msg.range_max) & np.isfinite(zone)]
        if valid.size:
            self.front.append(float(valid.min()) * 100.0)
        if self.still_since is not None and len(self.front) == self.REST_SCANS:
            self.rest = (float(np.median(self.front)), self.path)
            if self.pending is not None:
                self._report(self.pending)
                self.pending = None

    # ------------------------------------------------------------- motion
    def _speed(self, i: int) -> float:
        """Wheel speed (m/s) at sample i, from the path over +-SPEED_SPAN samples."""
        s = self.samples
        a, b = max(i - self.SPEED_SPAN, 0), min(i + self.SPEED_SPAN, len(s) - 1)
        dt = s[b][0] - s[a][0]
        return (s[b][1] - s[a][1]) / dt if dt > 0 else 0.0

    def _update_motion(self, t: float):
        if len(self.samples) < 2 * self.SPEED_SPAN + 1:
            return
        v = self._speed(len(self.samples) - 1 - self.SPEED_SPAN)
        if v > self.MOVING_SPEED:
            self.moving = True
            self.still_since = None
            self.front.clear()
        elif v < self.STILL_SPEED:
            if self.still_since is None:
                self.still_since = t
                self.front.clear()
            elif t - self.still_since >= self.STILL_S and self.moving:
                self.moving = False
                if self.stop_t is not None:
                    self.pending = self._analyse(self.stop_t)   # reported at the LiDAR rest
                    self.stop_t = None

    def _analyse(self, t_stop_cmd: float):
        """Brake onset, braking distance etc. of the brake whose STOP arrived at t_stop_cmd."""
        t = np.array([s[0] for s in self.samples])
        p = np.array([s[1] for s in self.samples])
        v = np.array([self._speed(i) for i in range(len(self.samples))])
        before = (t >= t_stop_cmd - self.CRUISE_WINDOW_S) & (t <= t_stop_cmd)
        if before.sum() < 3:
            self.get_logger().warn('BRAKE | not enough wheel samples before the STOP')
            return None
        v_ref = float(np.median(v[before]))
        after = np.where(t > t_stop_cmd)[0]
        # first stretch of the fall from 85% to 25% of the cruise speed
        idx = []
        for i in after:
            if 0.25 * v_ref <= v[i] <= 0.85 * v_ref:
                idx.append(i)
            elif idx:
                break                                   # end of the first stretch
        if len(idx) >= 3:
            b, a = np.polyfit(t[idx], v[idx], 1)        # v = a + b t, b < 0
            onset = (v_ref - a) / b if b < 0 else t[idx[0]]
        else:                                           # too few points: first slower sample
            slow = [i for i in after if v[i] < 0.95 * v_ref]
            if not slow:
                self.get_logger().warn('BRAKE | no deceleration found after the STOP')
                return None
            b, onset = float('nan'), t[slow[0]]
        onset = max(onset, t_stop_cmd)                 # cannot brake before the STOP arrives
        p_onset = float(np.interp(onset, t, p))
        p_stop_cmd = float(np.interp(t_stop_cmd, t, p))
        p_end = float(p[-1])
        return dict(v=v_ref, delay=onset - t_stop_cmd, d_brake=p_end - p_onset,
                    d_from_stop=p_end - p_stop_cmd, decel=-b, p_end=p_end)

    def _report(self, res: dict):
        line = (f'BRAKE | {res["v"] * 100:.1f} cm/s | decel starts {res["delay"] * 1000:.0f} ms '
                f'after the STOP | braking distance {res["d_brake"] * 100:.2f} cm '
                f'(onset -> standstill')
        line += (f', decel {res["decel"]:.2f} m/s^2)' if math.isfinite(res['decel']) else ')')
        line += f' | from STOP arrival {res["d_from_stop"] * 100:.2f} cm'
        if self.run_start is not None and self.rest is not None:
            lidar = self.run_start[0] - self.rest[0]                 # cm
            wheels = (self.rest[1] - self.run_start[1]) * 100.0      # cm
            if wheels > 1.0:
                k = lidar / wheels
                line += (f' | LiDAR check: run {lidar:.1f} cm vs wheels {wheels:.1f} cm '
                         f'(x{k:.3f}) -> braking distance {res["d_brake"] * 100 * k:.2f} cm')
        self.get_logger().info(line)
        self.run_start = None


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
