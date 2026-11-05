import bisect
import csv
import os
import sys
from datetime import datetime

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan

from monitor_pkg.network_link import LinkStats, NetworkMonitor


# ============================================================== TOGGLES
# Defaults; override at launch: ros2 run monitor_pkg controller_monitor SMART NOCSV
ENABLE_SMART = False    # True: adaptive threshold from network stats; False: fixed SAFE_THRESHOLD
ENABLE_CSV = True      # True: write one CSV row per scan into CSV_DIR
CSV_DIR = "/home/user/ros2_ws_A7"


class LaserProcessor(Node):
    # --- Laser ---
    K = 100                         # samples used at the beginning/end of the array

    # --- Robot parameters (MEASURED IN LAB) ---
    M = 5                           # safety margin (cm)
    BOT_LENGTH = 18.75              # bot length (cm)
    D_BRAKE = 18.1                  # braking distance (cm)
    VMAX = 43.77                    # max speed (cm/s)
    T = 50                          # control period (ms)
    SAFE_THRESHOLD = 75             # fixed brake threshold [S] (cm), used when SMART is off

    # NOTE: 0.5 m/s = 50 cm/s is ABOVE VMAX (43.77 cm/s) used for the SMART
    # threshold. Make sure the two are consistent.
    LINEAR_SPEED = 0.5              # m/s

    # --- Retransmission count N ---
    P_TARGET = 0.05
    MAX_LINEAR_STEPS = 4
    N_MAX = 200                     # hard cap (prevents endless loops if p ~ 1)

    # --- Timing ---
    SCAN_TIMEOUT_NS = 500_000_000   # 0.5 s without scans -> brake (lidar dead / driver crashed)
    STATUS_PERIOD = 1.0             # s, period of the LIDAR + NET status log

    def __init__(self, smart: bool = ENABLE_SMART, log_csv: bool = ENABLE_CSV):
        super().__init__('laser_processor')
        self.smart = smart

        # --- State ---
        self.safe_threshold = self.SAFE_THRESHOLD
        self.last_brake_state = True    # start braked until the first valid scan
        self.min_dist_cm = -1.0
        self.last_scan_ns = None
        self.scan_status = 'none'       # 'none' | 'empty' | 'invalid' | 'ok'
        self.scan_detail = ''
        self.link_seen = False
        self.current_N = 1
        self._last_sent_brake = None    # last command sent, to log only on changes

        # --- Network (placeholders until the first heartbeat from the bot) ---
        self.tau_tx = 10
        self.tau_rx = 10
        self.taue2e = self.tau_rx + self.tau_tx
        self.p_tx = 0.05
        self.p_rx = 0.05
        self.pe2e = self.p_rx + self.p_tx - (self.p_tx * self.p_rx)

        # --- ROS I/O ---
        self._msg = TwistStamped()
        self._msg.header.frame_id = "base_link"
        self.publisher_ = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(self.T / 1000.0, self.publish_cmd)
        self.status_timer = self.create_timer(self.STATUS_PERIOD, self.log_status)

        # --- CSV ---
        self.csv_file = None
        self.csv_writer = None
        if log_csv:
            self._open_csv()

        self.get_logger().info(
            f'START | SMART {"ON" if self.smart else "OFF"} | '
            f'CSV {self.csv_file.name if self.csv_file else "OFF"} | '
            f'speed {self.LINEAR_SPEED} m/s')

    # ------------------------------------------------------------------ CSV
    def _open_csv(self):
        os.makedirs(CSV_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = open(os.path.join(CSV_DIR, f"robot_log_{ts}.csv"), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp_ros', 'min_distance_cm', 'safe_threshold_cm', 'brake_state',
            'tau_tx_ms', 'tau_rx_ms', 'tau_e2e_ms', 'p_tx', 'p_rx', 'p_e2e'
        ])

    def _write_csv_row(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.csv_writer.writerow([
            f"{now:.4f}", f"{self.min_dist_cm:.2f}", f"{self.safe_threshold:.2f}",
            int(self.last_brake_state),
            f"{self.tau_tx:.2f}", f"{self.tau_rx:.2f}", f"{self.taue2e:.2f}",
            f"{self.p_tx:.4f}", f"{self.p_rx:.4f}", f"{self.pe2e:.4f}"
        ])

    def close_csv(self):
        """Idempotent: safe to call even if CSV logging is disabled."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

    # --------------------------------------------------------------- Network
    def on_link_update(self, s: LinkStats):
        """Called by NetworkMonitor once per heartbeat period (default 0.1 s).

        tx = uplink (controller -> bot), measured by the bot and reported back.
        rx = downlink (bot -> controller), measured here. If the link goes silent,
        loss rises towards 1 (counted per missed period), so current_N grows.
        """
        self.link_seen = True
        self.p_tx, self.p_rx, self.pe2e = s.p_tx, s.p_rx, s.p_e2e
        self.tau_tx, self.tau_rx, self.taue2e = s.tau_tx, s.tau_rx, s.tau_e2e
        self.update(s.p_tx, s.p_rx)

    # ----------------------------------------------------------------- Laser
    def _smart_threshold(self) -> float:
        v_cm_per_ms = self.VMAX / 1000.0
        time_buffer = self.T * (self.current_N + 1) + self.taue2e
        return v_cm_per_ms * time_buffer + self.BOT_LENGTH + self.M + self.D_BRAKE

    def scan_callback(self, msg: LaserScan):
        self.last_scan_ns = self.get_clock().now().nanoseconds

        # array.array -> numpy without copying the whole scan
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        zone = np.concatenate((ranges[:self.K], ranges[-self.K:]))
        valid = zone[(zone > msg.range_min) & (zone < msg.range_max) & np.isfinite(zone)]

        if self.smart:
            self.safe_threshold = self._smart_threshold()

        if valid.size > 0:
            self.scan_status = 'ok'
            self.min_dist_cm = float(valid.min()) * 100.0
            self.last_brake_state = self.min_dist_cm < self.safe_threshold
        else:
            self._mark_invalid_scan(zone, msg)
            self.min_dist_cm = -1.0
            self.last_brake_state = True

        if self.csv_writer is not None:
            self._write_csv_row()

    def _mark_invalid_scan(self, zone: np.ndarray, msg: LaserScan):
        if zone.size == 0:
            self.scan_status = 'empty'
            return
        self.scan_status = 'invalid'
        finite = zone[np.isfinite(zone)]
        raw = (f'measured {finite.min():.2f}..{finite.max():.2f} m'
               if finite.size else 'all inf/NaN')
        self.scan_detail = (f'{zone.size} samples, none inside '
                            f'[{msg.range_min:.2f}, {msg.range_max:.2f}] m ({raw})')

    def _scan_stale(self, now_ns: int) -> bool:
        return self.last_scan_ns is None or (now_ns - self.last_scan_ns) > self.SCAN_TIMEOUT_NS

    # -------------------------------------------------------------- Decision
    def _decision(self, now_ns: int) -> tuple[bool, str]:
        """(brake, reason). Same rule as before: brake if last scan said so or lidar is stale."""
        if self.scan_status == 'none':
            return True, 'no lidar data'
        if self._scan_stale(now_ns):
            return True, f'lidar stale ({(now_ns - self.last_scan_ns) * 1e-9:.1f} s)'
        if self.scan_status != 'ok':
            return True, f'{self.scan_status} scan'
        op = '<' if self.last_brake_state else '>='
        return self.last_brake_state, f'{self.min_dist_cm:.1f} {op} {self.safe_threshold:.1f} cm'

    def _mode_str(self) -> str:
        if self.smart:
            return f'SMART thr {self.safe_threshold:5.1f} cm (N={self.current_N})'
        return f'FIXED thr {self.safe_threshold:5.1f} cm'

    # ---------------------------------------------------------------- Status
    def log_status(self):
        """Every STATUS_PERIOD: NET, then DIST, then DECISION."""
        log = self.get_logger()
        now_ns = self.get_clock().now().nanoseconds

        # 1. Network
        if not self.link_seen:
            log.warn('NET      | no heartbeat from the bot yet')
        else:
            log.info(f'NET      | tau tx {self.tau_tx:5.1f} rx {self.tau_rx:5.1f} '
                     f'e2e {self.taue2e:5.1f} ms | '
                     f'p tx {self.p_tx * 100:4.1f}% rx {self.p_rx * 100:4.1f}% '
                     f'e2e {self.pe2e * 100:4.1f}%')

        # 2. Distance
        if self.scan_status == 'none':
            log.error('DIST     | NO DATA on /scan (topic missing or driver not running)')
        elif self._scan_stale(now_ns):
            log.warn('DIST     | lidar STALE (no recent scan)')
        elif self.scan_status == 'empty':
            log.warn('DIST     | EMPTY scan (ranges array is empty)')
        elif self.scan_status == 'invalid':
            log.warn(f'DIST     | NO VALID reading: {self.scan_detail}')
        else:
            log.info(f'DIST     | min {self.min_dist_cm:6.1f} cm')

        # 3. Decision
        brake, reason = self._decision(now_ns)
        action = 'BRAKE' if brake else f'SAFE, moving {self.LINEAR_SPEED} m/s'
        log.info(f'DECISION | {self._mode_str()} | {action} ({reason})')

    # --------------------------------------------------------------- Command
    def _publish_speed(self, speed: float):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._msg.twist.linear.x = speed
        self.publisher_.publish(self._msg)

    def publish_cmd(self):
        brake, reason = self._decision(self.get_clock().now().nanoseconds)
        self._publish_speed(0.0 if brake else self.LINEAR_SPEED)

        # Log immediately whenever the command changes (not only once per second)
        if brake != self._last_sent_brake:
            self._last_sent_brake = brake
            if brake:
                self.get_logger().warn(f'>>> BRAKE sent | {self._mode_str()} | {reason}')
            else:
                self.get_logger().info(f'>>> SAFE, moving | {self._mode_str()} | {reason}')

    def stop_robot(self):
        self._publish_speed(0.0)

    # ------------------------------------------------- Retransmission count N
    def _calculate_P_N(self, N: int, p_tx: float, p_rx: float) -> float:
        if abs(p_tx - p_rx) < 1e-6:
            p = p_tx
            return (p ** N) * (1 + (1 - p) * N)
        term_fraction = (1 - p_tx) * p_rx / (p_rx - p_tx)
        return p_tx ** N + term_fraction * (p_rx ** N - p_tx ** N)

    def _binary_search_fallback(self, p_tx: float, p_rx: float) -> int:
        # 1. Exponential bounding (bounded by N_MAX)
        high = 1
        while high < self.N_MAX and self._calculate_P_N(high, p_tx, p_rx) > self.P_TARGET:
            high *= 2
        high = min(high, self.N_MAX)

        if high == 1:
            return 1

        # 2. Binary search
        low = high // 2
        search_domain = range(low, high + 1)
        index = bisect.bisect_left(
            search_domain,
            -self.P_TARGET,
            key=lambda n: -self._calculate_P_N(n, p_tx, p_rx),
        )
        if index >= len(search_domain):
            return self.N_MAX  # target not reachable (links almost down)
        return search_domain[index]

    def update(self, p_tx: float, p_rx: float) -> int:
        temp_N = self.current_N
        steps = 0

        if self._calculate_P_N(temp_N, p_tx, p_rx) > self.P_TARGET:
            # Current N too weak
            while self._calculate_P_N(temp_N, p_tx, p_rx) > self.P_TARGET:
                temp_N += 1
                steps += 1
                if steps > self.MAX_LINEAR_STEPS:
                    self.current_N = self._binary_search_fallback(p_tx, p_rx)
                    return self.current_N
        else:
            # Current N strong enough: can we go lower?
            while temp_N > 1 and self._calculate_P_N(temp_N - 1, p_tx, p_rx) <= self.P_TARGET:
                temp_N -= 1
                steps += 1
                if steps > self.MAX_LINEAR_STEPS:
                    self.current_N = self._binary_search_fallback(p_tx, p_rx)
                    return self.current_N

        self.current_N = min(temp_N, self.N_MAX)
        return self.current_N


def parse_flags(argv):
    """Command-line toggles, e.g. `ros2 run monitor_pkg controller_monitor SMART NOCSV`.
    Anything not given falls back to ENABLE_SMART / ENABLE_CSV."""
    smart, log_csv = ENABLE_SMART, ENABLE_CSV
    for arg in argv:
        flag = arg.upper()
        if flag == 'SMART':
            smart = True
        elif flag == 'NOSMART':
            smart = False
        elif flag == 'CSV':
            log_csv = True
        elif flag == 'NOCSV':
            log_csv = False
        else:
            raise SystemExit(f"Unknown option '{arg}'. Use: SMART | NOSMART | CSV | NOCSV")
    return smart, log_csv


def main(args=None):
    # Strip --ros-args ... so only our own flags are left
    smart, log_csv = parse_flags(remove_ros_args(args=sys.argv)[1:])
    rclpy.init(args=args)

    laser_node = LaserProcessor(smart=smart, log_csv=log_csv)
    # log_period=None: the controller logs the network stats in log_status()
    network_node = NetworkMonitor(on_update=laser_node.on_link_update, log_period=None)

    executor = SingleThreadedExecutor()
    executor.add_node(laser_node)
    executor.add_node(network_node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            try:
                laser_node.stop_robot()
                executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                print(f"[Controller] Could not send stop command: {e}")

        laser_node.close_csv()
        executor.shutdown()
        laser_node.destroy_node()
        network_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
