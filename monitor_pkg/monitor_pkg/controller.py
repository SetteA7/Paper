import bisect
import csv
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan

from monitor_pkg import crash_model
from monitor_pkg.network_link import LinkStats, NetworkMonitor


# ============================================================== TOGGLES
# Defaults; override at launch: ros2 run monitor_pkg controller_monitor SUPERSMART NOCSV
# Automated test: ros2 run monitor_pkg controller_monitor SMART RUNS=20  (see AutoTest)
# Old open-ended mode (drive/brake until Ctrl+C, no return): ... controller_monitor MANUAL
DEFAULT_RUNS = 1        # runs when RUNS= is not given (each run returns to the start)
# Threshold mode (flags FIXED | SMART | SUPERSMART):
#   FIXED       S = SAFE_THRESHOLD
#   SMART       S_high: budget T*(N*+1), safe for any waits (conservative)
#   SUPERSMART  S_smart: budget B_min, lowest S with exact P_crash <= P_TARGET
DEFAULT_MODE = 'FIXED'
MODES = ('FIXED', 'SMART', 'SUPERSMART')
ENABLE_CSV = True      # True: write one CSV row per scan into CSV_DIR
# Downlink latency tau_rx (bot -> controller):
#   True:  from the network ping (heartbeat one-way latency, median of the last 100)
#   False: from the LiDAR scans: arrival time - header.stamp, median of the last
#          SCAN_TAU_WINDOW scans (includes driver + middleware delay of the real data)
# Loss p_rx and the uplink tau_tx always come from the ping.
usePingTauRX = True
# Velocity command to the bot. The bot (bot_monitor) holds the last value and publishes
# it on its own /cmd_vel at a fixed period, so network jitter/latency do not change the
# bot's control period. The controller answers each scan with one command.
CMD_TOPIC = 'cmd_vel_remote'
CSV_DIR = os.path.expanduser("~/ros2_ws_A7/csv")   # ~ = home of whoever runs it


class LaserProcessor(Node):
    # --- Laser ---
    K = 100                         # samples used at the beginning/end of the array

    # --- Robot parameters (MEASURED IN LAB) ---
    M = 5                           # safety margin (cm)
    BOT_LENGTH = 18.75              # bot length (cm)
    D_BRAKE = 12.59                  # braking distance (cm)
    VMAX = 43.84                    # max speed (cm/s)
    T = 100                         # control period (ms) SAME AS LIDAR
    SAFE_THRESHOLD = 75             # fixed brake threshold [S] (cm), used when SMART is off

    # NOTE: 0.5 m/s = 50 cm/s is ABOVE VMAX (43.77 cm/s) used for the SMART
    # threshold. Make sure the two are consistent.
    LINEAR_SPEED = 0.5              # m/s

    # --- Retransmission count N ---
    P_TARGET = 0.05
    MAX_LINEAR_STEPS = 4
    N_MAX = 20000                     # hard cap (prevents endless loops if p ~ 1) (was 200)

    # --- Timing ---
    SCAN_TIMEOUT_NS = 500_000_000   # 0.5 s without scans -> brake (lidar dead / driver crashed)
    STALE_CHECK_PERIOD = 0.1        # s, how often to check for stale scans (then send stop)
    STATUS_PERIOD = 1.0             # s, period of the LIDAR + NET status log
    SCAN_TAU_WINDOW = 100           # scans in the median for tau_rx when usePingTauRX = False

    # --- AUTO test mode (RUNS=n on the command line, see AutoTest) ---
    CRASH_DIST_CM = BOT_LENGTH      # lidar reading at/below this = crash (lidar -> front bumper)
    STOP_SETTLE_S = 2.0             # after the brake: wait, then measure where the bot stopped
    START_SETTLE_S = 2.0            # standstill at the start position before each run
    RETURN_SPEED = 0.2              # m/s, max reverse speed when going back to the start
    RETURN_MIN_SPEED = 0.05         # m/s, slowest reverse speed (below this it may not move)
    RETURN_KP = 0.01                # (m/s) per cm still to go while reversing
    RETURN_TOL_CM = 2.0             # start position reached when within this distance
    RETURN_TIMEOUT_S = 30.0         # abort if the start position is not reached in time
    REAR_STOP_CM = 30.0             # abort reversing if anything is this close behind the lidar

    # --- Alignment before each AUTO run: turn in place until perpendicular to the wall ---
    ALIGN_ENABLED = True
    ALIGN_FIT_DEG = 30.0            # max half-width of the fitted wall segment (deg)
    ALIGN_JUMP_CM = 5.0             # range jump between neighbours = edge of the surface
    ALIGN_TOL_DEG = 0.5             # aligned when |yaw error| <= this
    ALIGN_KP = 1.5                  # (rad/s) per rad of yaw error
    ALIGN_MAX_W = 0.4               # rad/s, max turn speed
    ALIGN_MIN_W = 0.05              # rad/s, slowest turn (below this the bot may not turn)
    ALIGN_SETTLE_S = 0.6            # standstill re-check: error must stay within tolerance
    ALIGN_TIMEOUT_S = 30.0          # abort the session if not aligned in time
    ALIGN_MAX_DEG = 15.0            # abort if the error grows past this (turning the wrong way?)
    ALIGN_SIGN = 1                  # set to -1 if the bot turns AWAY from perpendicular

    def __init__(self, mode: str = DEFAULT_MODE, log_csv: bool = ENABLE_CSV, runs: int = 0):
        super().__init__('laser_processor')
        self.mode = mode
        self.budget_ms = 0.0            # B = dt - tau_e2e for the current threshold
        self.p_crash = 1.0              # exact crash probability at the current threshold
        self.target_reached = True      # SUPERSMART: False if P_TARGET is unreachable

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
        self.rear_min_cm = float('inf')  # closest object behind (for reversing in AUTO mode)
        self.recent_dists = deque(maxlen=5)  # last valid front distances (cm), for medians
        self.scan_delays = deque(maxlen=self.SCAN_TAU_WINDOW)  # ms, arrival - header.stamp
        self.wall_yaw = None            # rad, yaw error w.r.t. the wall normal (+ = turn left)
        self.recent_yaws = deque(maxlen=3)

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
        self.publisher_ = self.create_publisher(TwistStamped, CMD_TOPIC, 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        # Commands are sent from scan_callback (one per scan). This timer only covers
        # the case where scans stop arriving: then it keeps sending stop.
        self.stale_timer = self.create_timer(self.STALE_CHECK_PERIOD, self._stale_watchdog)
        self.status_timer = self.create_timer(self.STATUS_PERIOD, self.log_status)

        # --- CSV ---
        self.csv_file = None
        self.csv_writer = None
        # AUTO mode opens one CSV per run itself
        self.auto = AutoTest(self, runs) if runs > 0 else None
        if log_csv and self.auto is None:
            self._open_csv()

        self.get_logger().info(
            f'START | mode {self.mode} | tau_rx from {"PING" if usePingTauRX else "LIDAR"} | '
            f'CSV {self.csv_file.name if self.csv_file else "OFF"} | '
            f'speed {self.LINEAR_SPEED} m/s'
            + (f' | AUTO {runs} runs -> {self.auto.session_dir}' if self.auto else ''))

    @property
    def finished(self) -> bool:
        return self.auto is not None and self.auto.finished

    # ------------------------------------------------------------------ CSV
    def _open_csv(self, path: str | None = None):
        if path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(CSV_DIR, f"robot_log_{ts}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.csv_file = open(path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp_lidar', 'timestamp_arrival', 'scan_delay_ms', 'min_distance_cm', 'safe_threshold_cm', 'brake_state',
            'tau_tx_ms', 'tau_rx_ms', 'tau_e2e_ms', 'p_tx', 'p_rx', 'p_e2e',
            'budget_ms', 'p_crash'
        ])

    def _write_csv_row(self, lidar_ns: int, arrival_ns: int):
        """lidar = msg.header.stamp (set by the driver on the bot when the scan was taken),
        arrival = controller clock when the scan reached scan_callback.
        delay = arrival - lidar; only meaningful if both clocks are synced (chrony)."""
        self.csv_writer.writerow([
            f"{lidar_ns * 1e-9:.4f}", f"{arrival_ns * 1e-9:.4f}",
            f"{(arrival_ns - lidar_ns) * 1e-6:.2f}", f"{self.min_dist_cm:.2f}", f"{self.safe_threshold:.2f}",
            int(self.last_brake_state),
            f"{self.tau_tx:.2f}", f"{self.tau_rx:.2f}", f"{self.taue2e:.2f}",
            f"{self.p_tx:.4f}", f"{self.p_rx:.4f}", f"{self.pe2e:.4f}",
            f"{self.budget_ms:.2f}", f"{self.p_crash:.6f}"
        ])

    def close_csv_run(self):
        """Close the current CSV only (AUTO mode: end of a run)."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        self.csv_writer = None

    def close_csv(self):
        """Idempotent: safe to call even if CSV logging is disabled."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        self.csv_writer = None
        if self.auto is not None:
            self.auto.close()

    # --------------------------------------------------------------- Network
    def on_link_update(self, s: LinkStats):
        """Called by NetworkMonitor once per heartbeat period (default 0.1 s).

        tx = uplink (controller -> bot), measured by the bot and reported back.
        rx = downlink (bot -> controller), measured here. If the link goes silent,
        loss rises towards 1 (counted per missed period), so current_N grows.
        """
        self.link_seen = True
        self.p_tx, self.p_rx, self.pe2e = s.p_tx, s.p_rx, s.p_e2e
        self.tau_tx = s.tau_tx
        if usePingTauRX:
            self.tau_rx = s.tau_rx
        self.taue2e = self.tau_tx + self.tau_rx
        self.update(s.p_tx, s.p_rx)

    # ----------------------------------------------------------------- Laser
    def _smart_threshold(self) -> float:
        v_cm_per_ms = self.VMAX / 1000.0
        time_buffer = self.T * (self.current_N + 1) + self.taue2e
        return v_cm_per_ms * time_buffer + self.BOT_LENGTH + self.M + self.D_BRAKE

    def _supersmart_threshold(self) -> float:
        """S_smart = vmax/1000 * (B_min + tau_e2e) + L + m + d_brake (thesis 4.7)."""
        B, self.target_reached = crash_model.b_min(
            self.T, self.p_tx, self.p_rx, self.P_TARGET, self.N_MAX)
        return crash_model.threshold_from_budget(
            B, self.taue2e, self.VMAX, self.BOT_LENGTH + self.M + self.D_BRAKE)

    def _update_threshold(self):
        """Threshold for the current mode, plus the exact P_crash it gives (all modes)."""
        if self.mode == 'SMART':
            self.safe_threshold = self._smart_threshold()
        elif self.mode == 'SUPERSMART':
            self.safe_threshold = self._supersmart_threshold()
        self.budget_ms = crash_model.budget_from_threshold(
            self.safe_threshold, self.taue2e, self.VMAX, self.BOT_LENGTH + self.M + self.D_BRAKE)
        self.p_crash = crash_model.p_crash(self.budget_ms, self.T, self.p_tx, self.p_rx)

    def scan_callback(self, msg: LaserScan):
        arrival_ns = self.get_clock().now().nanoseconds
        self.last_scan_ns = arrival_ns
        lidar_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if not usePingTauRX:
            # Downlink latency measured on the real sensor data (needs synced clocks)
            self.scan_delays.append((arrival_ns - lidar_ns) * 1e-6)
            self.tau_rx = float(np.median(self.scan_delays))
            self.taue2e = self.tau_tx + self.tau_rx

        # array.array -> numpy without copying the whole scan
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        zone = np.concatenate((ranges[:self.K], ranges[-self.K:]))
        valid = zone[(zone > msg.range_min) & (zone < msg.range_max) & np.isfinite(zone)]

        self._update_threshold()

        if valid.size > 0:
            self.scan_status = 'ok'
            self.min_dist_cm = float(valid.min()) * 100.0
            self.last_brake_state = self.min_dist_cm < self.safe_threshold
            self.recent_dists.append(self.min_dist_cm)
        else:
            self._mark_invalid_scan(zone, msg)
            self.min_dist_cm = -1.0
            self.last_brake_state = True

        # Rear sector (around the middle of the array), used only when reversing
        mid = ranges.size // 2
        rear = ranges[max(mid - self.K, 0):mid + self.K]
        rear = rear[(rear > msg.range_min) & (rear < msg.range_max) & np.isfinite(rear)]
        self.rear_min_cm = float(rear.min()) * 100.0 if rear.size else float('inf')

        if self.auto is not None and self.ALIGN_ENABLED:
            self.wall_yaw = self._fit_wall_yaw(ranges, msg)
            if self.wall_yaw is not None:
                self.recent_yaws.append(self.wall_yaw)

        if self.auto is not None:
            self.auto.on_scan()

        if self.csv_writer is not None:
            self._write_csv_row(lidar_ns, arrival_ns)

        # Answer this scan immediately with the velocity to hold
        self.publish_cmd()

    def _fit_wall_yaw(self, ranges: np.ndarray, msg: LaserScan) -> float | None:
        """Yaw error (rad) between the robot heading and the normal of the surface ahead.

        Index 0 of the scan is straight ahead and the end of the array wraps around to
        it (same convention as the front sector), so sample i is at angle i*inc for the
        start of the array and (i - n)*inc for the end.
        Only the surface the bot faces is used: starting straight ahead, the segment is
        grown in both directions until a range jump > ALIGN_JUMP_CM (its edge) or
        ALIGN_FIT_DEG. Points beyond the edges (things further away) would wreck the fit.
        Fits x = a + b*y (x forward, y left); perpendicular when b = 0.
        Surface rotated by alpha (robot must turn +alpha): b = -tan(alpha).
        """
        n = ranges.size
        inc = abs(msg.angle_increment)
        if n < 20 or inc <= 0:
            return None
        k = min(int(np.radians(self.ALIGN_FIT_DEG) / inc), n // 2 - 1)
        idx = np.r_[n - k:n, 0:k + 1]               # ordered: -k .. 0 .. +k
        r = ranges[idx].astype(np.float64)
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        c = k                                       # position of straight ahead
        if not ok[c]:
            return None
        jump = self.ALIGN_JUMP_CM / 100.0
        lo = c
        while lo > 0 and ok[lo - 1] and abs(r[lo - 1] - r[lo]) <= jump:
            lo -= 1
        hi = c
        while hi < 2 * k and ok[hi + 1] and abs(r[hi + 1] - r[hi]) <= jump:
            hi += 1
        if hi - lo + 1 < 20:
            return None
        th = (np.arange(lo, hi + 1) - c) * msg.angle_increment
        rr = r[lo:hi + 1]
        x = rr * np.cos(th)
        y = rr * np.sin(th)
        b, a = np.polyfit(y, x, 1)
        # one pass of outlier rejection (noise, small objects on the surface)
        res = np.abs(x - (a + b * y))
        keep = res <= max(3.0 * float(np.median(res)), 0.01)
        if keep.sum() < 20:
            return None
        b, a = np.polyfit(y[keep], x[keep], 1)
        return float(-np.arctan(b))

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
        pc = f'Pcrash {self.p_crash:.4f}'
        if self.mode == 'SMART':
            return f'SMART thr {self.safe_threshold:5.1f} cm (N={self.current_N}, {pc})'
        if self.mode == 'SUPERSMART':
            warn = '' if self.target_reached else ', TARGET UNREACHABLE'
            return (f'SUPERSMART thr {self.safe_threshold:5.1f} cm '
                    f'(B={self.budget_ms:.0f} ms, {pc}{warn})')
        return f'FIXED thr {self.safe_threshold:5.1f} cm ({pc})'

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

        # 3. Decision (in AUTO mode it only drives the bot during the approach)
        if self.auto is None or self.auto.phase == 'approach':
            brake, reason = self._decision(now_ns)
            action = 'BRAKE' if brake else f'SAFE, moving {self.LINEAR_SPEED} m/s'
            log.info(f'DECISION | {self._mode_str()} | {action} ({reason})')

        # 4. Automated test
        if self.auto is not None:
            log.info(f'AUTO     | {self.auto.status()}')

    # --------------------------------------------------------------- Command
    def _publish_speed(self, speed: float, turn: float = 0.0):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._msg.twist.linear.x = speed
        self._msg.twist.angular.z = turn
        self.publisher_.publish(self._msg)

    def publish_cmd(self):
        """Control step: called for every scan, sends the velocity the bot must hold."""
        if self.auto is not None:
            self.auto.step()
        else:
            self.drive_by_decision()

    def scan_fresh_ok(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        return self.scan_status == 'ok' and not self._scan_stale(now_ns)

    def drive_by_decision(self) -> tuple[bool, str]:
        """The braking experiment itself: move forward unless the decision says brake."""
        brake, reason = self._decision(self.get_clock().now().nanoseconds)
        self._publish_speed(0.0 if brake else self.LINEAR_SPEED)

        # Log immediately whenever the command changes (not only once per second)
        if brake != self._last_sent_brake:
            self._last_sent_brake = brake
            if brake:
                self.get_logger().warn(f'>>> BRAKE sent | {self._mode_str()} | {reason}')
            else:
                self.get_logger().info(f'>>> SAFE, moving | {self._mode_str()} | {reason}')
        return brake, reason

    def stop_robot(self):
        self._publish_speed(0.0)

    def _stale_watchdog(self):
        """No scans for SCAN_TIMEOUT: nothing triggers publish_cmd, so send stop here
        (the bot would otherwise hold the last velocity)."""
        if self._scan_stale(self.get_clock().now().nanoseconds):
            if self.auto is None:
                self.drive_by_decision()    # brakes and logs the reason
            else:
                self.stop_robot()

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


class AutoTest:
    """Automated braking test: repeats the manual procedure `runs` times.

    Each run:
      align    turn in place until perpendicular to the wall (ALIGN_*), re-checked at rest
      wait     standstill START_SETTLE_S at the start position
      approach drive forward with the normal braking logic (same as manual mode),
               until the first distance-based brake (min dist < threshold)
      stopping send 0 for STOP_SETTLE_S, then measure where the bot stopped
      return   reverse (forward/backward only) to the start distance
    The start distance is the front distance measured before run 1.
    The session stops after `runs` runs, or at the first crash
    (closest front reading during the run <= CRASH_DIST_CM).

    Output in CSV_DIR/auto_<mode>_<time>/:
      run_001.csv ...  same columns as the manual CSV, approach + stopping only
      summary.csv      one row per run
    """

    SUMMARY_HEADER = [
        'run', 'csv_file', 'start_dist_cm', 'brake_dist_cm', 'safe_threshold_cm', 'N_at_brake',
        'tau_e2e_ms_at_brake', 'p_e2e_at_brake', 'final_dist_cm', 'min_dist_cm',
        'stop_distance_cm', 'margin_to_crash_cm', 'crashed', 'mode', 'budget_ms_at_brake',
        'p_crash_at_brake', 'align_err_deg',
        # robot / model parameters used for this run (constants of LaserProcessor)
        'VMAX_cm_s', 'T_ms', 'D_BRAKE_cm', 'L_cm', 'M_cm', 'V_CMD_cm_s',
    ]

    def __init__(self, node: 'LaserProcessor', runs: int):
        self.node = node
        self.runs = runs
        self.run = 0
        self.finished = False
        self.start_dist = None
        self.brake = None               # values recorded at the brake decision
        self.run_min = float('inf')
        self.results = []               # (run, final_dist, crashed)
        self.align_err = float('nan')   # deg, residual yaw error after aligning
        self.align_ok_since = None
        self._enter('align')

        mode = node.mode
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(CSV_DIR, f'auto_{mode}_{ts}')
        os.makedirs(self.session_dir, exist_ok=True)
        self.summary_file = open(os.path.join(self.session_dir, 'summary.csv'), 'w', newline='')
        self.summary = csv.writer(self.summary_file)
        self.summary.writerow(self.SUMMARY_HEADER)
        self.summary_file.flush()

    # -------------------------------------------------------------- helpers
    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _enter(self, phase: str):
        self.phase = phase
        self.t0 = self._now()

    def _elapsed(self) -> float:
        return self._now() - self.t0

    def _median_dist(self) -> float:
        return float(np.median(self.node.recent_dists))

    def _log(self):
        return self.node.get_logger()

    def _finish(self, why: str):
        self.node.stop_robot()
        self.node.close_csv()           # closes the run CSV and the summary
        self.finished = True
        crashes = sum(1 for _, _, c in self.results if c)
        self._log().info(f'AUTO DONE | {why} | {len(self.results)} runs, {crashes} crash(es) | '
                         f'results in {self.session_dir}')

    def close(self):
        if not self.summary_file.closed:
            self.summary_file.flush()
            self.summary_file.close()

    def status(self) -> str:
        start = f'{self.start_dist:.1f} cm' if self.start_dist is not None else '?'
        yaw = self.node.wall_yaw
        wall = f'{np.degrees(yaw):+.2f} deg' if yaw is not None else '?'
        return (f'run {self.run}/{self.runs} | phase {self.phase} | start dist {start} | '
                f'wall angle {wall}')

    # ---------------------------------------------------------------- hooks
    def on_scan(self):
        """Track the closest front reading while the run is being recorded."""
        if self.phase in ('approach', 'stopping') and self.node.scan_status == 'ok':
            self.run_min = min(self.run_min, self.node.min_dist_cm)

    def step(self):
        """Called every control period T instead of the manual-mode command."""
        if self.finished:
            self.node.stop_robot()
            return
        getattr(self, f'_step_{self.phase}')()

    # --------------------------------------------------------------- phases
    def _step_align(self):
        n = self.node
        if not n.ALIGN_ENABLED:
            self._enter('wait')
            return
        if self._elapsed() > n.ALIGN_TIMEOUT_S:
            yaw = n.wall_yaw
            self._finish('could not align to the wall within '
                         f'{n.ALIGN_TIMEOUT_S:.0f} s (last error '
                         f'{np.degrees(yaw) if yaw is not None else float("nan"):+.2f} deg)')
            return
        if not n.scan_fresh_ok() or n.wall_yaw is None:
            n.stop_robot()
            self.align_ok_since = None
            return
        tol = np.radians(n.ALIGN_TOL_DEG)
        err = n.wall_yaw
        if abs(err) > np.radians(n.ALIGN_MAX_DEG):
            self._finish(f'wall angle {np.degrees(err):+.1f} deg is beyond ALIGN_MAX_DEG: '
                         'place the bot roughly facing the wall, or flip ALIGN_SIGN if it '
                         'turned the wrong way')
            return
        if abs(err) <= tol:
            # Stop, then require the error to stay in tolerance at rest
            n.stop_robot()
            now = self._now()
            if self.align_ok_since is None:
                self.align_ok_since = now
            elif (now - self.align_ok_since >= n.ALIGN_SETTLE_S
                  and all(abs(e) <= tol for e in n.recent_yaws)):
                self.align_err = float(np.degrees(np.median(n.recent_yaws)))
                self._log().info(f'AUTO | aligned to the wall ({self.align_err:+.2f} deg)')
                self.align_ok_since = None
                self._enter('wait')
            return
        self.align_ok_since = None
        w = min(max(n.ALIGN_KP * abs(err), n.ALIGN_MIN_W), n.ALIGN_MAX_W)
        n._publish_speed(0.0, n.ALIGN_SIGN * (w if err > 0 else -w))

    def _step_wait(self):
        n = self.node
        n.stop_robot()
        if self._elapsed() < self.node.START_SETTLE_S or not n.scan_fresh_ok() \
                or len(n.recent_dists) < n.recent_dists.maxlen or not n.link_seen:
            return
        dist = self._median_dist()
        if self.start_dist is None:
            self.start_dist = dist
            self._log().info(f'AUTO | start distance set to {dist:.1f} cm')
        if dist <= n.safe_threshold:
            self._log().warn(f'AUTO | start {dist:.1f} cm is already below the threshold '
                             f'{n.safe_threshold:.1f} cm: the bot will brake immediately')

        self.run += 1
        self.run_start = dist
        self.run_min = float('inf')
        self.brake = None
        n._open_csv(os.path.join(self.session_dir, f'run_{self.run:03d}.csv'))
        n._last_sent_brake = None       # log the first command of the run
        self._log().info(f'AUTO | ===== run {self.run}/{self.runs} start ({dist:.1f} cm) =====')
        self._enter('approach')

    def _step_approach(self):
        n = self.node
        brake, reason = n.drive_by_decision()
        # Only a distance-based brake ends the run; stale/invalid scans just pause it
        if brake and n.scan_fresh_ok() and n.last_brake_state:
            self.brake = dict(dist=n.min_dist_cm, thr=n.safe_threshold, N=n.current_N,
                              B=n.budget_ms, pc=n.p_crash,
                              tau=n.taue2e, p=n.pe2e)
            self._log().info(f'AUTO | run {self.run} braked ({reason})')
            self._enter('stopping')

    def _step_stopping(self):
        n = self.node
        n.stop_robot()
        if self._elapsed() < self.node.STOP_SETTLE_S or not n.scan_fresh_ok():
            return
        final = self._median_dist()
        crashed = self.run_min <= self.node.CRASH_DIST_CM
        b = self.brake
        self.summary.writerow([
            self.run, f'run_{self.run:03d}.csv', f'{self.run_start:.2f}',
            f'{b["dist"]:.2f}', f'{b["thr"]:.2f}', b['N'], f'{b["tau"]:.2f}', f'{b["p"]:.4f}',
            f'{final:.2f}', f'{self.run_min:.2f}', f'{b["dist"] - final:.2f}',
            f'{self.run_min - self.node.CRASH_DIST_CM:.2f}', int(crashed),
            n.mode, f'{b["B"]:.2f}', f'{b["pc"]:.6f}', f'{self.align_err:+.3f}',
            f'{n.VMAX:g}', f'{n.T:g}', f'{n.D_BRAKE:g}', f'{n.BOT_LENGTH:g}', f'{n.M:g}',
            f'{n.LINEAR_SPEED * 100:g}',
        ])
        self.summary_file.flush()
        n.close_csv_run()
        self.results.append((self.run, final, crashed))
        self._log().info(f'AUTO | run {self.run} result: braked at {b["dist"]:.1f} cm, '
                         f'stopped at {final:.1f} cm, closest {self.run_min:.1f} cm'
                         f'{" -> CRASH" if crashed else ""}')
        if crashed:
            self._finish(f'CRASH in run {self.run}')
        else:
            self._enter('return')

    def _step_return(self):
        n = self.node
        if self._elapsed() > self.node.RETURN_TIMEOUT_S:
            self._finish(f'start position not reached within {self.node.RETURN_TIMEOUT_S:.0f} s')
            return
        if not n.scan_fresh_ok():
            n.stop_robot()              # wait for valid scans
            return
        if n.rear_min_cm < self.node.REAR_STOP_CM:
            self._finish(f'obstacle {n.rear_min_cm:.1f} cm behind while reversing '
                         f'(limit REAR_STOP_CM = {self.node.REAR_STOP_CM:.0f} cm)')
            return
        to_go = self.start_dist - n.min_dist_cm
        if to_go <= self.node.RETURN_TOL_CM:
            n.stop_robot()
            if self.run >= self.runs:
                self._finish(f'all {self.runs} runs completed')
            else:
                self._enter('align')
            return
        speed = min(max(self.node.RETURN_KP * to_go, self.node.RETURN_MIN_SPEED), self.node.RETURN_SPEED)
        n._publish_speed(-speed)


def parse_flags(argv):
    """Command-line toggles, e.g. `ros2 run monitor_pkg controller_monitor SMART NOCSV`.
    `RUNS=n` runs the automated test (AutoTest) n times (default DEFAULT_RUNS).
    `MANUAL` = old open-ended mode: no runs, no return, until Ctrl+C.
    FIXED | SMART | SUPERSMART pick the threshold mode (NOSMART = FIXED).
    Anything not given falls back to DEFAULT_MODE / ENABLE_CSV."""
    mode, log_csv, runs = DEFAULT_MODE, ENABLE_CSV, DEFAULT_RUNS
    for arg in argv:
        flag = arg.upper()
        if flag == 'MANUAL':
            runs = 0
        elif flag.startswith('RUNS='):
            try:
                runs = int(flag[5:])
            except ValueError:
                raise SystemExit(f"Bad option '{arg}': use RUNS=<number>")
            if runs < 1:
                raise SystemExit('RUNS must be >= 1')
        elif flag in MODES:
            mode = flag
        elif flag == 'NOSMART':
            mode = 'FIXED'
        elif flag == 'CSV':
            log_csv = True
        elif flag == 'NOCSV':
            log_csv = False
        else:
            raise SystemExit(
                f"Unknown option '{arg}'. Use: FIXED | SMART | SUPERSMART | CSV | NOCSV "
                "| RUNS=<n> | MANUAL")
    return mode, log_csv, runs


def main(args=None):
    # Strip --ros-args ... so only our own flags are left
    mode, log_csv, runs = parse_flags(remove_ros_args(args=sys.argv)[1:])
    rclpy.init(args=args)

    laser_node = LaserProcessor(mode=mode, log_csv=log_csv, runs=runs)
    # log_period=None: the controller logs the network stats in log_status()
    network_node = NetworkMonitor(on_update=laser_node.on_link_update, log_period=None)

    executor = SingleThreadedExecutor()
    executor.add_node(laser_node)
    executor.add_node(network_node)

    try:
        # AUTO mode ends by itself (all runs done / crash); manual mode runs until Ctrl+C
        while rclpy.ok() and not laser_node.finished:
            executor.spin_once(timeout_sec=0.1)
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
