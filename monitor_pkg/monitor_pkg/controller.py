import bisect
import csv
import os
import random
import sys
import time
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
from std_msgs.msg import Float32

from monitor_pkg import crash_model
from monitor_pkg.client import BRAKE_QOS, BRAKE_TOPIC
from monitor_pkg.network_link import LinkStats, NetworkMonitor, ScanLoss


# ============================================================== TOGGLES
# Defaults; override at launch: ros2 run monitor_pkg controller_monitor SUPERSMART NOCSV
# Automated test: ros2 run monitor_pkg controller_monitor SMART RUNS=20  (see AutoTest)
#   = calibration run 0 (FIXED mode, measures VMAX and D_BRAKE, see CALIBRATION_RUN) + runs 1..20
# Old open-ended mode (drive/brake until Ctrl+C, no return): ... controller_monitor MANUAL
DEFAULT_RUNS = 1        # runs when RUNS= is not given (each run returns to the start)
# Threshold mode (flags FIXED | SMART | SUPERSMART):
#   FIXED       S = SAFE_THRESHOLD
#   SMART       S_high: budget T*(N*+1), safe for any waits (conservative)
#   SUPERSMART  S_smart: budget B_min, lowest S with exact P_crash <= P_TARGET, given the
#               measured psi (thesis 5.7)
# psi = wait from a scan reaching the controller to the control tick (timer, period T)
# that sends the command. Measured here on every tick; it enters the budget
# B = dt - tau_e2e - psi of SUPERSMART and of the logged P_crash (all modes).
# CSV: psi_ms = psi used by the model (PSI_QUANTILE of the last PSI_WINDOW waits, nearly
# constant), psi_sample_ms = the measured wait of the command based on that row's scan.
DEFAULT_MODE = 'FIXED'
MODES = ('FIXED', 'SMART', 'SUPERSMART')
ENABLE_CSV = True      # True: write one CSV row per scan into CSV_DIR
# Downlink latency and loss (bot -> controller) used by the MODEL (N, S, B, P_crash).
# Override at launch: USEPING=False / USEPING=True
#   True:  ping: the network node's heartbeat one-way latency (median of the last 100)
#          and loss p_rx
#   False: scans: latency = (arrival time - header.stamp) - scan_time, median of the last
#          SCAN_TAU_WINDOW scans. header.stamp is the start of the scan, so scan_time
#          (one rotation, msg.scan_time) is the acquisition, not latency, and is removed.
#          loss = p_scan: scans missing from the header.stamp sequence (gaps of more than
#          one scan period) over the last SCAN_LOSS_WINDOW scans, see ScanLoss. A whole
#          scan, lost if any of its fragments is lost on the way (zenoh bridge + Starlink)
# Logged either way: CSV tau_tx_ms / tau_rx_ms / tau_e2e_ms and p_tx / p_rx / p_e2e = the
# network node's (ping) values, tau_rx_scan_ms and p_scan = the scan values,
# tau_e2e_model_ms = the latency the model used.
# The uplink loss p_tx and latency tau_tx always come from the ping.
usePingTauRX = True
CSV_DIR = os.path.expanduser("~/ros2_ws_A7/csv")   # ~ = home of whoever runs it


class LaserProcessor(Node):
    # --- Laser ---
    K = 10                         # samples used at the beginning/end of the array

    # --- Robot parameters (MEASURED IN LAB) ---
    M = 5                           # safety margin (cm)
    BOT_LENGTH = 18.75              # bot length (cm)
    D_BRAKE = 11                 # braking distance (cm), AUTO: run 0 only (see CALIBRATION_RUN)
    VMAX = 43.5                    # max speed (cm/s), AUTO: run 0 only (see CALIBRATION_RUN)
    T = 100                         # control period (ms) SAME AS LIDAR
    SAFE_THRESHOLD = 75             # fixed brake threshold [S] (cm), used when SMART is off

    # NOTE: 0.5 m/s = 50 cm/s is ABOVE VMAX since actuators don't reach real speed
    LINEAR_SPEED = 0.5              # m/s

    # --- Retransmission count N ---
    P_TARGET = 0.05
    MAX_LINEAR_STEPS = 4
    N_MAX = 20000                   # hard cap (prevents endless loops if p ~ 1) (was 200)

    # --- Timing ---
    SCAN_TIMEOUT_NS = 500_000_000   # 0.5 s without scans -> brake (lidar dead / driver crashed)
    STATUS_PERIOD = 1.0             # s, period of the LIDAR + NET status log
    SCAN_TAU_WINDOW = 100           # scans in the median of tau_rx_scan (see usePingTauRX)
    SCAN_LOSS_WINDOW = 100          # scans in p_scan (10 s, as the heartbeats' window)

    # --- psi: wait from scan arrival to the control tick that sends the command ---
    PSI_WINDOW = 50                 # psi samples kept, one per control tick (5 s)
    PSI_QUANTILE = 0.9              # psi used by the model: this quantile of the window. High,
                                    # so if jitter makes psi wrap around a tick, the late case counts
    PSI_MIN_SAMPLES = 10            # fewer samples than this: use psi = T (worst case)

    # --- AUTO test mode (RUNS=n on the command line, see AutoTest) ---
    CRASH_DIST_CM = BOT_LENGTH      # lidar reading at/below this = crash (lidar -> front bumper)
    STOP_SETTLE_S = 2.0             # after the brake: wait, then measure where the bot stopped
    START_SETTLE_S = 2.0            # standstill at the start position before each run
    RETURN_SPEED = 0.2              # m/s, max reverse speed when going back to the start
    RETURN_MIN_SPEED = 0.05         # m/s, slowest reverse speed (below this it may not move)
    RETURN_KP = 0.01                # (m/s) per cm still to go while reversing
    RETURN_TOL_CM = 2.0             # start position reached when within this distance
    # Random start: run k starts at start + START_MARGIN_CM + U(0, VMAX*T) (VMAX*T ~ 4.5 cm,
    # one LiDAR period of travel), so S is crossed at a uniform phase: phi ~ U(0, T)
    RANDOM_START = True
    START_MARGIN_CM = 3.0           # > RETURN_TOL_CM: the bot always reverses to its start (run 1 too)
    RETURN_TIMEOUT_S = 30.0         # abort if the start position is not reached in time
    REAR_STOP_CM = 30.0             # abort reversing if anything is this close behind the lidar

    # --- Calibration (AUTO mode): before the n runs, a run 0 measures the cruise speed and
    # the bot's d_brake and sets VMAX and D_BRAKE; runs 1..n use them unchanged. Run 0 always
    # brakes in FIXED mode (S = SAFE_THRESHOLD, whatever the session mode), uses the
    # constants above, needs both values (else the session ends), and is logged as
    # run 0 (run_000.csv, run = 0 and mode = FIXED in summary.csv): drop it from the analysis ---
    CALIBRATION_RUN = True          # False: no run 0; VMAX_FROM_RUNS / DBRAKE_FROM_RUNS
                                    # update the values after every run instead

    # --- Cruise speed (AUTO mode): measured in each run (logged), see CALIBRATION_RUN ---
    VMAX_FROM_RUNS = True           # without CALIBRATION_RUN: VMAX of the next run (False: constant)
    CRUISE_FRAC = 0.9               # cruise = speed >= this fraction of the speed just before the brake
    CRUISE_MIN_SCANS = 4            # fewer cruise scans than this: speed not measured
    VMAX_MAX_CHANGE = 0.2           # measured speed > 20% away from the current VMAX: not used

    # --- Braking distance (AUTO mode): measured by the bot (client.py, BrakeMonitor) after
    # each run's brake, received on BRAKE_TOPIC (logged), see CALIBRATION_RUN ---
    DBRAKE_FROM_RUNS = True         # without CALIBRATION_RUN: D_BRAKE of the next run (False: constant)
    DBRAKE_MAX_CHANGE = 0.5         # measured d_brake > 50% away from the current D_BRAKE: not used
    DBRAKE_WAIT_S = 4.0             # stopping phase waits up to this (s from the STOP) for the bot's
                                    # value; the bot gives up 3 s after the STOP (client TIMEOUT_S)

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

    def __init__(self, mode: str = DEFAULT_MODE, log_csv: bool = ENABLE_CSV, runs: int = 0,
                 use_ping: bool = usePingTauRX):
        super().__init__('laser_processor')
        self.mode = mode
        self.use_ping = use_ping        # downlink tau of the model: ping (True) or scans
        self.budget_ms = 0.0            # B = dt - tau_e2e - psi for the current threshold
        self.p_crash = 1.0              # exact crash probability at the current threshold
        self.target_reached = True      # SUPERSMART: False if P_TARGET is unreachable

        # --- State ---
        self.safe_threshold = self.SAFE_THRESHOLD
        self.last_brake_state = True    # start braked until the first valid scan
        self.min_dist_cm = -1.0
        self.last_scan_ns = None
        self.last_lidar_ns = None       # header.stamp of the last scan (bot clock)
        self.scan_status = 'none'       # 'none' | 'empty' | 'invalid' | 'ok'
        self.scan_detail = ''
        self.link_seen = False
        self.current_N = 1
        self._last_sent_brake = None    # last command sent, to log only on changes
        self.rear_min_cm = float('inf')  # closest object behind (for reversing in AUTO mode)
        self.recent_dists = deque(maxlen=5)  # last valid front distances (cm), for medians
        self.scan_delays = deque(maxlen=self.SCAN_TAU_WINDOW)  # ms, arrival - stamp - scan_time
        self.wall_yaw = None            # rad, yaw error w.r.t. the wall normal (+ = turn left)
        self.recent_yaws = deque(maxlen=3)

        # --- psi (controller clock only, no sync needed) ---
        self.psi_samples = deque(maxlen=self.PSI_WINDOW)  # ms
        self.psi = float(self.T)        # ms, value used by the model (worst case until measured)
        self.psi_last = float('nan')    # ms, last sample: wait of the last command sent
        self._scan_arrival_mono = None  # monotonic ns of the newest scan not yet answered

        # --- Network (placeholders until the first heartbeat from the bot) ---
        # tau_tx, tau_rx, taue2e: always the network node's (ping) values
        self.tau_tx = 10
        self.tau_rx = 10
        self.taue2e = self.tau_rx + self.tau_tx
        self.tau_rx_scan = float('nan')  # ms, median of (arrival - stamp - scan_time), always
        self.p_tx = 0.05
        self.p_rx = 0.05
        self.pe2e = self.p_rx + self.p_tx - (self.p_tx * self.p_rx)
        # downlink loss of whole scans (gaps in header.stamp), always; NaN until known
        self.scan_loss = ScanLoss(self.SCAN_LOSS_WINDOW, self.SCAN_TIMEOUT_NS * 1e-9)
        self.p_scan = float('nan')

        # --- ROS I/O ---
        self._msg = TwistStamped()
        self._msg.header.frame_id = "base_link"
        self.publisher_ = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Float32, BRAKE_TOPIC, self.on_brake_distance, BRAKE_QOS)
        self.timer = self.create_timer(self.T / 1000.0, self.publish_cmd)
        self.status_timer = self.create_timer(self.STATUS_PERIOD, self.log_status)

        # --- CSV ---
        self.csv_file = None
        self.csv_writer = None
        self._pending_row = None        # row of the newest scan, written at the tick that answers it
        # AUTO mode opens one CSV per run itself
        self.auto = AutoTest(self, runs) if runs > 0 else None
        if log_csv and self.auto is None:
            self._open_csv()

        self.get_logger().info(
            f'START | mode {mode} | model downlink tau/p from {self.tau_source} | '
            f'CSV {self.csv_file.name if self.csv_file else "OFF"} | '
            f'speed {self.LINEAR_SPEED} m/s'
            + (f' | AUTO {"calibration run 0 (FIXED) + " if self.CALIBRATION_RUN else ""}'
               f'{runs} runs '
               f'-> {self.auto.session_dir}' if self.auto else ''))

    @property
    def finished(self) -> bool:
        return self.auto is not None and self.auto.finished

    @property
    def tau_source(self) -> str:
        """Where the model's downlink latency and loss come from: 'PING' or 'SCAN'."""
        return 'PING' if self.use_ping else 'SCAN'

    @property
    def tau_e2e_model(self) -> float:
        """tau_e2e used by the model (S, B, P_crash): the ping's, or with use_ping False the
        ping's uplink + the scans' downlink (ping until the first scan has arrived).
        Clock offset: ping uplink = true + theta, scan downlink = true - theta, the sum
        is still right, as for the ping alone."""
        if self.use_ping or np.isnan(self.tau_rx_scan):
            return self.taue2e
        return self.tau_tx + self.tau_rx_scan

    @property
    def p_rx_model(self) -> float:
        """Downlink loss used by the model (N, B_min, P_crash): the ping's p_rx, or with
        use_ping False the scans' p_scan (ping until p_scan is known). p_tx, p_rx, pe2e
        themselves always keep the network node's values."""
        if self.use_ping or np.isnan(self.p_scan):
            return self.p_rx
        return self.p_scan

    # ------------------------------------------------------------------ CSV
    def _open_csv(self, path: str | None = None):
        if path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(CSV_DIR, f"robot_log_{ts}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.csv_file = open(path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp_lidar', 'timestamp_arrival', 'scan_delay_ms', 'scan_time_ms',
            'min_distance_cm', 'safe_threshold_cm', 'brake_state',
            'tau_tx_ms', 'tau_rx_ms', 'tau_e2e_ms', 'p_tx', 'p_rx', 'p_e2e',
            'budget_ms', 'p_crash', 'vmax_cm_s', 'psi_ms', 'tau_rx_scan_ms', 'tau_e2e_model_ms',
            'psi_sample_ms', 'p_scan'
        ])

    def _csv_row(self, lidar_ns: int, arrival_ns: int, scan_time_ms: float) -> list:
        """Row of one scan; psi_sample_ms is known only at the next tick, so it is a None
        placeholder filled by _flush_csv_row.
        lidar = msg.header.stamp (set by the driver on the bot when the scan was taken),
        arrival = controller clock when the scan reached scan_callback.
        delay = arrival - lidar; only meaningful if both clocks are synced (chrony).
        psi_ms = psi used by the model for this row's budget_ms / p_crash."""
        return [
            f"{lidar_ns * 1e-9:.4f}", f"{arrival_ns * 1e-9:.4f}",
            f"{(arrival_ns - lidar_ns) * 1e-6:.2f}", f"{scan_time_ms:.2f}",
            f"{self.min_dist_cm:.2f}", f"{self.safe_threshold:.2f}",
            int(self.last_brake_state),
            f"{self.tau_tx:.2f}", f"{self.tau_rx:.2f}", f"{self.taue2e:.2f}",
            f"{self.p_tx:.4f}", f"{self.p_rx:.4f}", f"{self.pe2e:.4f}",
            f"{self.budget_ms:.2f}", f"{self.p_crash:.6f}", f"{self.VMAX:.2f}", f"{self.psi:.2f}",
            f"{self.tau_rx_scan:.2f}", f"{self.tau_e2e_model:.2f}",
            None,                       # psi_sample_ms (_flush_csv_row)
            f"{self.p_scan:.4f}"
        ]

    def _flush_csv_row(self, psi_sample: float | None = None):
        """Write the pending scan row. psi_sample_ms = wait from its arrival to the tick that
        sent the command based on it (controller clock only); empty if no tick answered it
        (a newer scan arrived first, or the CSV is being closed)."""
        row, self._pending_row = self._pending_row, None
        if row is not None and self.csv_writer is not None:
            row[row.index(None)] = f"{psi_sample:.2f}" if psi_sample is not None else ''
            self.csv_writer.writerow(row)

    def close_csv_run(self):
        """Close the current CSV only (AUTO mode: end of a run)."""
        self._flush_csv_row()
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        self.csv_writer = None

    def close_csv(self):
        """Idempotent: safe to call even if CSV logging is disabled."""
        self._flush_csv_row()
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
        tau_tx / tau_rx / taue2e always keep the network node's values; the model's
        latency is tau_e2e_model (see usePingTauRX).
        """
        self.link_seen = True
        self.p_tx, self.p_rx, self.pe2e = s.p_tx, s.p_rx, s.p_e2e
        self.tau_tx = s.tau_tx
        self.tau_rx = s.tau_rx
        self.taue2e = self.tau_tx + self.tau_rx
        self.update(self.p_tx, self.p_rx_model)

    def on_brake_distance(self, msg: Float32):
        """Braking distance (cm) measured by the bot (client.py) after a STOP."""
        d = float(msg.data)
        if self.auto is not None and self.auto.on_brake_distance(d):
            return                      # belongs to the current run (logged with its result)
        self.get_logger().info(f'BRAKE    | bot measured d_brake {d:.2f} cm'
                               + (' (not a run brake: ignored)' if self.auto else ''))

    # ----------------------------------------------------------------- Laser
    def _smart_threshold(self) -> float:
        v_cm_per_ms = self.VMAX / 1000.0
        time_buffer = self.T * (self.current_N + 1) + self.tau_e2e_model
        return v_cm_per_ms * time_buffer + self.BOT_LENGTH + self.M + self.D_BRAKE

    def _supersmart_threshold(self) -> float:
        """S_smart = vmax/1000 * (B_min + tau_e2e + psi) + L + m + d_brake (thesis 5.7)."""
        B, self.target_reached = crash_model.b_min(
            self.T, self.p_tx, self.p_rx_model, self.P_TARGET, self.N_MAX)
        return crash_model.threshold_from_budget(
            B, self.tau_e2e_model, self.psi, self.VMAX, self.BOT_LENGTH + self.M + self.D_BRAKE)

    def _update_threshold(self):
        """Threshold for the current mode, plus the exact P_crash it gives (all modes)."""
        if self.mode == 'SMART':
            self.safe_threshold = self._smart_threshold()
        elif self.mode == 'SUPERSMART':
            self.safe_threshold = self._supersmart_threshold()
        else:
            self.safe_threshold = self.SAFE_THRESHOLD
        self.budget_ms = crash_model.budget_from_threshold(
            self.safe_threshold, self.tau_e2e_model, self.psi, self.VMAX,
            self.BOT_LENGTH + self.M + self.D_BRAKE)
        self.p_crash = crash_model.p_crash(self.budget_ms, self.T, self.p_tx, self.p_rx_model)

    def _measure_psi(self):
        """Called at the start of every control tick: if a scan arrived since the last
        tick, psi = now - its arrival is the wait of the command sent now.
        psi is constant while the nodes run (LiDAR and timer both have period T), up to
        latency jitter; the model uses a high quantile of the last samples."""
        if self._scan_arrival_mono is None:
            return
        self.psi_last = (time.monotonic_ns() - self._scan_arrival_mono) * 1e-6
        self._scan_arrival_mono = None
        self._flush_csv_row(self.psi_last)          # that scan's row gets its own wait
        self.psi_samples.append(self.psi_last)
        if len(self.psi_samples) >= self.PSI_MIN_SAMPLES:
            self.psi = float(np.quantile(self.psi_samples, self.PSI_QUANTILE))

    def scan_callback(self, msg: LaserScan):
        self._scan_arrival_mono = time.monotonic_ns()   # for psi (newest scan wins)
        arrival_ns = self.get_clock().now().nanoseconds
        self.last_scan_ns = arrival_ns
        lidar_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.last_lidar_ns = lidar_ns
        # Downlink loss of whole scans from gaps in header.stamp (end to end, see ScanLoss),
        # always logged; the model uses it only with use_ping False (see p_rx_model)
        self.scan_loss.on_message(lidar_ns * 1e-9, msg.scan_time or self.T / 1000.0,
                                  self._scan_arrival_mono * 1e-9)
        self.p_scan = self.scan_loss.loss
        # Downlink latency measured on the real sensor data (needs synced clocks), always
        # logged; the model uses it only with use_ping False (see tau_e2e_model):
        # tau_rx_scan = (now - msg.header.stamp) - scan_time
        # NOTE: HERE !!! HERE IS THE SCAN TIME DOUBT
        self.scan_delays.append((arrival_ns - lidar_ns) * 1e-6 - msg.scan_time * 1000.0)
        self.tau_rx_scan = float(np.median(self.scan_delays))

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

        # Row held until the tick that answers this scan fills psi_sample_ms
        self._flush_csv_row()                       # previous scan never got a tick of its own
        if self.csv_writer is not None:
            self._pending_row = self._csv_row(lidar_ns, arrival_ns, msg.scan_time * 1000.0)

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
                    f'(B={self.budget_ms:.0f} ms, psi={self.psi:.0f} ms, {pc}{warn})')
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
            log.info(f'NET      | ping tau tx {self.tau_tx:5.1f} rx {self.tau_rx:5.1f} '
                     f'e2e {self.taue2e:5.1f} ms | scan rx {self.tau_rx_scan:5.1f} ms | '
                     f'model e2e {self.tau_e2e_model:5.1f} ms ({self.tau_source}) | '
                     f'p tx {self.p_tx * 100:4.1f}% rx {self.p_rx * 100:4.1f}% '
                     f'e2e {self.pe2e * 100:4.1f}% | scan p rx {self.p_scan * 100:4.1f}% | '
                     f'model p rx {self.p_rx_model * 100:4.1f}%')
        if len(self.psi_samples) < self.PSI_MIN_SAMPLES:
            log.info(f'PSI      | {len(self.psi_samples)} samples, using T = {self.psi:.0f} ms')
        else:
            q10, q50 = np.quantile(self.psi_samples, [0.1, 0.5])
            line = (f'PSI      | used {self.psi:5.1f} ms (q{self.PSI_QUANTILE * 100:.0f}) | '
                    f'median {q50:5.1f} | spread q10..q{self.PSI_QUANTILE * 100:.0f} '
                    f'{self.psi - q10:4.1f} ms')
            if self.psi - q10 > self.T / 2:
                log.warn(line + ' | psi wraps around a tick (latency jitter near a tick)')
            else:
                log.info(line)

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
        """Control step, every T ms."""
        self._measure_psi()
        self.scan_loss.on_tick(time.monotonic())    # counts scans lost while /scan is silent
        self.p_scan = self.scan_loss.loss
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


def cruise_speed(t_s, d_cm, frac: float, min_scans: int) -> tuple[float | None, int]:
    """Average speed (cm/s) over the constant-speed part of an approach.

    t_s, d_cm: lidar stamp (s) and front distance (cm) of each valid scan of the
    approach, up to and including the scan that triggered the brake (the bot is
    still at full speed when that scan is taken: braking starts only after it).
    Walking back from the brake, scans belong to the cruise while the speed over
    two intervals stays >= frac * the speed just before the brake; this drops the
    acceleration from standstill. The speed is the least-squares slope of distance
    vs time over the cruise scans, which averages out the lidar noise.
    Returns (speed, cruise scans), or (None, scans) if fewer than min_scans or if
    the bot was still accelerating (second half of the window faster than the
    first by more than 1/frac: the start was too close to the threshold).
    """
    t = np.asarray(t_s, dtype=np.float64)
    d = np.asarray(d_cm, dtype=np.float64)
    if t.size < max(min_scans, 4):
        return None, 0
    dt = t[2:] - t[:-2]
    if np.any(dt <= 0):
        return None, 0
    s = (d[:-2] - d[2:]) / dt              # s[k]: speed over scans k .. k+2
    v_ref = float(np.median(s[-3:]))
    if v_ref <= 0:
        return None, 0
    k = s.size - 1
    while k > 0 and s[k - 1] >= frac * v_ref:
        k -= 1
    n = t.size - k                          # cruise scans: k .. end
    if n < min_scans:
        return None, n
    m = k + (n - 1) // 2                    # split the window in two halves
    v_first = (d[k] - d[m]) / (t[m] - t[k])
    v_second = (d[m] - d[-1]) / (t[-1] - t[m])
    if v_first <= 0 or v_second > v_first / frac:
        return None, n
    slope = np.polyfit(t[k:] - t[k], d[k:], 1)[0]
    return (float(-slope), n) if slope < 0 else (None, n)


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
    Random start (RANDOM_START): each run starts at start + START_MARGIN_CM + U(0, VMAX*T),
    reached by the return phase (also before run 1). VMAX*T is the travel in one LiDAR
    period, so the crossing of S falls at a uniform phase of the period and phi is
    U(0, T) in every run. The return stops within RETURN_TOL_CM of the target, but that
    error is independent of the random offset, so phi stays uniform. Without it, phi is
    fixed by the start position and the runs are not independent.
    The session stops after `runs` runs, or at the first crash
    (closest front reading during the run <= CRASH_DIST_CM).
    Cruise speed: at the brake, the speed of the constant-speed part of the approach
    is measured (see cruise_speed).
    Braking distance: after the brake the bot (client.py) measures d_brake with its LiDAR
    and publishes it on BRAKE_TOPIC; the stopping phase waits for it (up to DBRAKE_WAIT_S).
    Calibration (CALIBRATION_RUN): a run 0 before the `runs` runs, same procedure, always in
    FIXED mode (S = SAFE_THRESHOLD; the session mode is set back after it), uses the
    VMAX and D_BRAKE constants; its cruise speed and d_brake become VMAX and D_BRAKE of
    runs 1..n, which then stay fixed (both must be measured and pass the *_MAX_CHANGE
    checks, else the session ends). Without it, VMAX_FROM_RUNS / DBRAKE_FROM_RUNS update
    them after every run instead. Every run logs its own measured values in the summary.

    Output in CSV_DIR/auto_<mode>_<time>/:
      run_000.csv      calibration run (CALIBRATION_RUN)
      run_001.csv ...  same columns as the manual CSV, approach + stopping only
      summary.csv      one row per run (run 0 = calibration)
    """

    SUMMARY_HEADER = [
        'run', 'csv_file', 'start_dist_cm', 'brake_dist_cm', 'safe_threshold_cm', 'N_at_brake',
        'tau_e2e_ms_at_brake', 'p_e2e_at_brake', 'final_dist_cm', 'min_dist_cm',
        'stop_distance_cm', 'margin_to_crash_cm', 'crashed', 'mode', 'budget_ms_at_brake',
        'p_crash_at_brake', 'align_err_deg',
        # robot / model parameters used for this run (constants of LaserProcessor)
        'VMAX_cm_s', 'T_ms', 'D_BRAKE_cm', 'L_cm', 'M_cm', 'V_CMD_cm_s',
        # cruise speed measured in this run, scans it was fitted on (VMAX of runs 1..n if
        # this is the calibration run 0)
        'v_cruise_cm_s', 'cruise_scans',
        # psi used by the model at the brake, and the measured wait of the STOP itself
        'psi_ms_at_brake', 'psi_stop_ms',
        # start position this run was driven to (start + margin + U(0, VMAX*T) if RANDOM_START)
        'start_target_cm',
        # braking distance measured by the bot in this run (D_BRAKE of runs 1..n if run 0)
        'd_brake_meas_cm',
        # downlink latency and loss source of the model (PING or SCAN, flag USEPING=) and the
        # tau_e2e the model used at the brake (tau_e2e_ms_at_brake is always the ping's)
        'tau_source', 'tau_e2e_model_ms_at_brake',
    ]

    def __init__(self, node: 'LaserProcessor', runs: int):
        self.node = node
        self.runs = runs                # measured runs 1..runs
        self.calibration = node.CALIBRATION_RUN
        self.run = -1 if self.calibration else 0   # +1 when a run starts: run 0 = calibration
        self.finished = False
        self.start_dist = None
        self.brake = None               # values recorded at the brake decision
        self.run_min = float('inf')
        self.track = []                 # (lidar stamp s, front dist cm) of the approach scans
        self.results = []               # (run, final_dist, crashed)
        self.align_err = float('nan')   # deg, residual yaw error after aligning
        self.align_ok_since = None
        self.target = None              # cm, start position of the next run (return phase goal)
        self.run_target = None          # cm, start position of the current run
        self.d_brake_meas = None        # cm, d_brake the bot measured for this run's brake
        self._enter('align')

        mode = node.mode
        self.session_mode = mode        # mode of runs 1..n
        if self.calibration:
            node.mode = 'FIXED'         # run 0 (and the phases before it) in FIXED mode
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
        done = sum(1 for r, _, _ in self.results if r > 0)
        crashes = sum(1 for _, _, c in self.results if c)
        cal = ' + calibration' if any(r == 0 for r, _, _ in self.results) else ''
        self._log().info(f'AUTO DONE | {why} | {done} runs{cal}, {crashes} crash(es) | '
                         f'results in {self.session_dir}')

    def _run_name(self) -> str:
        if self.calibration and self.run <= 0:
            return 'run 0 (calibration)'
        return f'run {self.run}/{self.runs}'

    def close(self):
        if not self.summary_file.closed:
            self.summary_file.flush()
            self.summary_file.close()

    def status(self) -> str:
        start = f'{self.start_dist:.1f} cm' if self.start_dist is not None else '?'
        if self.target is not None and self.target != self.start_dist:
            start += f' (next run from {self.target:.1f} cm)'
        yaw = self.node.wall_yaw
        wall = f'{np.degrees(yaw):+.2f} deg' if yaw is not None else '?'
        return (f'{self._run_name()} | phase {self.phase} | start dist {start} | '
                f'wall angle {wall}')

    # ---------------------------------------------------------------- hooks
    def on_brake_distance(self, d: float) -> bool:
        """d_brake from the bot: kept if it is the brake of the current run (we are in the
        stopping phase, the only STOP sent there is the run's brake). Returns True if kept."""
        if self.phase != 'stopping' or self.d_brake_meas is not None:
            return False
        self.d_brake_meas = d
        return True

    def on_scan(self):
        """Track the closest front reading while the run is being recorded."""
        if self.phase in ('approach', 'stopping') and self.node.scan_status == 'ok':
            self.run_min = min(self.run_min, self.node.min_dist_cm)
            if self.phase == 'approach':
                # lidar stamps: spaced by the scan period, free of network jitter
                self.track.append((self.node.last_lidar_ns * 1e-9, self.node.min_dist_cm))

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
            self._pick_start()
            if n.RANDOM_START:
                self._enter('return')   # drive to the random start of the first run first
                return
        if dist <= n.safe_threshold:
            self._log().warn(f'AUTO | start {dist:.1f} cm is already below the threshold '
                             f'{n.safe_threshold:.1f} cm: the bot will brake immediately')

        self.run += 1
        self.run_start = dist
        self.run_target = self.target
        self.run_min = float('inf')
        self.track = []
        self.brake = None
        n._open_csv(os.path.join(self.session_dir, f'run_{self.run:03d}.csv'))
        n._last_sent_brake = None       # log the first command of the run
        self._log().info(f'AUTO | ===== {self._run_name()} start ({dist:.1f} cm) =====')
        self._enter('approach')

    def _step_approach(self):
        n = self.node
        brake, reason = n.drive_by_decision()
        # Only a distance-based brake ends the run; stale/invalid scans just pause it
        if brake and n.scan_fresh_ok() and n.last_brake_state:
            # on_scan ran before this step, so the track already holds the brake scan
            t, d = zip(*self.track) if self.track else ((), ())
            v, n_cruise = cruise_speed(t, d, n.CRUISE_FRAC, n.CRUISE_MIN_SCANS)
            self.brake = dict(dist=n.min_dist_cm, thr=n.safe_threshold, N=n.current_N,
                              B=n.budget_ms, pc=n.p_crash,
                              tau=n.taue2e, tau_model=n.tau_e2e_model,
                              p=n.pe2e, v=v, n_cruise=n_cruise,
                              # psi_last was measured at the start of this tick: the
                              # wait of the STOP that is being sent now
                              psi=n.psi, psi_stop=n.psi_last)
            cruise = (f'cruise {v:.2f} cm/s over {n_cruise} scans' if v is not None else
                      f'cruise speed not measured (no steady part of >= '
                      f'{n.CRUISE_MIN_SCANS} scans before the brake)')
            self._log().info(f'AUTO | run {self.run} braked ({reason}) | {cruise}')
            self.d_brake_meas = None
            self._enter('stopping')

    def _step_stopping(self):
        n = self.node
        n.stop_robot()
        if self._elapsed() < self.node.STOP_SETTLE_S or not n.scan_fresh_ok():
            return
        if self.d_brake_meas is None and self._elapsed() < n.DBRAKE_WAIT_S:
            return                      # the bot's d_brake for this brake is not in yet
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
            f'{b["v"]:.2f}' if b['v'] is not None else '', b['n_cruise'],
            f'{b["psi"]:.2f}', f'{b["psi_stop"]:.2f}',
            f'{self.run_target:.2f}' if self.run_target is not None else '',
            f'{self.d_brake_meas:.2f}' if self.d_brake_meas is not None else '',
            n.tau_source, f'{b["tau_model"]:.2f}',
        ])
        self.summary_file.flush()
        n.close_csv_run()
        calib_failed = self._update_params(b)
        self.results.append((self.run, final, crashed))
        dbm = f'{self.d_brake_meas:.2f} cm' if self.d_brake_meas is not None else '?'
        self._log().info(f'AUTO | {self._run_name()} result: braked at {b["dist"]:.1f} cm, '
                         f'stopped at {final:.1f} cm, closest {self.run_min:.1f} cm, '
                         f'bot d_brake {dbm}'
                         f'{" -> CRASH" if crashed else ""}')
        if crashed:
            self._finish(f'CRASH in run {self.run}')
        elif calib_failed:
            self._finish(f'calibration failed: {calib_failed}. Check the setup (bot_monitor '
                         'running? start far enough to reach cruise speed?) or the VMAX / '
                         'D_BRAKE constants, then restart')
        else:
            if n.mode != self.session_mode:
                n.mode = self.session_mode  # end of the FIXED calibration run 0
                self._log().info(f'AUTO | runs 1-{self.runs} in {n.mode} mode')
            self._pick_start()          # after _update_params: the span uses the new VMAX
            self._enter('return')

    def _update_params(self, b: dict) -> str | None:
        """VMAX and D_BRAKE from the cruise speed and the bot's d_brake measured in this run.

        CALIBRATION_RUN: only run 0 sets them, both are required, and runs 1..n keep them.
        Returns why the calibration failed, or None.
        Otherwise (VMAX_FROM_RUNS / DBRAKE_FROM_RUNS) every run updates them for the next
        one, and a value that can't be used only gives a warning (returns None).
        """
        n = self.node
        calibrating = self.calibration and self.run == 0
        if self.calibration and not calibrating:
            return None                 # runs 1..n: the calibrated values stay fixed
        use = f'runs 1-{self.runs}' if calibrating else 'the next run'
        problems = []                   # (what went wrong, value kept instead)
        if calibrating or n.VMAX_FROM_RUNS:
            v = b['v']
            keep = f'VMAX stays {n.VMAX:.2f} cm/s'
            if v is None:
                problems.append(('cruise speed not measured', keep))
            elif abs(v - n.VMAX) > n.VMAX_MAX_CHANGE * n.VMAX:
                problems.append((f'cruise speed {v:.2f} cm/s is more than '
                                 f'{n.VMAX_MAX_CHANGE:.0%} away from VMAX {n.VMAX:.2f}', keep))
            else:
                self._log().info(f'AUTO | VMAX for {use}: {v:.2f} cm/s (was {n.VMAX:.2f})')
                n.VMAX = v              # instance value, overrides the class constant
        if calibrating or n.DBRAKE_FROM_RUNS:
            d = self.d_brake_meas
            keep = f'D_BRAKE stays {n.D_BRAKE:.2f} cm'
            if d is None:
                problems.append((f'no d_brake from the bot within {n.DBRAKE_WAIT_S:g} s '
                                 '(bot_monitor running?)', keep))
            elif d <= 0 or abs(d - n.D_BRAKE) > n.DBRAKE_MAX_CHANGE * n.D_BRAKE:
                problems.append((f'measured d_brake {d:.2f} cm is more than '
                                 f'{n.DBRAKE_MAX_CHANGE:.0%} away from D_BRAKE {n.D_BRAKE:.2f}',
                                 keep))
            else:
                self._log().info(f'AUTO | D_BRAKE for {use}: {d:.2f} cm (was {n.D_BRAKE:.2f})')
                n.D_BRAKE = d           # instance value, overrides the class constant
        if calibrating:
            return '; '.join(p for p, _ in problems) or None
        for p, keep in problems:
            self._log().warn(f'AUTO | {p}: not used, {keep}')
        return None

    def _pick_start(self):
        """Start position of the next run: start + START_MARGIN_CM + U(0, VMAX*T) with
        RANDOM_START (VMAX in cm/s, T in ms: VMAX*T/1000 cm = travel in one period),
        else the start distance itself."""
        n = self.node
        if not n.RANDOM_START:
            self.target = self.start_dist
            return
        span = n.VMAX * n.T / 1000.0
        self.target = self.start_dist + n.START_MARGIN_CM + random.uniform(0.0, span)
        self._log().info(f'AUTO | next start {self.target:.2f} cm '
                         f'(start {self.start_dist:.1f} + {n.START_MARGIN_CM:g} + U(0, {span:.2f}))')

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
        to_go = self.target - n.min_dist_cm
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
    `RUNS=n` runs the automated test (AutoTest) n times (default DEFAULT_RUNS), after a
    calibration run 0 if CALIBRATION_RUN.
    `MANUAL` = old open-ended mode: no runs, no return, until Ctrl+C.
    FIXED | SMART | SUPERSMART pick the threshold mode (NOSMART = FIXED).
    `USEPING=True|False` sets usePingTauRX: the model's downlink latency and loss from the
    ping or from the scans.
    Anything not given falls back to DEFAULT_MODE / ENABLE_CSV / DEFAULT_RUNS / usePingTauRX."""
    mode, log_csv, runs, use_ping = DEFAULT_MODE, ENABLE_CSV, DEFAULT_RUNS, usePingTauRX
    for arg in argv:
        flag = arg.upper()
        if flag.startswith('USEPING='):
            value = flag[len('USEPING='):]
            if value in ('TRUE', '1', 'YES', 'ON'):
                use_ping = True
            elif value in ('FALSE', '0', 'NO', 'OFF'):
                use_ping = False
            else:
                raise SystemExit(f"Bad option '{arg}': use USEPING=True or USEPING=False")
        elif flag == 'MANUAL':
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
                "| RUNS=<n> | MANUAL | USEPING=True|False")
    return mode, log_csv, runs, use_ping


def main(args=None):
    # Strip --ros-args ... so only our own flags are left
    mode, log_csv, runs, use_ping = parse_flags(remove_ros_args(args=sys.argv)[1:])
    rclpy.init(args=args)

    laser_node = LaserProcessor(mode=mode, log_csv=log_csv, runs=runs, use_ping=use_ping)
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
