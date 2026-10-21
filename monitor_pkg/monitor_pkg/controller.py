import bisect
import csv
import os
import statistics
import time
from collections import deque
from datetime import datetime

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import TwistStamped
from link_monitor_msg.msg import LinkState
from sensor_msgs.msg import LaserScan


class LaserProcessor(Node):
    # Hard cap for the retransmission search (prevents endless loops if p ~ 1)
    N_MAX = 200
    # If no scan arrives for this long, brake (lidar dead / driver crashed)
    SCAN_TIMEOUT_NS = 500_000_000  # 0.5 s

    def __init__(self):
        super().__init__('laser_processor')

        # --- Laser ---
        self.k = 100                # samples used at the beginning/end of the array

        # --- Robot parameters ---
        self.m = 5                  # safety margin (cm)
        self.bot_length = 18.75     # bot length (cm) MEASURED IN LAB
        self.d_brake = 18.1         # braking distance (cm) MEASURED IN LAB
        self.vmax = 43.77           # max speed (cm/s) MEASURED IN LAB
        self.T = 50                 # control period (ms)
        self.safe_threshold = 75    # brake threshold [S] (cm)

        # --- Static configuration ---
        # NOTE: 0.5 m/s = 50 cm/s is ABOVE vmax (43.77 cm/s) used for the SMART
        # threshold. Make sure the two are consistent.
        self.linear_speed = 0.5     # m/s
        self.SMART = False

        # Start braked: don't move until the first valid scan arrives
        self.last_brake_state = True
        self.min_dist_cm = -1.0
        self.last_scan_ns = None

        self.P_crash_target = 0.05
        self.P_target = self.P_crash_target
        self.max_linear_steps = 4
        self.current_N = 1

        # --- Network variables ---
        self.tau_tx = 10
        self.tau_rx = 10
        self.taue2e = self.tau_rx + self.tau_tx

        self.p_tx = 0.05
        self.p_rx = 0.05
        self.pe2e = self.p_rx + self.p_tx - (self.p_tx * self.p_rx)

        # --- Pre-allocated message ---
        self._msg = TwistStamped()
        self._msg.header.frame_id = "base_link"

        # --- Communication ---
        self.publisher_ = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan_filtered', self.scan_callback, qos_profile_sensor_data
        )
        self.timer = self.create_timer(0.05, self.publish_cmd)

        # --- CSV logging (DISABLED: set to True to re-enable) ---
        self.log_csv = False
        self.csv_file = None
        self.csv_writer = None
        self.csv_filename = None
        if self.log_csv:
            self._open_csv()

    # ------------------------------------------------------------------ CSV
    def _open_csv(self):
        save_dir = "/home/user/ros2_ws_A7"
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(save_dir, f"robot_log_{ts}.csv")
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp_ros', 'min_distance_cm', 'safe_threshold_cm', 'brake_state',
            'tau_tx_ms', 'tau_rx_ms', 'tau_e2e_ms', 'p_tx', 'p_rx', 'p_e2e'
        ])

    def close_csv(self):
        """Idempotent: safe to call even if CSV logging is disabled."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

    def log_to_csv(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.csv_writer.writerow([
            f"{now:.4f}", f"{self.min_dist_cm:.2f}", f"{self.safe_threshold:.2f}",
            int(self.last_brake_state),
            f"{self.tau_tx:.2f}", f"{self.tau_rx:.2f}", f"{self.taue2e:.2f}",
            f"{self.p_tx:.4f}", f"{self.p_rx:.4f}", f"{self.pe2e:.4f}"
        ])

    # ----------------------------------------------------------------- Laser
    def scan_callback(self, msg: LaserScan):
        self.last_scan_ns = self.get_clock().now().nanoseconds

        # array.array -> numpy without copying the whole scan
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        zone = np.concatenate((ranges[:self.k], ranges[-self.k:]))
        valid = zone[(zone > msg.range_min) & (zone < msg.range_max) & np.isfinite(zone)]

        if self.SMART:
            v_cm_per_ms = self.vmax / 1000.0
            time_buffer = self.T * (self.current_N + 1) + self.taue2e
            self.safe_threshold = (
                v_cm_per_ms * time_buffer + self.bot_length + self.m + self.d_brake
            )

        if valid.size > 0:
            self.min_dist_cm = float(valid.min()) * 100.0
            self.last_brake_state = self.min_dist_cm < self.safe_threshold
        else:
            self.get_logger().warn(
                "No valid lidar readings!", throttle_duration_sec=5.0
            )
            self.min_dist_cm = -1.0
            self.last_brake_state = True

        if self.csv_writer is not None:
            self.log_to_csv()

    def publish_cmd(self):
        now = self.get_clock().now()
        stale = (
            self.last_scan_ns is None
            or (now.nanoseconds - self.last_scan_ns) > self.SCAN_TIMEOUT_NS
        )
        brake = self.last_brake_state or stale

        self._msg.header.stamp = now.to_msg()
        self._msg.twist.linear.x = 0.0 if brake else self.linear_speed
        self.publisher_.publish(self._msg)

    def stop_robot(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._msg.twist.linear.x = 0.0
        self.publisher_.publish(self._msg)
        self.close_csv()

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
        while high < self.N_MAX and self._calculate_P_N(high, p_tx, p_rx) > self.P_target:
            high *= 2
        high = min(high, self.N_MAX)

        if high == 1:
            return 1

        # 2. Binary search
        low = high // 2
        search_domain = range(low, high + 1)
        index = bisect.bisect_left(
            search_domain,
            -self.P_target,
            key=lambda n: -self._calculate_P_N(n, p_tx, p_rx),
        )
        if index >= len(search_domain):
            return self.N_MAX  # target not reachable (links almost down)
        return search_domain[index]

    def update(self, p_tx: float, p_rx: float) -> int:
        temp_N = self.current_N
        steps = 0

        if self._calculate_P_N(temp_N, p_tx, p_rx) > self.P_target:
            # Current N too weak
            while self._calculate_P_N(temp_N, p_tx, p_rx) > self.P_target:
                temp_N += 1
                steps += 1
                if steps > self.max_linear_steps:
                    self.current_N = self._binary_search_fallback(p_tx, p_rx)
                    return self.current_N
        else:
            # Current N strong enough: can we go lower?
            while temp_N > 1 and self._calculate_P_N(temp_N - 1, p_tx, p_rx) <= self.P_target:
                temp_N -= 1
                steps += 1
                if steps > self.max_linear_steps:
                    self.current_N = self._binary_search_fallback(p_tx, p_rx)
                    return self.current_N

        self.current_N = min(temp_N, self.N_MAX)
        return self.current_N


class NetworkMonitor(Node):
    def __init__(self, target_node: LaserProcessor):
        super().__init__('network_monitor_client')

        # QoS must match the server's (see its `best_effort` parameter)
        self.declare_parameter('best_effort', False)
        if self.get_parameter('best_effort').value:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
        else:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )

        self.pub = self.create_publisher(LinkState, 'network_uplink', qos)
        self.sub = self.create_subscription(LinkState, 'network_downlink', self.callback, qos)

        self.target_node = target_node
        self.create_timer(0.1, self.send_packet)

        self.seq = 0
        self.expected_dn_seq = 0
        self.win = 100

        self.history_up_lat = deque(maxlen=self.win)
        self.history_dn_lat = deque(maxlen=self.win)
        self.history_received_dn = deque(maxlen=self.win)  # 1 = received, 0 = lost

    def send_packet(self):
        msg = LinkState()
        msg.ts_sent = time.time()
        msg.seq = self.seq
        self.pub.publish(msg)
        self.seq += 1

    def _update_loss_window(self, seq: int):
        if seq >= self.expected_dn_seq:
            gap = seq - self.expected_dn_seq
            self.history_received_dn.extend([0] * min(gap, self.win))
            self.history_received_dn.append(1)
            self.expected_dn_seq = seq + 1
        # else: duplicate / late packet -> ignored

    def callback(self, msg):
        now = time.time()

        self._update_loss_window(msg.seq)

        # Clamp at 0: negative values only come from clock offset between machines
        # (the real fix is clock sync via chrony/PTP; this just avoids shrinking
        # the SMART threshold with nonsense values).
        self.history_up_lat.append(max(0.0, msg.up_lat))
        self.history_dn_lat.append(max(0.0, (now - msg.ts_sent) * 1000.0))

        tau_up = statistics.median(self.history_up_lat)
        tau_dn = statistics.median(self.history_dn_lat)
        tau_e2e = tau_up + tau_dn

        p_up = msg.up_loss
        p_dn = 1.0 - (sum(self.history_received_dn) / len(self.history_received_dn))
        p_e2e = p_up + p_dn - (p_up * p_dn)

        t = self.target_node
        t.p_tx, t.p_rx, t.pe2e = p_up, p_dn, p_e2e
        t.tau_tx, t.tau_rx, t.taue2e = tau_up, tau_dn, tau_e2e

        N = t.update(p_up, p_dn)

        # One status line every 2 s instead of 3 prints per packet
        self.get_logger().info(
            f"lat up/dn/e2e {tau_up:.1f}/{tau_dn:.1f}/{tau_e2e:.1f} ms | "
            f"loss up/dn/e2e {p_up * 100:.1f}/{p_dn * 100:.1f}/{p_e2e * 100:.1f} % | N={N}",
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)

    laser_node = LaserProcessor()
    network_node = NetworkMonitor(laser_node)

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