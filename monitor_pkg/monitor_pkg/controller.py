Now tell me what to fix here. Reduce the logs in both the file output (disable it completely for now) and also the terminal logs. Is this code slow?

import csv
import math
import signal
import bisect
import statistics
import time
import os  # Aggiunto per la gestione dei percorsi
from collections import deque
from datetime import datetime
from itertools import chain

# Librerie di terze parti
import numpy as np

# ROS 2 Core
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import (
    SingleThreadedExecutor, 
    MultiThreadedExecutor, 
    ExternalShutdownException
)
from rclpy.qos import (
    QoSProfile, 
    ReliabilityPolicy, 
    HistoryPolicy, 
    qos_profile_sensor_data
)

# Messaggi ROS 2
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from link_monitor_msg.msg import LinkMetrics, LinkState

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class LaserProcessor(Node):
    def __init__(self):
        super().__init__('laser_processor')
        
        # --- Laser Variables ---
        self.k = 100                # Number of samples to use at beginning/end of array

        # --- Controller Variables ---
        # Robot Parameters
        self.m = 5                  # safety margin (cm)
        self.bot_length = 18.75        # bot length (cm) MEASURED IN LAB
        self.d_brake = 18.1       # braking distance (cm) MEASURED IN LAB
        self.vmax = 43.77  #83.18           # max speed (cm/s) MEASURED IN LAB
        self.T = 50                # control period (ms)
        self.safe_threshold = 75 #100   # brake threshold [S] (cm)

        # --- Configurazione Statica ---
        self.linear_speed = 0.5      # m/s
        self.last_brake_state = False
        
        self.P_crash_target = 0.05  
        self.SMART = False

        self.P_target = self.P_crash_target
        self.max_linear_steps = 4
        self.current_N = 1

        # --- Newtork Variables ---
        self.tau_tx = 10
        self.tau_rx = 10
        self.taue2e = self.tau_rx + self.tau_tx

        self.p_tx = 0.05
        self.p_rx = 0.05
        self.pe2e = self.p_rx + self.p_tx - (self.p_tx * self.p_rx)

        # --- Pre-allocazione Messaggi (Ottimizzazione Memoria) ---
        self._msg = TwistStamped()
        self._msg.header.frame_id = "base_link"
        
        # --- Comunicazione ---
        self.publisher_ = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_filtered',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        self.timer = self.create_timer(0.05, self.publish_cmd)

        # --- CSV Setup ---
        save_dir = "/home/user/ros2_ws_A7"
        os.makedirs(save_dir, exist_ok=True)  # Crea la cartella se non esiste
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(save_dir, f"robot_log_{timestamp_str}.csv")
        
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'timestamp_ros', 'min_distance_cm', 'safe_threshold_cm', 'brake_state', 
            'tau_tx_ms', 'tau_rx_ms', 'tau_e2e_ms', 
            'p_tx', 'p_rx', 'p_e2e'
        ])
        
        self.get_logger().info(f"Logging avviato in: {self.csv_filename}")

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        interest_zone = np.concatenate((ranges[:self.k], ranges[-self.k:]))
        
        mask = (interest_zone > msg.range_min) & \
               (interest_zone < msg.range_max) & \
               (np.isfinite(interest_zone))
        
        valid_readings = interest_zone[mask]
 
        if self.SMART:
            v_ms = self.vmax / 1000.0 # cm/ms
            time_buffer = self.T * (self.current_N + 1) + self.taue2e
            s_cm_smart = (v_ms * time_buffer) + self.bot_length + self.m + self.d_brake             
            self.safe_threshold = s_cm_smart
            self.get_logger().info(f"SMART Thresh: {self.safe_threshold:.2f} cm")

        if valid_readings.size > 0:
            self.min_dist_cm = np.min(valid_readings) * 100.0
            self.last_brake_state = self.min_dist_cm < self.safe_threshold
            self.get_logger().info(f"Distance: {np.min(valid_readings)*100:.2f} cm")
        else:
            self.get_logger().warn("Nessuna lettura valida dal Lidar!", throttle_duration_sec=2.0)
            self.min_dist_cm = -1.0
            self.last_brake_state = True

        self.log_to_csv()

    def log_to_csv(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.csv_writer.writerow([
            f"{now:.4f}",
            f"{self.min_dist_cm:.2f}",
            f"{self.safe_threshold:.2f}",
            int(self.last_brake_state),
            f"{self.tau_tx:.2f}",
            f"{self.tau_rx:.2f}",
            f"{self.taue2e:.2f}",
            f"{self.p_tx:.4f}",
            f"{self.p_rx:.4f}",
            f"{self.pe2e:.4f}"
        ])

    def publish_cmd(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._msg.twist.linear.x = 0.0 if self.last_brake_state else self.linear_speed
        self.publisher_.publish(self._msg)

    def stop_robot(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._msg.twist.linear.x = 0.0
        self.publisher_.publish(self._msg)
        
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f"Log salvato correttamente in: {self.csv_filename}")
        
    def _calculate_P_N(self, N: int, p_tx: float, p_rx: float) -> float:
        if abs(p_tx - p_rx) < 1e-6:
            p = p_tx
            return (p**N) * (1 + (1 - p) * N)
        else:
            term_fraction = (1 - p_tx) * p_rx / (p_rx - p_tx)
            return p_tx**N + term_fraction * (p_rx**N - p_tx**N)

    # Exp Binary Fallback
    def _binary_search_fallback(self, p_tx: float, p_rx: float) -> int:
        # 1. Exponential Bounding (Find the ceiling)
        high = 1
        while self._calculate_P_N(high, p_tx, p_rx) > self.P_target:
            high *= 2
        
        if high == 1:
            return 1
            
        # 2. Binary Search using bisect
        low = high // 2
        search_domain = range(low, high + 1)
        
        # We invert the values (-) to satisfy bisect's requirement for ascending order
        index = bisect.bisect_left(
            search_domain, 
            -self.P_target, 
            key=lambda n: -self._calculate_P_N(n, p_tx, p_rx)
        )
        
        return search_domain[index]

    # Threshold Update (called from NetworkMonitor)
    def update(self, p_tx: float, p_rx: float) -> int:
        temp_N = self.current_N
        steps = 0
        
        # CASE 1: Current N is too weak (Failure probability is too high)
        if self._calculate_P_N(temp_N, p_tx, p_rx) > self.P_target:
            while self._calculate_P_N(temp_N, p_tx, p_rx) > self.P_target:
                temp_N += 1
                steps += 1
                
                # Abort local search if drifting too far
                if steps > self.max_linear_steps:
                    self.current_N = self._binary_search_fallback(p_tx, p_rx)
                    return self.current_N
                    
        # CASE 2: Current N is strong enough (Can we use a smaller N?)
        else:
            while temp_N > 1 and self._calculate_P_N(temp_N - 1, p_tx, p_rx) <= self.P_target:
                temp_N -= 1
                steps += 1
                
                # Abort local search if conditions improved massively
                if steps > self.max_linear_steps:
                    self.current_N = self._binary_search_fallback(p_tx, p_rx)
                    return self.current_N
                    
        # Save the new optimal N for the next tick
        self.current_N = temp_N

        #self.get_logger().info(f"UPDATE!!!: {self.current_N}")
        
        return self.current_N


class NetworkMonitor(Node):
    def __init__(self, target_node):
        super().__init__('network_monitor_client')

        if False:
            self.qos_best_effort = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            self.pub = self.create_publisher(
                LinkState, 
                'network_uplink', 
                self.qos_best_effort
            )
            self.sub = self.create_subscription(
                LinkState, 
                'network_downlink', 
                self.callback, 
                self.qos_best_effort
            )
        else:
            self.pub = self.create_publisher(LinkState, 'network_uplink', 10)
            self.sub = self.create_subscription(LinkState, 'network_downlink', self.callback, 10)

        self.target_node = target_node
        self.create_timer(0.1, self.send_packet)

        self.seq = 0
        self.expected_dn_seq = 0
        self.win = 100 

        self.history_up_lat = deque(maxlen=self.win)
        self.history_dn_lat = deque(maxlen=self.win)
        self.history_loss_dn = deque(maxlen=self.win)

        print(f"\n{'SEQ':<5} | {'MED LAT UP':<8} | {'MED LAT DN':<8} | {'LAT E2E':<8}")
        print(f"\n{'SEQ':<5} | {'MED Ploss UP':<8} | {'MED Ploss DN':<8} | {'P E2E':<8}")
        print("-" * 65)

    def send_packet(self):
        msg = LinkState()
        msg.ts_sent = time.time()
        msg.seq = self.seq
        self.pub.publish(msg)
        self.seq += 1

    def callback(self, msg):
        now = time.time()
        dn_lat_raw = (now - msg.ts_sent) * 1000

        if msg.seq > self.expected_dn_seq:
            for _ in range(msg.seq - self.expected_dn_seq):
                self.history_loss_dn.append(0)
        
        self.history_loss_dn.append(1) 
        self.expected_dn_seq = msg.seq + 1

        self.history_up_lat.append(msg.up_lat)
        self.history_dn_lat.append(dn_lat_raw)

        tau_up = statistics.median(self.history_up_lat)
        tau_dn = statistics.median(self.history_dn_lat)
        tau_e2e = tau_up + tau_dn

        p_up = msg.up_loss 
        p_dn = 1.0 - (sum(self.history_loss_dn) / len(self.history_loss_dn))
        p_e2e = (p_up + p_dn - (p_up * p_dn))

        print(f"{msg.seq:<5} | {tau_up:6.1f}ms | {tau_dn:6.1f}ms | {tau_e2e:6.1f}ms")
        print(f"{msg.seq:<5} | {p_up*100:6.1f}% | {p_dn*100:6.1f}% | {p_e2e*100:6.1f}%")
        print(f" ")
        
        self.target_node.p_tx = p_up
        self.target_node.p_rx = p_dn
        self.target_node.pe2e = p_e2e
        self.target_node.tau_tx = tau_up
        self.target_node.tau_rx = tau_dn
        self.target_node.taue2e = tau_e2e

        self.target_node.update(p_up, p_dn)

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
        print("\n[Controller] Stopping robot and saving data...")
        if rclpy.ok():
            try:
                laser_node.stop_robot()
                executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                print(f"[Controller] Could not send stop command: {e}")

        # Sicurezza aggiuntiva per il salvataggio CSV durante il force-kill
        if hasattr(laser_node, 'csv_file') and not laser_node.csv_file.closed:
            laser_node.csv_file.flush()
            laser_node.csv_file.close()
            print(f"[Controller] Data forcefully saved to: {laser_node.csv_filename}")

        executor.shutdown()
        laser_node.destroy_node()
        network_node.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()
        
        print("[Controller] Clean exit complete.")

if __name__ == '__main__':
    main()