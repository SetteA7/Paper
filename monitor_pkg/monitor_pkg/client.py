import rclpy
from rclpy.node import Node
import time
from collections import deque
import statistics

from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from link_monitor_msg.msg import LinkMetrics, LinkState

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class NetworkServer(Node):
    def __init__(self):
        super().__init__('network_monitor_server')

        if False:
            self.qos_best_effort = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            self.sub = self.create_subscription(
                LinkState, 
                'network_uplink', 
                self.callback,
                self.qos_best_effort
            )
            self.pub = self.create_publisher(
                LinkState, 
                'network_downlink', 
                self.qos_best_effort
            )
        else:
            self.sub = self.create_subscription(LinkState, 'network_uplink', self.callback, 10)
            self.pub = self.create_publisher(LinkState, 'network_downlink', 10)
        
        self.expected_up_seq = 0
        self.win = 100
        self.history_loss_up = deque(maxlen=self.win)

    def callback(self, msg):
        now = time.time()
        up_lat_raw = (now - msg.ts_sent) * 1000

        # --- GESTIONE PERDITA UPLINK (WINDOWED) ---
        if msg.seq > self.expected_up_seq:
            for _ in range(msg.seq - self.expected_up_seq):
                self.history_loss_up.append(0)
        
        self.history_loss_up.append(1)
        self.expected_up_seq = msg.seq + 1

        # Calcolo P_up windowed
        p_up = 1.0 - (sum(self.history_loss_up) / len(self.history_loss_up))

        # Risposta
        res = LinkState()
        res.ts_sent = time.time()
        res.seq = msg.seq
        res.up_lat = up_lat_raw
        res.up_loss = float(p_up)
        self.pub.publish(res)

def main(args=None):
    # Inizializza il contesto ROS2
    rclpy.init(args=args)
    try:
        node = NetworkServer()
        # L'Executor gestisce i callback in modo più granulare dello spin() globale
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            # Assicura che il nodo venga distrutto correttamente
            executor.shutdown()
            node.destroy_node()
            
    except KeyboardInterrupt:
        # Gestisce il Ctrl+C senza stampare errori brutti nel terminale
        pass
    finally:
        # Shutdown del contesto rclpy
        if rclpy.ok():
            rclpy.shutdown()

#def main(args=None):
#    rclpy.init(args=args)
#    node = NetworkServer()
#    rclpy.spin(node)
#    node.destroy_node()
#    rclpy.shutdown()

if __name__ == '__main__':
    main()