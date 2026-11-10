# Bot side of the link monitor: run with `ros2 run monitor_pkg bot_monitor`.
import rclpy
from rclpy.executors import ExternalShutdownException

from monitor_pkg.network_link import NetworkServer


def main(args=None):
    rclpy.init(args=args)
    node = NetworkServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()