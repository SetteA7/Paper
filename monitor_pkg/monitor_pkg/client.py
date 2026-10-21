# Responder node: echoes the controller's probes back (class is called NetworkServer).
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