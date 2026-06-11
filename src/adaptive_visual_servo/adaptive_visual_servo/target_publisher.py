"""
target_publisher.py
===================
Publica la posición del objeto objetivo en los 3 escenarios de validación:

  Scenario 1 — static:   objeto quieto en (0.6, 0.0, 0.4)
  Scenario 2 — linear:   movimiento lineal a lo largo del eje Y
  Scenario 3 — sinusoidal: trayectoria curva sinusoidal en plano XY

Mueve el modelo Gazebo por servicio y publica la pose para registro.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist, Vector3


class TargetPublisherNode(Node):

    SCENARIOS = {
        'static': {
            'description': 'Objeto estático en posición fija',
            'motion_fn': None,
        },
        'linear': {
            'description': 'Movimiento lineal uniforme en eje Y',
            'amplitude': 0.3,
            'frequency': 0.1,
        },
        'sinusoidal': {
            'description': 'Trayectoria sinusoidal en plano XY',
            'amplitude_x': 0.2,
            'amplitude_y': 0.15,
            'frequency': 0.15,
        },
    }

    def __init__(self):
        super().__init__('target_publisher')

        self.declare_parameter('scenario', 'static')
        self.declare_parameter('target_model_name', 'target_sphere')
        self.declare_parameter('base_x', 0.6)
        self.declare_parameter('base_y', 0.0)
        self.declare_parameter('base_z', 0.4)
        self.declare_parameter('update_rate_hz', 50.0)

        self.scenario = self.get_parameter('scenario').value
        self.model_name = self.get_parameter('target_model_name').value
        self.base_x = self.get_parameter('base_x').value
        self.base_y = self.get_parameter('base_y').value
        self.base_z = self.get_parameter('base_z').value
        rate = self.get_parameter('update_rate_hz').value

        self._t0 = self.get_clock().now()

        # Pub de posición para registro
        self.pub_pose = self.create_publisher(
            PoseStamped, '/vs/target_pose', 10)

        # Cliente Gazebo
        self.cli_set_state = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.create_timer(1.0 / rate, self._update)

        cfg = self.SCENARIOS.get(self.scenario, self.SCENARIOS['static'])
        self.get_logger().info(
            f'TargetPublisher | escenario={self.scenario} | {cfg["description"]}')

    def _update(self):
        t = (self.get_clock().now() - self._t0).nanoseconds / 1e9
        x, y, z = self._compute_position(t)

        # Publicar pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.w = 1.0
        self.pub_pose.publish(pose_msg)

        # Mover modelo en Gazebo
        if self.cli_set_state.service_is_ready():
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = self.model_name
            req.state.pose = pose_msg.pose
            req.state.twist = Twist()
            req.state.reference_frame = 'world'
            self.cli_set_state.call_async(req)

    def _compute_position(self, t: float):
        if self.scenario == 'static':
            return self.base_x, self.base_y, self.base_z

        elif self.scenario == 'linear':
            cfg = self.SCENARIOS['linear']
            # Movimiento lineal senoidal (ida y vuelta)
            y = self.base_y + cfg['amplitude'] * math.sin(
                2 * math.pi * cfg['frequency'] * t)
            return self.base_x, y, self.base_z

        elif self.scenario == 'sinusoidal':
            cfg = self.SCENARIOS['sinusoidal']
            x = self.base_x + cfg['amplitude_x'] * math.sin(
                2 * math.pi * cfg['frequency'] * t)
            y = self.base_y + cfg['amplitude_y'] * math.sin(
                4 * math.pi * cfg['frequency'] * t)  # Lissajous XY
            return x, y, self.base_z

        return self.base_x, self.base_y, self.base_z


def main(args=None):
    rclpy.init(args=args)
    node = TargetPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
