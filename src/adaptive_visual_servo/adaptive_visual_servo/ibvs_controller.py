"""
ibvs_controller.py
==================
Controlador IBVS clásico — usado como BASELINE en las comparativas.

Implementa IBVS estándar con:
  - Jacobiano de interacción fijo (profundidad media asumida constante)
  - Sin penalización de torque (λ_τ = 0)
  - Sin adaptación del Jacobiano

Este es el denominador de todas las métricas comparativas del paper:
    reducción_energía = (E_baseline - E_propuesto) / E_baseline × 100%
"""

import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32MultiArray, Bool, Float64MultiArray
from sensor_msgs.msg import JointState

from adaptive_visual_servo.adaptive_jacobian import (
    ImageInteractionMatrix,
    TorquePenalizedController,
)


class IBVSBaselineController(Node):
    """IBVS clásico: ley de control q̇ = -λ · L⁺_fija · e"""

    def __init__(self):
        super().__init__('ibvs_baseline_controller')

        self.declare_parameter('lambda_s', 0.5)
        self.declare_parameter('assumed_depth', 0.8)
        self.declare_parameter('focal_length', 554.0)
        self.declare_parameter('n_joints', 6)
        self.declare_parameter('joint_vel_limit', 0.8)

        self.lambda_s = self.get_parameter('lambda_s').value
        self.Z0 = self.get_parameter('assumed_depth').value
        self.f = self.get_parameter('focal_length').value
        self.n_joints = self.get_parameter('n_joints').value
        self.vel_limit = self.get_parameter('joint_vel_limit').value

        self._interaction = ImageInteractionMatrix(focal_length=self.f)
        # Jacobiano analítico fijo (contribución C2 desactivada)
        self._L_fixed = self._build_fixed_jacobian()

        self._features_current = np.zeros(8, dtype=np.float32)
        self._features_desired = np.zeros(8, dtype=np.float32)
        self._tau_current = np.zeros(self.n_joints, dtype=np.float64)
        self._target_detected = False

        # Métricas baseline (mismas que el método propuesto para comparar)
        self._history_power: list = []
        self._history_error: list = []

        self.sub_feat_curr = self.create_subscription(
            Float32MultiArray, '/vs/features_current',
            lambda m: setattr(self, '_features_current',
                              np.array(m.data, dtype=np.float32)), 10)
        self.sub_feat_des = self.create_subscription(
            Float32MultiArray, '/vs/features_desired',
            lambda m: setattr(self, '_features_desired',
                              np.array(m.data, dtype=np.float32)), 10)
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self.sub_detected = self.create_subscription(
            Bool, '/vs/target_detected',
            lambda m: setattr(self, '_target_detected', m.data), 10)

        self.pub_vel = self.create_publisher(
            Float64MultiArray,
            '/joint_group_vel_controller/commands', 10)
        self.pub_metrics = self.create_publisher(
            Float32MultiArray, '/vs/metrics', 10)

        self.create_timer(1.0 / 30.0, self._control_loop)
        self.get_logger().info('IBVSBaselineController listo (λ_τ=0, L fija)')

    def _build_fixed_jacobian(self) -> np.ndarray:
        """
        Construye L⁺ fija asumiendo Z=Z0 para los 4 puntos en el centro
        de la imagen (320, 240) normalizado a (0, 0).
        """
        features_center = np.array([
            [-0.1, -0.1],
            [0.1, -0.1],
            [0.1,  0.1],
            [-0.1,  0.1],
        ])
        depths = np.full(4, self.Z0)
        L = self._interaction.compute(features_center, depths)
        return np.linalg.pinv(L)  # (6, 8)

    def _joint_cb(self, msg: JointState):
        if msg.effort:
            self._tau_current = np.array(msg.effort[:self.n_joints])
        elif msg.velocity:
            self._tau_current = np.array(msg.velocity[:self.n_joints]) * 10.0

    def _control_loop(self):
        if not self._target_detected:
            self._zero_vel()
            return

        error = self._features_current - self._features_desired
        err_norm = float(np.linalg.norm(error))
        if err_norm < 5.0:
            self._zero_vel()
            return

        # q̇ = -λ · L⁺_fija · e   (sin penalización de torque)
        vel_6d = -self.lambda_s * (self._L_fixed @ error)

        # A velocidades articulares (6DOF directo en simulación simplificada)
        q_dot = vel_6d
        q_dot_norm = np.linalg.norm(q_dot)
        if q_dot_norm > self.vel_limit:
            q_dot = q_dot * self.vel_limit / q_dot_norm

        cmd = Float64MultiArray()
        cmd.data = q_dot.tolist()
        self.pub_vel.publish(cmd)

        # Métricas baseline
        power = float(np.dot(np.abs(self._tau_current), np.abs(q_dot)))
        self._history_power.append(power)
        self._history_error.append(err_norm)

        m = Float32MultiArray()
        m.data = [err_norm, float(np.sqrt(np.mean(self._tau_current**2))),
                  power, float(np.linalg.norm(q_dot)), 0.0]
        self.pub_metrics.publish(m)

    def _zero_vel(self):
        cmd = Float64MultiArray()
        cmd.data = [0.0] * self.n_joints
        self.pub_vel.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = IBVSBaselineController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
