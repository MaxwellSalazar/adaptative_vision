"""
adaptive_vs_controller.py
==========================
Nodo principal del controlador visual servo adaptativo (Contribución C1+C2).

Este nodo es el núcleo del sistema: recibe features de imagen y torques
articulares, calcula el comando de velocidad articular con penalización
energética, y publica las métricas para análisis.

Subscriptions:
    /vs/features_current        (Float32MultiArray) — features actuales s
    /vs/features_desired        (Float32MultiArray) — features deseadas s*
    /vs/depth_estimates         (Float32MultiArray) — profundidades Z_i
    /joint_states               (sensor_msgs/JointState) — q, q̇, τ

Publications:
    /joint_group_vel_controller/commands (Float64MultiArray) — q̇ comandada
    /vs/metrics                 (Float32MultiArray) — métricas paso a paso
    /vs/energy_metrics          (Float32MultiArray) — energía acumulada
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np

from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from adaptive_visual_servo.adaptive_jacobian import TorquePenalizedController


class AdaptiveVSController(Node):

    def __init__(self):
        super().__init__('adaptive_vs_controller')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('n_joints', 6)
        self.declare_parameter('lambda_s', 0.5)
        self.declare_parameter('lambda_tau', 0.1)
        self.declare_parameter('joint_vel_limit', 0.8)
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('baseline_mode', False)
        self.declare_parameter('tau_max', [150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

        n_joints = self.get_parameter('n_joints').value
        lambda_s = self.get_parameter('lambda_s').value
        lambda_tau = self.get_parameter('lambda_tau').value
        vel_limit = self.get_parameter('joint_vel_limit').value
        tau_max = np.array(self.get_parameter('tau_max').value)
        self.baseline_mode = self.get_parameter('baseline_mode').value

        if self.baseline_mode:
            # Modo baseline: lambda_tau=0 → IBVS clásico sin penalización
            lambda_tau = 0.0
            self.get_logger().warn(
                'Modo BASELINE activo: lambda_tau=0 (IBVS clásico)')

        # ── Controlador ───────────────────────────────────────────────────────
        self.controller = TorquePenalizedController(
            n_joints=n_joints,
            lambda_s=lambda_s,
            lambda_tau=lambda_tau,
            tau_max=tau_max,
            joint_vel_limit=vel_limit,
        )

        # ── Estado interno ────────────────────────────────────────────────────
        self._features_current = np.zeros(8, dtype=np.float32)
        self._features_desired = np.zeros(8, dtype=np.float32)
        self._depths = np.ones(4, dtype=np.float32)
        self._tau_current = np.zeros(n_joints, dtype=np.float64)
        self._q_prev = None
        self._s_prev = None
        self._target_detected = False

        # ── Subscriptions ─────────────────────────────────────────────────────
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT)

        self.sub_feat_curr = self.create_subscription(
            Float32MultiArray, '/vs/features_current',
            self._feat_curr_cb, 10)
        self.sub_feat_des = self.create_subscription(
            Float32MultiArray, '/vs/features_desired',
            self._feat_des_cb, 10)
        self.sub_depths = self.create_subscription(
            Float32MultiArray, '/vs/depth_estimates',
            self._depth_cb, 10)
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_cb, 10)
        self.sub_detected = self.create_subscription(
            Bool, '/vs/target_detected',
            self._detected_cb, 10)

        # ── Publications ──────────────────────────────────────────────────────
        self.pub_vel_cmd = self.create_publisher(
            Float64MultiArray,
            '/joint_group_vel_controller/commands', 10)
        self.pub_metrics = self.create_publisher(
            Float32MultiArray, '/vs/metrics', 10)
        self.pub_energy = self.create_publisher(
            Float32MultiArray, '/vs/energy_metrics', 10)

        # ── Timer de control ──────────────────────────────────────────────────
        rate = self.get_parameter('control_rate_hz').value
        self._control_timer = self.create_timer(
            1.0 / rate, self._control_loop)

        # ── Timer de reporte energético (cada 2 s) ────────────────────────────
        self._energy_timer = self.create_timer(2.0, self._publish_energy)

        self.get_logger().info(
            f'AdaptiveVSController listo | λ_s={lambda_s} λ_τ={lambda_tau} '
            f'| baseline={self.baseline_mode}')

    # ── Callbacks de subscripciones ───────────────────────────────────────────

    def _feat_curr_cb(self, msg: Float32MultiArray):
        self._features_current = np.array(msg.data, dtype=np.float32)

    def _feat_des_cb(self, msg: Float32MultiArray):
        self._features_desired = np.array(msg.data, dtype=np.float32)

    def _depth_cb(self, msg: Float32MultiArray):
        self._depths = np.array(msg.data[:4], dtype=np.float32)

    def _joint_state_cb(self, msg: JointState):
        if msg.effort:
            self._tau_current = np.array(msg.effort[:6])
        elif msg.velocity:
            # Fallback: estimar torque desde velocidad (simulación)
            self._tau_current = np.array(msg.velocity[:6]) * 10.0

    def _detected_cb(self, msg: Bool):
        self._target_detected = msg.data

    # ── Loop de control ───────────────────────────────────────────────────────

    def _control_loop(self):
        if not self._target_detected:
            self._send_zero_velocity()
            return

        if len(self._features_current) < 8 or len(self._features_desired) < 8:
            return

        # Error en espacio imagen: e = s - s*
        image_error = self._features_current - self._features_desired

        # Umbral de convergencia (< 5 px → detenerse)
        if np.linalg.norm(image_error) < 5.0:
            self._send_zero_velocity()
            return

        # Calcular comando de velocidad articular
        q_dot, metrics = self.controller.compute_control(
            image_error=image_error,
            tau_current=self._tau_current,
        )

        # Enviar comando
        cmd_msg = Float64MultiArray()
        cmd_msg.data = q_dot.tolist()
        self.pub_vel_cmd.publish(cmd_msg)

        # Actualizar Jacobiano adaptativo si hay estado previo
        if self._s_prev is not None and self._q_prev is not None:
            delta_s = self._features_current - self._s_prev
            delta_q = self._tau_current[:self.controller.n_joints] * 0.0  # placeholder
            if np.linalg.norm(delta_s) > 0.1:
                self.controller.update_jacobian(
                    np.zeros(self.controller.n_joints), delta_s)

        self._s_prev = self._features_current.copy()

        # Publicar métricas
        self._publish_metrics(metrics)

    def _send_zero_velocity(self):
        cmd = Float64MultiArray()
        cmd.data = [0.0] * self.controller.n_joints
        self.pub_vel_cmd.publish(cmd)

    def _publish_metrics(self, metrics: dict):
        msg = Float32MultiArray()
        msg.data = [
            metrics.get('image_error_norm', 0.0),
            metrics.get('joint_torque_rms', 0.0),
            metrics.get('instantaneous_power', 0.0),
            metrics.get('control_effort', 0.0),
            metrics.get('torque_penalty_trace', 0.0),
        ]
        self.pub_metrics.publish(msg)

    def _publish_energy(self):
        summary = self.controller.get_energy_summary()
        if not summary:
            return
        msg = Float32MultiArray()
        msg.data = [
            summary.get('total_energy_joules', 0.0),
            summary.get('mean_power_watts', 0.0),
            summary.get('peak_power_watts', 0.0),
            summary.get('torque_rms_mean', 0.0),
            float(summary.get('iterations', 0)),
        ]
        self.pub_energy.publish(msg)
        self.get_logger().info(
            f"[Energy] E={summary['total_energy_joules']:.2f}J  "
            f"P_mean={summary['mean_power_watts']:.2f}W  "
            f"iter={summary['iterations']}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveVSController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
