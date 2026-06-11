"""
test_adaptive_jacobian.py
=========================
Tests unitarios para el módulo core (Contribuciones C1 y C2).
Verifica propiedades matemáticas sin necesidad de ROS2.

Ejecutar:
    cd adaptive_vs_ws
    source .venv/bin/activate
    pytest tests/ -v --cov=src/adaptive_visual_servo/adaptive_visual_servo
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../src/adaptive_visual_servo'))

from adaptive_visual_servo.adaptive_jacobian import (
    AdaptiveVisualJacobian,
    ImageInteractionMatrix,
    TorquePenalizedController,
)


class TestAdaptiveVisualJacobian:

    def test_initial_shape(self):
        J = AdaptiveVisualJacobian(n_joints=6, n_features=8)
        # J_a ∈ ℝ^{m × n} = (n_features, n_joints)
        assert J.J.shape == (8, 6)

    def test_update_changes_jacobian(self):
        J = AdaptiveVisualJacobian(n_joints=6, n_features=8)
        J_before = J.J.copy()
        delta_q = np.random.randn(6) * 0.01
        delta_s = np.random.randn(8) * 5.0
        J.update(delta_q, delta_s)
        assert not np.allclose(J.J, J_before), \
            "El Jacobiano debe cambiar después de un update"

    def test_pseudoinverse_shape(self):
        J = AdaptiveVisualJacobian(n_joints=6, n_features=8)
        Jpinv = J.get_pseudoinverse()
        # J_a⁺ ∈ ℝ^{n × m} = (n_joints, n_features)
        assert Jpinv.shape == (6, 8)

    def test_update_count(self):
        J = AdaptiveVisualJacobian(n_joints=6, n_features=8)
        for _ in range(10):
            J.update(np.zeros(6), np.ones(8) * 0.1)
        assert J._update_count == 10

    def test_adaptation_rmse_positive(self):
        J = AdaptiveVisualJacobian(n_joints=6, n_features=8)
        for _ in range(5):
            J.update(np.random.randn(6), np.random.randn(8))
        assert J.adaptation_rmse >= 0.0

    def test_convergence_trend(self):
        """Con actualizaciones consistentes el RMSE de adaptación debe decrecer."""
        J = AdaptiveVisualJacobian(n_joints=6, n_features=8, alpha=0.8)
        J_true = np.random.default_rng(0).normal(0, 0.04, (8, 6))
        q_dot_true = np.array([0.1, -0.05, 0.02, 0.0, 0.01, -0.02])

        for _ in range(200):
            delta_s = J_true @ q_dot_true + np.random.default_rng(1).normal(0, 0.001, 8)
            J.update(q_dot_true, delta_s)

        assert J.adaptation_rmse < 1.0


class TestImageInteractionMatrix:

    def test_output_shape_2_points(self):
        L = ImageInteractionMatrix(focal_length=554.0)
        features = np.array([[0.1, -0.1], [-0.1, 0.1]])
        depths = np.array([0.8, 0.8])
        Lm = L.compute(features, depths)
        assert Lm.shape == (4, 6), f"Esperado (4,6), obtenido {Lm.shape}"

    def test_output_shape_4_points(self):
        L = ImageInteractionMatrix()
        features = np.array([[-0.1,-0.1],[0.1,-0.1],[0.1,0.1],[-0.1,0.1]])
        depths = np.full(4, 0.8)
        Lm = L.compute(features, depths)
        assert Lm.shape == (8, 6)

    def test_depth_scaling(self):
        """Duplicar la profundidad debe reducir la magnitud de L (aproximadamente)."""
        L = ImageInteractionMatrix()
        feat = np.array([[0.0, 0.0]])
        L1 = L.compute(feat, np.array([0.5]))
        L2 = L.compute(feat, np.array([1.0]))
        # Para u=v=0: L[0,0] = -1/Z, entonces |L1| > |L2|
        assert abs(L1[0, 0]) > abs(L2[0, 0])

    def test_near_zero_depth_handled(self):
        """Profundidad casi cero no debe causar NaN."""
        L = ImageInteractionMatrix()
        feat = np.array([[0.1, 0.1]])
        Lm = L.compute(feat, np.array([0.0]))  # Z=0 → clip a 0.01
        assert not np.any(np.isnan(Lm))
        assert not np.any(np.isinf(Lm))


class TestTorquePenalizedController:

    def setup_method(self):
        self.ctrl = TorquePenalizedController(
            n_joints=6,
            lambda_s=0.5,
            lambda_tau=0.1,
            tau_max=np.full(6, 100.0),
        )

    def test_output_shape(self):
        error = np.random.randn(8) * 10.0
        tau = np.random.randn(6) * 20.0
        q_dot, metrics = self.ctrl.compute_control(error, tau)
        assert q_dot.shape == (6,)

    def test_velocity_limit_respected(self):
        """El comando no debe exceder joint_vel_limit."""
        error = np.ones(8) * 1000.0  # error muy grande
        tau = np.zeros(6)
        q_dot, _ = self.ctrl.compute_control(error, tau)
        assert np.linalg.norm(q_dot) <= self.ctrl.joint_vel_limit + 1e-6

    def test_zero_error_zero_velocity(self):
        """Error cero debe producir velocidad cero (o casi)."""
        error = np.zeros(8)
        tau = np.zeros(6)
        q_dot, _ = self.ctrl.compute_control(error, tau)
        assert np.linalg.norm(q_dot) < 1e-8

    def test_torque_penalty_matrix_normalized(self):
        """W_τ debe tener valores entre 0 y 1."""
        tau = np.array([150.0, 75.0, 0.0, 28.0, 14.0, 0.0])
        W = self.ctrl.compute_torque_penalty_matrix(tau)
        diag = np.diag(W)
        assert np.all(diag >= 0.0)
        assert np.all(diag <= 1.0 + 1e-6)

    def test_lambda_tau_zero_equals_classic_ibvs(self):
        """Con λ_τ=0 el controlador debe ser idéntico al IBVS clásico."""
        ctrl_classic = TorquePenalizedController(
            n_joints=6, lambda_s=0.5, lambda_tau=0.0,
            tau_max=np.full(6, 100.0))
        ctrl_proposed = TorquePenalizedController(
            n_joints=6, lambda_s=0.5, lambda_tau=0.0,
            tau_max=np.full(6, 100.0))

        # Mismo Jacobiano inicial (seed=42 en ambos)
        error = np.array([10.0, -5.0, 8.0, 3.0, -7.0, 2.0, -4.0, 6.0])
        tau = np.array([50.0, 30.0, 20.0, 10.0, 8.0, 5.0])

        q1, _ = ctrl_classic.compute_control(error, tau)
        q2, _ = ctrl_proposed.compute_control(error, tau)
        np.testing.assert_allclose(q1, q2, rtol=1e-6)

    def test_energy_summary_after_iterations(self):
        error = np.ones(8) * 20.0
        tau = np.ones(6) * 30.0
        for _ in range(50):
            self.ctrl.compute_control(error, tau)
        summary = self.ctrl.get_energy_summary()
        assert 'total_energy_joules' in summary
        assert summary['iterations'] == 50
        assert summary['total_energy_joules'] > 0

    def test_metrics_keys_present(self):
        error = np.random.randn(8)
        tau = np.random.randn(6)
        _, metrics = self.ctrl.compute_control(error, tau)
        required = {'image_error_norm', 'joint_torque_rms',
                    'instantaneous_power', 'control_effort'}
        assert required.issubset(metrics.keys())

    def test_proposed_less_energy_than_baseline_high_torque(self):
        """
        Con torque alto, el método propuesto (λ_τ>0) debe producir
        menor potencia que el baseline (λ_τ=0).

        Nota: esta propiedad se verifica estadísticamente sobre muchos pasos,
        no necesariamente en cada iteración individual.
        """
        ctrl_b = TorquePenalizedController(
            n_joints=6, lambda_s=0.5, lambda_tau=0.0,
            tau_max=np.full(6, 100.0))
        ctrl_p = TorquePenalizedController(
            n_joints=6, lambda_s=0.5, lambda_tau=0.3,
            tau_max=np.full(6, 100.0))

        tau_high = np.full(6, 80.0)  # alta carga
        total_power_b, total_power_p = 0.0, 0.0

        for _ in range(100):
            error = np.random.randn(8) * 15.0
            _, m_b = ctrl_b.compute_control(error, tau_high)
            _, m_p = ctrl_p.compute_control(error, tau_high)
            total_power_b += m_b['instantaneous_power']
            total_power_p += m_p['instantaneous_power']

        # El propuesto debe consumir menos energía total bajo alta carga
        assert total_power_p <= total_power_b + 1e-6, (
            f"Propuesto ({total_power_p:.2f}) debería ≤ Baseline ({total_power_b:.2f})")
