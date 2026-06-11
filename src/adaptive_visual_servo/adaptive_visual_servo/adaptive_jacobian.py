"""
adaptive_jacobian.py
====================
Contribución C1 + C2 del paper.

Convención de dimensiones (ÚNICA, consistente en todo el módulo):
  n  = n_joints   (DOF del brazo, default 6)
  m  = n_features (dimensión del vector de features imagen, default 8)

  J_a  ∈ ℝ^{m × n}   Jacobiano imagen-robot: ṡ = J_a · q̇
  J_a⁺ ∈ ℝ^{n × m}   Pseudoinversa:          q̇_min = J_a⁺ · ṡ

Ley de control propuesta (C1):
  q̇ = -(λ_s·I_n + λ_τ·W_τ) · J_a⁺ · e

donde:
  e    ∈ ℝ^m   error imagen s - s*
  W_τ  ∈ ℝ^{n×n}  diag(|τᵢ|/τᵢᵐᵃˣ)  matriz de penalización de torque
"""

import numpy as np
from typing import Tuple, Optional


class AdaptiveVisualJacobian:
    """
    Jacobiano visual adaptativo J_a ∈ ℝ^{m × n} estimado online (C2).

    Regla de Broyden:
        J_a(k+1) = J_a(k) + α·(Δs - J_a(k)·Δq)·Δqᵀ / (‖Δq‖² + ε)

    donde Δs ∈ ℝ^m y Δq ∈ ℝ^n.
    """

    def __init__(
        self,
        n_joints: int = 6,
        n_features: int = 8,
        alpha: float = 0.5,
        epsilon: float = 1e-6,
        regularization: float = 1e-4,
    ):
        self.n_joints = n_joints        # n
        self.n_features = n_features    # m
        self.alpha = alpha
        self.epsilon = epsilon
        self.regularization = regularization

        # J_a ∈ ℝ^{m × n}
        rng = np.random.default_rng(seed=42)
        self.J = rng.normal(0, 0.01, (n_features, n_joints))

        self._update_count = 0
        self._adaptation_errors: list = []

    def update(self, delta_q: np.ndarray, delta_s: np.ndarray) -> None:
        """
        Actualiza J_a con observación (Δq, Δs).

        Parameters
        ----------
        delta_q : Δq ∈ ℝ^n  cambio articular
        delta_s : Δs ∈ ℝ^m  cambio de features imagen
        """
        delta_q = delta_q.reshape(self.n_joints)
        delta_s = delta_s.reshape(self.n_features)

        dq_norm_sq = float(np.dot(delta_q, delta_q)) + self.epsilon
        prediction_error = delta_s - self.J @ delta_q   # ℝ^m
        self.J += self.alpha * np.outer(prediction_error, delta_q) / dq_norm_sq
        self._update_count += 1
        self._adaptation_errors.append(float(np.linalg.norm(prediction_error)))

    def get_pseudoinverse(self) -> np.ndarray:
        """
        Pseudoinversa regularizada de Tikhonov.

        J_a  ∈ ℝ^{m × n}
        J_a⁺ ∈ ℝ^{n × m}  =  J_aᵀ · (J_a · J_aᵀ + ρ·I_m)^{-1}
        """
        # J_a · J_aᵀ ∈ ℝ^{m × m}
        JJt = self.J @ self.J.T
        reg = self.regularization * np.eye(self.n_features)
        # J_a⁺ = J_aᵀ · (J_a · J_aᵀ + ρI)^{-1}  →  shape (n, m)
        return self.J.T @ np.linalg.inv(JJt + reg)

    @property
    def adaptation_rmse(self) -> float:
        if not self._adaptation_errors:
            return 0.0
        return float(np.sqrt(np.mean(np.array(self._adaptation_errors[-50:]) ** 2)))


class ImageInteractionMatrix:
    """
    Matriz de interacción L ∈ ℝ^{2N × 6} para N puntos imagen.

    Para punto (u, v) con profundidad Z:
        L_p = [ -1/Z   0    u/Z    uv      -(1+u²)   v  ]
              [  0    -1/Z  v/Z   1+v²     -uv       -u  ]
    """

    def __init__(self, focal_length: float = 554.0):
        self.focal_length = focal_length

    def compute(self, features: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        features : (N, 2) coordenadas normalizadas
        depths   : (N,)   profundidades Z

        Returns
        -------
        L : (2N, 6)
        """
        N = features.shape[0]
        L = np.zeros((2 * N, 6))
        for i in range(N):
            u, v = features[i]
            Z = max(float(depths[i]), 0.01)
            r = 2 * i
            L[r,   :] = [-1/Z, 0,    u/Z,  u*v,       -(1+u**2),  v]
            L[r+1, :] = [ 0,  -1/Z,  v/Z,  1+v**2,    -u*v,      -u]
        return L


class TorquePenalizedController:
    """
    Controlador IBVS con penalización de torque articular (C1 + C2).

    Ley de control:
        q̇ = -(λ_s·I_n + λ_τ·W_τ) · J_a⁺ · e

    Dimensiones:
        e    : (m,)    error imagen
        J_a⁺ : (n, m)  pseudoinversa del Jacobiano adaptativo
        W_τ  : (n, n)  penalización de torque
        q̇    : (n,)    velocidades articulares
    """

    def __init__(
        self,
        n_joints: int = 6,
        n_features: int = 8,
        lambda_s: float = 0.5,
        lambda_tau: float = 0.1,
        tau_max: Optional[np.ndarray] = None,
        joint_vel_limit: float = 1.0,
    ):
        self.n_joints = n_joints
        self.n_features = n_features
        self.lambda_s = lambda_s
        self.lambda_tau = lambda_tau
        self.tau_max = tau_max if tau_max is not None else np.full(n_joints, 150.0)
        self.joint_vel_limit = joint_vel_limit

        self.adaptive_jacobian = AdaptiveVisualJacobian(
            n_joints=n_joints, n_features=n_features)
        self.interaction_matrix = ImageInteractionMatrix()

        self._history = {
            'image_error_norm': [],
            'joint_torque_rms': [],
            'instantaneous_power': [],
        }

    def compute_torque_penalty_matrix(self, tau: np.ndarray) -> np.ndarray:
        """W_τ = diag(|τᵢ|/τᵢᵐᵃˣ) ∈ ℝ^{n×n}, valores en [0,1]."""
        norm = np.abs(tau[:self.n_joints]) / (self.tau_max + 1e-8)
        return np.diag(np.clip(norm, 0.0, 1.0))

    def compute_control(
        self,
        image_error: np.ndarray,
        tau_current: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """
        Calcula q̇ con ley de control propuesta.

        Parameters
        ----------
        image_error  : e = s - s*  ∈ ℝ^m
        tau_current  : τ           ∈ ℝ^n  (Nm)

        Returns
        -------
        q_dot   : ∈ ℝ^n  (rad/s)
        metrics : dict
        """
        e = image_error.reshape(self.n_features)

        # J_a⁺ ∈ ℝ^{n × m}
        J_pinv = self.adaptive_jacobian.get_pseudoinverse()

        # W_τ ∈ ℝ^{n × n}
        W_tau = self.compute_torque_penalty_matrix(tau_current)

        # Ganancia total G = λ_s·I_n + λ_τ·W_τ  ∈ ℝ^{n × n}
        G = self.lambda_s * np.eye(self.n_joints) + self.lambda_tau * W_tau

        # q̇ = -G · J_a⁺ · e
        # J_a⁺ · e : (n,m) @ (m,) = (n,)
        # G · (...) : (n,n) @ (n,) = (n,)
        q_dot = -G @ (J_pinv @ e)

        # Saturación
        norm = np.linalg.norm(q_dot)
        if norm > self.joint_vel_limit:
            q_dot = q_dot * self.joint_vel_limit / norm

        tau_n = tau_current[:self.n_joints]
        power = float(np.dot(np.abs(tau_n), np.abs(q_dot)))
        metrics = {
            'image_error_norm': float(np.linalg.norm(e)),
            'joint_torque_rms': float(np.sqrt(np.mean(tau_n**2))),
            'instantaneous_power': power,
            'control_effort': float(np.linalg.norm(q_dot)),
            'torque_penalty_trace': float(np.trace(W_tau)),
        }
        for k in ('image_error_norm', 'joint_torque_rms', 'instantaneous_power'):
            self._history[k].append(metrics[k])

        return q_dot, metrics

    def update_jacobian(self, delta_q: np.ndarray, delta_s: np.ndarray) -> None:
        self.adaptive_jacobian.update(delta_q, delta_s)

    def get_energy_summary(self) -> dict:
        p = self._history['instantaneous_power']
        if not p:
            return {}
        arr = np.array(p)
        tau_arr = np.array(self._history['joint_torque_rms'])
        return {
            'total_energy_joules': float(np.trapezoid(arr)),
            'mean_power_watts': float(np.mean(arr)),
            'peak_power_watts': float(np.max(arr)),
            'torque_rms_mean': float(np.mean(tau_arr)),
            'iterations': len(arr),
        }
