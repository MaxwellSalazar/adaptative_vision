"""
depth_estimator_node.py
=======================
Estimación de profundidad por divergencia de flujo óptico (Contribución C2).

Principio:
    La divergencia del campo de flujo óptico θ = ∂u/∂x + ∂v/∂y
    es proporcional a la velocidad de acercamiento:

        Z_est = f · v_z / θ

    donde f es la focal length y v_z la velocidad axial de la cámara.

    Este enfoque es libre de dependencias externas (sin MiDaS, sin red),
    reproducible analíticamente, y verificable con datos sintéticos de Gazebo.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge


class DepthEstimatorNode(Node):
    """
    Nodo ROS2 que estima profundidad de N puntos de interés
    usando divergencia de flujo óptico Lucas-Kanade.

    Subscriptions:
        /camera/image_raw      (sensor_msgs/Image)
        /camera/camera_info    (sensor_msgs/CameraInfo)

    Publications:
        /vs/depth_estimates    (std_msgs/Float32MultiArray)
        /vs/optical_flow_debug (sensor_msgs/Image) — solo si debug=True
    """

    def __init__(self):
        super().__init__('depth_estimator')

        # Parámetros ROS2
        self.declare_parameter('focal_length', 554.0)
        self.declare_parameter('debug_viz', False)
        self.declare_parameter('max_corners', 50)
        self.declare_parameter('quality_level', 0.3)
        self.declare_parameter('min_distance', 20.0)

        self.focal_length = self.get_parameter('focal_length').value
        self.debug_viz = self.get_parameter('debug_viz').value
        self.max_corners = self.get_parameter('max_corners').value
        self.quality_level = self.get_parameter('quality_level').value
        self.min_distance = self.get_parameter('min_distance').value

        self.bridge = CvBridge()
        self._prev_gray = None
        self._prev_points = None
        self._cam_vz = 0.0  # velocidad axial estimada de la cámara (m/s)

        # Parámetros LK flow
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Subs y pubs
        self.sub_img = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, 10)
        self.pub_depth = self.create_publisher(
            Float32MultiArray, '/vs/depth_estimates', 10)

        if self.debug_viz:
            self.pub_debug = self.create_publisher(
                Image, '/vs/optical_flow_debug', 10)

        self.get_logger().info('DepthEstimatorNode iniciado')

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        depths = self._estimate_depths(gray, frame)

        out_msg = Float32MultiArray()
        out_msg.data = [float(d) for d in depths]
        self.pub_depth.publish(out_msg)

        self._prev_gray = gray.copy()

    def _estimate_depths(
        self, gray: np.ndarray, frame: np.ndarray
    ) -> np.ndarray:
        """
        Estima profundidades de los puntos de interés actuales.

        Retorna array de profundidades (metros). Si no hay frame previo,
        devuelve un valor por defecto (1.0 m) para bootstrap.
        """
        # Detectar corners en frame actual
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )

        if corners is None or self._prev_gray is None:
            self._prev_points = corners
            n = self.max_corners if corners is None else len(corners)
            return np.ones(n, dtype=np.float32)

        # Flujo óptico LK
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        good_new = new_pts[status == 1]
        good_old = self._prev_points[status == 1]

        if len(good_new) < 4:
            return np.ones(len(corners), dtype=np.float32)

        # Divergencia del flujo óptico
        flow = good_new - good_old  # shape (N, 1, 2) → squeeze
        flow = flow.reshape(-1, 2)
        old_pts = good_old.reshape(-1, 2)

        divergence = self._compute_divergence(old_pts, flow)

        # Z_est = f * v_z / divergence
        # v_z se aproxima de la magnitud media del flujo cuando hay movimiento puro
        v_z_proxy = float(np.mean(np.linalg.norm(flow, axis=1))) + 1e-6
        depths = self.focal_length * v_z_proxy / (np.abs(divergence) + 1e-6)
        depths = np.clip(depths, 0.1, 5.0)  # rango físico razonable en simulación

        self._prev_points = corners
        return depths.astype(np.float32)

    def _compute_divergence(
        self, points: np.ndarray, flow: np.ndarray
    ) -> np.ndarray:
        """
        Estima divergencia local del flujo óptico en cada punto.

        Aproximación: para cada punto p_i, busca vecinos dentro de un radio
        y calcula ∂u/∂x + ∂v/∂y por diferencias finitas.
        """
        n = len(points)
        divergences = np.zeros(n)
        radius = 40.0  # píxeles

        for i in range(n):
            # Vecinos
            diffs = points - points[i]
            dists = np.linalg.norm(diffs, axis=1)
            mask = (dists > 1e-3) & (dists < radius)

            if mask.sum() < 2:
                divergences[i] = 1e-6
                continue

            neighbors_pos = diffs[mask]
            neighbors_flow = flow[mask] - flow[i]

            # Ajuste lineal: du ≈ a*dx + b*dy, dv ≈ c*dx + d*dy
            # divergence ≈ a + d
            try:
                A = neighbors_pos  # (k, 2)
                b_u = neighbors_flow[:, 0]
                b_v = neighbors_flow[:, 1]
                coeff_u, _, _, _ = np.linalg.lstsq(A, b_u, rcond=None)
                coeff_v, _, _, _ = np.linalg.lstsq(A, b_v, rcond=None)
                divergences[i] = coeff_u[0] + coeff_v[1]
            except Exception:
                divergences[i] = 1e-6

        return divergences


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
