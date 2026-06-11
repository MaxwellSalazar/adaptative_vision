"""
feature_tracker.py
==================
Nodo ROS2 de detección y seguimiento de features en el espacio imagen.

Detecta el objeto objetivo por color HSV (configurable) y extrae
4 puntos de interés en su contorno. Estos 4 puntos forman el vector
de features s = [u1,v1, u2,v2, u3,v3, u4,v4] ∈ ℝ^8 usado por el controlador.

Subscriptions:
    /camera/image_raw     (sensor_msgs/Image)

Publications:
    /vs/features_current  (std_msgs/Float32MultiArray) — features actuales s
    /vs/features_desired  (std_msgs/Float32MultiArray) — features deseadas s*
    /vs/feature_image     (sensor_msgs/Image)           — debug visualization
    /vs/target_detected   (std_msgs/Bool)
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from cv_bridge import CvBridge


class FeatureTrackerNode(Node):

    def __init__(self):
        super().__init__('feature_tracker')

        # Parámetros configurables
        self.declare_parameter('hsv_lower', [20, 100, 100])   # naranja por defecto
        self.declare_parameter('hsv_upper', [40, 255, 255])
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('desired_u', 320.0)  # centro imagen
        self.declare_parameter('desired_v', 240.0)
        self.declare_parameter('desired_size_px', 80.0)  # tamaño deseado del objeto

        self.hsv_lower = np.array(self.get_parameter('hsv_lower').value)
        self.hsv_upper = np.array(self.get_parameter('hsv_upper').value)
        self.img_w = self.get_parameter('image_width').value
        self.img_h = self.get_parameter('image_height').value

        # Features deseadas s* = 4 puntos alrededor del centro deseado
        desired_u = self.get_parameter('desired_u').value
        desired_v = self.get_parameter('desired_v').value
        half = self.get_parameter('desired_size_px').value / 2
        self._features_desired = np.array([
            desired_u - half, desired_v - half,
            desired_u + half, desired_v - half,
            desired_u + half, desired_v + half,
            desired_u - half, desired_v + half,
        ], dtype=np.float32)

        self.bridge = CvBridge()
        self._target_detected = False
        self._last_features = np.zeros(8, dtype=np.float32)

        # Subscriptions y publicaciones
        self.sub_img = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, 10)

        self.pub_feat_curr = self.create_publisher(
            Float32MultiArray, '/vs/features_current', 10)
        self.pub_feat_des = self.create_publisher(
            Float32MultiArray, '/vs/features_desired', 10)
        self.pub_detected = self.create_publisher(
            Bool, '/vs/target_detected', 10)
        self.pub_debug_img = self.create_publisher(
            Image, '/vs/feature_image', 10)

        self.get_logger().info('FeatureTrackerNode iniciado')
        self._publish_desired_features()

    def _publish_desired_features(self):
        msg = Float32MultiArray()
        msg.data = self._features_desired.tolist()
        self.pub_feat_des.publish(msg)

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        features, debug_frame = self._detect_and_track(frame)

        detected = features is not None
        self._target_detected = detected

        if detected:
            self._last_features = features
        else:
            # Mantener últimas features conocidas (degrada suavemente)
            features = self._last_features

        # Publicar features actuales
        feat_msg = Float32MultiArray()
        feat_msg.data = features.tolist()
        self.pub_feat_curr.publish(feat_msg)

        # Publicar estado de detección
        det_msg = Bool()
        det_msg.data = detected
        self.pub_detected.publish(det_msg)

        # Publicar imagen debug
        debug_ros = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        self.pub_debug_img.publish(debug_ros)

        # Republica features deseadas periódicamente
        self._publish_desired_features()

    def _detect_and_track(
        self, frame: np.ndarray
    ):
        """
        Detecta el objeto por color HSV y extrae 4 puntos de su bounding box.

        Returns
        -------
        features : np.ndarray shape (8,) o None si no detectado
        debug    : frame anotado para visualización
        """
        debug = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morfología para limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cv2.putText(debug, 'No target', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return None, debug

        # Contorno más grande
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 200:
            return None, debug

        # Bounding box → 4 features (esquinas)
        x, y, w, h = cv2.boundingRect(cnt)
        features = np.array([
            float(x),     float(y),
            float(x + w), float(y),
            float(x + w), float(y + h),
            float(x),     float(y + h),
        ], dtype=np.float32)

        # Dibujar en debug
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        corners = features.reshape(4, 2).astype(int)
        des_corners = self._features_desired.reshape(4, 2).astype(int)
        for pt, dp in zip(corners, des_corners):
            cv2.circle(debug, tuple(pt), 5, (0, 255, 0), -1)
            cv2.circle(debug, tuple(dp), 5, (0, 0, 255), -1)
            cv2.line(debug, tuple(pt), tuple(dp), (255, 255, 0), 1)

        # Error imagen
        err = np.linalg.norm(features - self._features_desired)
        cv2.putText(debug, f'err={err:.1f}px', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return features, debug


def main(args=None):
    rclpy.init(args=args)
    node = FeatureTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
