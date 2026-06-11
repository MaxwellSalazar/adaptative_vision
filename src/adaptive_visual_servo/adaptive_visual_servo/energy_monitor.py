"""
energy_monitor.py
=================
Nodo de monitoreo y registro de métricas energéticas para publicación.

Suscribe a /vs/metrics y /vs/energy_metrics, guarda CSV con todas las
métricas por iteración, y genera reportes de comparación baseline vs propuesto.

Publications:
    /vs/energy_report  (std_msgs/String) — JSON con resumen

Output files:
    ~/vs_results/metrics_<timestamp>.csv
    ~/vs_results/energy_summary_<timestamp>.json
"""

import rclpy
from rclpy.node import Node
import numpy as np
import csv
import json
import os
from datetime import datetime

from std_msgs.msg import Float32MultiArray, String


METRIC_COLS = [
    'timestamp_s',
    'image_error_norm_px',
    'joint_torque_rms_Nm',
    'instantaneous_power_W',
    'control_effort_rad_s',
    'torque_penalty_trace',
]


class EnergyMonitorNode(Node):

    def __init__(self):
        super().__init__('energy_monitor')

        self.declare_parameter('output_dir', os.path.expanduser('~/vs_results'))
        self.declare_parameter('experiment_name', 'run')

        self.output_dir = self.get_parameter('output_dir').value
        exp_name = self.get_parameter('experiment_name').value
        os.makedirs(self.output_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._csv_path = os.path.join(
            self.output_dir, f'metrics_{exp_name}_{ts}.csv')
        self._json_path = os.path.join(
            self.output_dir, f'energy_summary_{exp_name}_{ts}.json')

        # Abrir CSV e inicializar
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=METRIC_COLS)
        self._csv_writer.writeheader()

        self._start_time = self.get_clock().now()
        self._rows: list = []

        # Subs
        self.sub_metrics = self.create_subscription(
            Float32MultiArray, '/vs/metrics', self._metrics_cb, 10)
        self.sub_energy = self.create_subscription(
            Float32MultiArray, '/vs/energy_metrics', self._energy_cb, 10)

        # Pub
        self.pub_report = self.create_publisher(
            String, '/vs/energy_report', 10)

        self.get_logger().info(
            f'EnergyMonitorNode iniciado | CSV: {self._csv_path}')

    def _metrics_cb(self, msg: Float32MultiArray):
        now_s = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        data = list(msg.data) + [0.0] * max(0, 5 - len(msg.data))
        row = dict(zip(METRIC_COLS, [now_s] + data[:5]))
        self._csv_writer.writerow(row)
        self._rows.append(row)

    def _energy_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 4:
            return
        summary = {
            'total_energy_J': round(msg.data[0], 4),
            'mean_power_W': round(msg.data[1], 4),
            'peak_power_W': round(msg.data[2], 4),
            'torque_rms_mean_Nm': round(msg.data[3], 4),
            'iterations': int(msg.data[4]) if len(msg.data) > 4 else len(self._rows),
        }

        # Calcular métricas adicionales desde CSV en memoria
        if self._rows:
            errors = [r['image_error_norm_px'] for r in self._rows]
            summary['tracking_error_mean_px'] = round(float(np.mean(errors)), 4)
            summary['tracking_error_std_px'] = round(float(np.std(errors)), 4)
            summary['tracking_error_final_px'] = round(errors[-1], 4)

        report_str = json.dumps(summary, indent=2)

        rep_msg = String()
        rep_msg.data = report_str
        self.pub_report.publish(rep_msg)

        # Guardar JSON
        with open(self._json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(f'[Report] {report_str}')

    def destroy_node(self):
        self._csv_file.close()
        self.get_logger().info(
            f'EnergyMonitor: CSV guardado en {self._csv_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EnergyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
