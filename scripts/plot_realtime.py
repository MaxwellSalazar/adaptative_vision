"""
plot_realtime.py
================
Visualización en tiempo real de las métricas del controlador VS.

Lee los tópicos ROS2 /vs/metrics y /vs/energy_metrics y grafica
en vivo durante la simulación. Útil para monitorear la convergencia
y el consumo energético mientras corre el experimento.

Uso (en terminal separado mientras corre simulation.launch.py):
    cd adaptive_vs_ws
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    source .venv/bin/activate
    python scripts/plot_realtime.py
"""

import sys
import os
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ROS2 import (opcional: si no está disponible, genera datos de prueba)
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("AVISO: rclpy no disponible. Ejecutando en modo demo con datos sintéticos.")


# ── Configuración de estilo ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 9,
    'axes.labelsize': 10,
    'figure.facecolor': '#1E1E1E',
    'axes.facecolor': '#252526',
    'axes.edgecolor': '#555555',
    'axes.labelcolor': '#CCCCCC',
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'grid.color': '#333333',
    'text.color': '#CCCCCC',
    'lines.linewidth': 1.5,
})

WINDOW = 300  # puntos visibles en pantalla
COLORS = {
    'error': '#4EC9B0',
    'power': '#CE9178',
    'torque': '#9CDCFE',
    'energy': '#DCDCAA',
}


class MetricsBuffer:
    """Buffer thread-safe para métricas en tiempo real."""

    def __init__(self, maxlen: int = WINDOW):
        self.lock = threading.Lock()
        self.error = deque(maxlen=maxlen)
        self.power = deque(maxlen=maxlen)
        self.torque = deque(maxlen=maxlen)
        self.energy_total = deque(maxlen=maxlen)
        self.t = deque(maxlen=maxlen)
        self._t0 = time.time()

    def push_metrics(self, data):
        with self.lock:
            t = time.time() - self._t0
            self.t.append(t)
            self.error.append(data[0] if len(data) > 0 else 0)
            self.torque.append(data[1] if len(data) > 1 else 0)
            self.power.append(data[2] if len(data) > 2 else 0)

    def push_energy(self, data):
        with self.lock:
            if self.t:
                self.energy_total.append(data[0] if len(data) > 0 else 0)

    def snapshot(self):
        with self.lock:
            return {
                't': list(self.t),
                'error': list(self.error),
                'power': list(self.power),
                'torque': list(self.torque),
                'energy_total': list(self.energy_total),
            }


buffer = MetricsBuffer()


# ── Modo ROS2 ─────────────────────────────────────────────────────────────────

if ROS2_AVAILABLE:
    class MetricsSubscriberNode(Node):
        def __init__(self):
            super().__init__('realtime_plotter')
            self.sub_m = self.create_subscription(
                Float32MultiArray, '/vs/metrics',
                lambda msg: buffer.push_metrics(msg.data), 10)
            self.sub_e = self.create_subscription(
                Float32MultiArray, '/vs/energy_metrics',
                lambda msg: buffer.push_energy(msg.data), 10)

    def ros2_spin_thread():
        rclpy.init()
        node = MetricsSubscriberNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()


# ── Modo demo (sin ROS2) ──────────────────────────────────────────────────────

def demo_data_thread():
    """Genera datos sintéticos para probar la visualización sin ROS2."""
    t = 0.0
    energy = 0.0
    error = 120.0
    while True:
        error = max(2.0, error * 0.995 + np.random.normal(0, 0.5))
        power = 15.0 + 10.0 * np.exp(-t * 0.02) + np.random.normal(0, 1)
        torque = 40.0 + 20.0 * np.exp(-t * 0.01) + np.random.normal(0, 2)
        energy += power * (1/30)
        buffer.push_metrics([error, torque, power])
        buffer.push_energy([energy])
        t += 1/30
        time.sleep(1/30)


# ── Figura ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle('adaptive_visual_servo — Monitor en tiempo real', fontsize=11,
             color='#CCCCCC')

ax_err, ax_pow, ax_tau, ax_eng = axes.flatten()

line_err, = ax_err.plot([], [], color=COLORS['error'])
line_pow, = ax_pow.plot([], [], color=COLORS['power'])
line_tau, = ax_tau.plot([], [], color=COLORS['torque'])
line_eng, = ax_eng.plot([], [], color=COLORS['energy'])

for ax, title, ylabel in [
    (ax_err, 'Error imagen ‖e‖', 'px'),
    (ax_pow, 'Potencia instantánea', 'W'),
    (ax_tau, 'Torque articular RMS', 'Nm'),
    (ax_eng, 'Energía acumulada', 'J'),
]:
    ax.set_title(title, color='#CCCCCC', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('t (s)')
    ax.grid(True)

# Línea de umbral de convergencia
ax_err.axhline(5.0, color='#FF6B6B', linestyle='--', linewidth=1,
               alpha=0.7, label='5px')
ax_err.legend(fontsize=8)

# Textos de valor actual
txt_err = ax_err.text(0.97, 0.95, '', transform=ax_err.transAxes,
                      ha='right', va='top', color=COLORS['error'],
                      fontsize=9, fontfamily='monospace')
txt_pow = ax_pow.text(0.97, 0.95, '', transform=ax_pow.transAxes,
                      ha='right', va='top', color=COLORS['power'],
                      fontsize=9, fontfamily='monospace')

plt.tight_layout(rect=[0, 0, 1, 0.95])


def update(frame):
    data = buffer.snapshot()
    if not data['t']:
        return line_err, line_pow, line_tau, line_eng

    t = data['t']

    def set_line(line, ax, x, y, text_widget=None, fmt='.1f'):
        if not y:
            return
        line.set_data(x, y)
        ax.relim()
        ax.autoscale_view()
        if text_widget:
            text_widget.set_text(f'{y[-1]:{fmt}}')

    set_line(line_err, ax_err, t, data['error'], txt_err, '.1f')
    set_line(line_pow, ax_pow, t, data['power'], txt_pow, '.1f')
    set_line(line_tau, ax_tau, t, data['torque'])

    # Energía acumulada: usar longitud de t como eje si hay desfase
    if data['energy_total']:
        t_e = t[:len(data['energy_total'])]
        set_line(line_eng, ax_eng, t_e, data['energy_total'])

    return line_err, line_pow, line_tau, line_eng


def main():
    # Arrancar hilo de datos
    if ROS2_AVAILABLE:
        t = threading.Thread(target=ros2_spin_thread, daemon=True)
    else:
        t = threading.Thread(target=demo_data_thread, daemon=True)
    t.start()

    ani = animation.FuncAnimation(
        fig, update, interval=100, blit=True, cache_frame_data=False)

    print("Monitor activo. Cierra la ventana para salir.")
    plt.show()


if __name__ == '__main__':
    main()
