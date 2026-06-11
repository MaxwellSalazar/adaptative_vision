"""
sensitivity_analysis.py
=======================
Análisis de sensibilidad del parámetro λ_τ (Figura 4 del paper).

Ejecuta el controlador propuesto en modo simulación pura (sin ROS2)
variando λ_τ en el rango [0.0, 0.5] y registra:
  - Error de seguimiento final (px)
  - Energía total consumida (J)
  - Velocidad de convergencia (iteraciones hasta < 5px)

Genera:
  - fig4_sensitivity.pdf / .png
  - sensitivity_data.csv

Uso:
    cd adaptive_vs_ws
    source .venv/bin/activate
    python scripts/sensitivity_analysis.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../src/adaptive_visual_servo'))

from adaptive_visual_servo.adaptive_jacobian import TorquePenalizedController

# ── Estilo publicación ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
})


def simulate_experiment(
    lambda_s: float,
    lambda_tau: float,
    n_steps: int = 500,
    tau_level: float = 60.0,
    seed: int = 0,
) -> dict:
    """
    Simula el controlador en lazo cerrado en espacio imagen (sin ROS2).

    Modelo simplificado de la dinámica imagen:
        s(k+1) = s(k) + J_true · q̇(k) · dt + ruido

    donde J_true es un Jacobiano verdadero fijo (desconocido para el controlador).
    """
    rng = np.random.default_rng(seed)

    # Jacobiano verdadero (representa la relación imagen-robot real)
    J_true = rng.normal(0, 0.05, (8, 6))

    ctrl = TorquePenalizedController(
        n_joints=6,
        lambda_s=lambda_s,
        lambda_tau=lambda_tau,
        tau_max=np.full(6, 100.0),
        joint_vel_limit=1.0,
    )

    # Estado inicial: objeto fuera del centro (error ~100px)
    s = rng.uniform(50, 150, 8).astype(np.float32)
    s_star = np.zeros(8, dtype=np.float32)  # features deseadas = centro

    # Torque simulado: combina componente estática + dinámica
    tau_base = rng.uniform(0.3, 1.0, 6) * tau_level

    errors, powers, q_dots = [], [], []
    converged_at = None

    dt = 1.0 / 30.0  # 30 Hz

    for k in range(n_steps):
        error = s - s_star

        # Torque varía suavemente con el movimiento
        tau = tau_base * (1.0 + 0.1 * np.sin(0.1 * k * np.ones(6)))

        q_dot, metrics = ctrl.compute_control(error, tau)

        # Actualizar Jacobiano adaptativo
        if k > 0:
            delta_s_meas = s - s_prev
            if np.linalg.norm(delta_s_meas) > 0.1:
                ctrl.update_jacobian(q_dot * dt, delta_s_meas)

        s_prev = s.copy()

        # Dinámica imagen: s(k+1) = s(k) + J_true·q̇·dt + ruido
        s = s + (J_true @ q_dot) * dt + rng.normal(0, 0.3, 8)

        errors.append(metrics['image_error_norm'])
        powers.append(metrics['instantaneous_power'])
        q_dots.append(metrics['control_effort'])

        if metrics['image_error_norm'] < 5.0 and converged_at is None:
            converged_at = k

    summary = ctrl.get_energy_summary()

    return {
        'lambda_tau': lambda_tau,
        'final_error_px': float(errors[-1]),
        'mean_error_px': float(np.mean(errors[-50:])),  # últimos 50 pasos
        'total_energy_J': summary.get('total_energy_joules', 0.0),
        'mean_power_W': summary.get('mean_power_watts', 0.0),
        'converged_at': converged_at if converged_at is not None else n_steps,
        'converged': converged_at is not None,
        'errors': errors,
        'powers': powers,
    }


def main():
    output_dir = os.path.expanduser('~/vs_results/figures')
    os.makedirs(output_dir, exist_ok=True)

    lambda_s = 0.5
    lambda_tau_values = np.linspace(0.0, 0.5, 20)
    n_runs = 5  # promedio sobre múltiples seeds para robustez estadística

    print(f'Analizando {len(lambda_tau_values)} valores de λ_τ '
          f'× {n_runs} repeticiones...')

    records = []
    curves = {}

    for lam in lambda_tau_values:
        run_results = [
            simulate_experiment(lambda_s, lam, seed=s)
            for s in range(n_runs)
        ]
        mean_error = np.mean([r['final_error_px'] for r in run_results])
        std_error = np.std([r['final_error_px'] for r in run_results])
        mean_energy = np.mean([r['total_energy_J'] for r in run_results])
        std_energy = np.std([r['total_energy_J'] for r in run_results])
        mean_conv = np.mean([r['converged_at'] for r in run_results])
        conv_rate = np.mean([r['converged'] for r in run_results])

        records.append({
            'lambda_tau': round(float(lam), 4),
            'final_error_mean_px': round(float(mean_error), 4),
            'final_error_std_px': round(float(std_error), 4),
            'total_energy_mean_J': round(float(mean_energy), 4),
            'total_energy_std_J': round(float(std_energy), 4),
            'convergence_iter_mean': round(float(mean_conv), 1),
            'convergence_rate': round(float(conv_rate), 3),
        })

        # Guardar curvas del punto medio para Fig 4c
        mid_idx = len(run_results) // 2
        curves[float(lam)] = run_results[mid_idx]['errors']

        print(f'  λ_τ={lam:.3f}  err={mean_error:.1f}px  '
              f'E={mean_energy:.1f}J  conv={conv_rate*100:.0f}%')

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'sensitivity_data.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nDatos guardados: {csv_path}')

    # ── Figura 4: 3 subplots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    lams = df['lambda_tau'].values

    # 4a: Error final vs λ_τ
    ax = axes[0]
    ax.plot(lams, df['final_error_mean_px'], 'o-', color='#1D9E75', ms=4)
    ax.fill_between(
        lams,
        df['final_error_mean_px'] - df['final_error_std_px'],
        df['final_error_mean_px'] + df['final_error_std_px'],
        alpha=0.2, color='#1D9E75')
    ax.axhline(5.0, color='gray', linestyle='--', linewidth=1,
               label='Umbral 5px')
    ax.set_xlabel('λ_τ (peso de penalización)')
    ax.set_ylabel('Error final (px)')
    ax.set_title('(a) Precisión de seguimiento')
    ax.legend()

    # 4b: Energía total vs λ_τ
    ax = axes[1]
    ax.plot(lams, df['total_energy_mean_J'], 's-', color='#E24B4A', ms=4)
    ax.fill_between(
        lams,
        df['total_energy_mean_J'] - df['total_energy_std_J'],
        df['total_energy_mean_J'] + df['total_energy_std_J'],
        alpha=0.2, color='#E24B4A')
    ax.set_xlabel('λ_τ (peso de penalización)')
    ax.set_ylabel('Energía total (J)')
    ax.set_title('(b) Consumo energético')

    # Marcar punto óptimo (menor energía con error < umbral)
    valid = df[df['final_error_mean_px'] < 10.0]
    if len(valid) > 0:
        opt_idx = valid['total_energy_mean_J'].idxmin()
        opt_lam = valid.loc[opt_idx, 'lambda_tau']
        opt_e = valid.loc[opt_idx, 'total_energy_mean_J']
        ax.scatter([opt_lam], [opt_e], color='gold', s=100, zorder=5,
                   label=f'Óptimo λ_τ={opt_lam:.2f}')
        ax.legend()

    # 4c: Curvas de convergencia para 4 valores de λ_τ
    ax = axes[2]
    selected_lams = [0.0, 0.1, 0.2, 0.4]
    palette = ['#E24B4A', '#1D9E75', '#185FA5', '#BA7517']
    for lam_val, color in zip(selected_lams, palette):
        # Buscar clave más cercana
        closest = min(curves.keys(), key=lambda x: abs(x - lam_val))
        errs = curves[closest]
        ax.plot(errs, color=color, alpha=0.8,
                label=f'λ_τ={lam_val:.1f}', linewidth=1.2)
    ax.axhline(5.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Iteración')
    ax.set_ylabel('‖e‖ (px)')
    ax.set_title('(c) Curvas de convergencia')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 300)

    fig.suptitle(
        'Análisis de sensibilidad: efecto de λ_τ en precisión y energía',
        fontsize=12, y=1.02)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'fig4_sensitivity.{ext}')
        fig.savefig(path)
        print(f'Guardada: {path}')

    plt.close(fig)

    # ── Resumen para el paper ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  HALLAZGO PRINCIPAL (para sección de resultados del paper)')
    print('='*60)
    baseline_e = df[df['lambda_tau'] < 0.01]['total_energy_mean_J'].values
    if len(valid) > 0 and len(baseline_e) > 0:
        opt_energy = valid.loc[opt_idx, 'total_energy_mean_J']
        reduction = (baseline_e[0] - opt_energy) / (baseline_e[0] + 1e-8) * 100
        print(f'  λ_τ óptimo:         {opt_lam:.2f}')
        print(f'  Reducción energía:  {reduction:.1f}%')
        print(f'  Error final:        {valid.loc[opt_idx, "final_error_mean_px"]:.2f} px')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
