"""
analyze_results.py
==================
Script de análisis y generación de figuras para el paper.

Genera:
  - Fig 1: Curvas de convergencia (error imagen vs tiempo)
  - Fig 2: Comparativa de consumo energético (baseline vs propuesto)
  - Fig 3: Torque RMS por junta
  - Fig 4: Análisis de sensibilidad λ_τ
  - Tabla 1: Métricas numéricas resumen

Uso:
    python scripts/analyze_results.py \
        --baseline  ~/vs_results/metrics_static_baseline_*.csv \
        --proposed  ~/vs_results/metrics_static_proposed_*.csv \
        --output    ~/vs_results/figures/
"""

import argparse
import os
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Estilo para publicación ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'baseline': '#E24B4A',
    'proposed': '#1D9E75',
    'shade_b': '#FCEBEB',
    'shade_p': '#E1F5EE',
}


def load_csv(path_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f'No CSV encontrado: {path_pattern}')
    df = pd.read_csv(files[-1])  # Usar el más reciente
    print(f'  Cargado: {files[-1]} ({len(df)} filas)')
    return df


def smooth(series: np.ndarray, window: int = 15) -> np.ndarray:
    """Suavizado con media móvil para visualización."""
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode='valid')


def fig_convergence(df_b: pd.DataFrame, df_p: pd.DataFrame, out: str):
    """Fig 1: Curvas de convergencia del error imagen."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    t_b = df_b['timestamp_s'].values
    t_p = df_p['timestamp_s'].values
    e_b = df_b['image_error_norm_px'].values
    e_p = df_p['image_error_norm_px'].values

    # Suavizado
    win = 20
    t_b_s = t_b[win-1:]
    t_p_s = t_p[win-1:]

    ax.plot(t_b_s, smooth(e_b, win), color=COLORS['baseline'],
            label='IBVS clásico (baseline)', zorder=3)
    ax.plot(t_p_s, smooth(e_p, win), color=COLORS['proposed'],
            label='IBVS propuesto (λ_τ=0.1)', zorder=3)
    ax.fill_between(t_b_s, smooth(e_b, win), alpha=0.15,
                    color=COLORS['baseline'])
    ax.fill_between(t_p_s, smooth(e_p, win), alpha=0.15,
                    color=COLORS['proposed'])

    ax.axhline(5.0, color='gray', linestyle='--', linewidth=1,
               label='Umbral convergencia (5 px)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('‖e‖ (píxeles)')
    ax.set_title('Convergencia del error en espacio imagen')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig1_convergence.pdf'))
    fig.savefig(os.path.join(out, 'fig1_convergence.png'))
    plt.close(fig)
    print('  fig1_convergence guardada')


def fig_energy(df_b: pd.DataFrame, df_p: pd.DataFrame, out: str):
    """Fig 2: Comparativa de consumo energético."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Potencia instantánea
    ax = axes[0]
    t_b = df_b['timestamp_s'].values
    t_p = df_p['timestamp_s'].values
    p_b = df_b['instantaneous_power_W'].values
    p_p = df_p['instantaneous_power_W'].values
    win = 20
    ax.plot(t_b[win-1:], smooth(p_b, win), color=COLORS['baseline'],
            label='Baseline')
    ax.plot(t_p[win-1:], smooth(p_p, win), color=COLORS['proposed'],
            label='Propuesto')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Potencia (W)')
    ax.set_title('Potencia instantánea')
    ax.legend()

    # Energía acumulada
    ax = axes[1]
    e_accum_b = np.cumsum(p_b) * np.mean(np.diff(t_b)) if len(t_b) > 1 else p_b
    e_accum_p = np.cumsum(p_p) * np.mean(np.diff(t_p)) if len(t_p) > 1 else p_p
    ax.plot(t_b, e_accum_b, color=COLORS['baseline'], label='Baseline')
    ax.plot(t_p, e_accum_p, color=COLORS['proposed'], label='Propuesto')

    # Anotación de reducción
    if len(e_accum_b) > 0 and len(e_accum_p) > 0:
        red_pct = (e_accum_b[-1] - e_accum_p[-1]) / (e_accum_b[-1] + 1e-8) * 100
        ax.annotate(f'↓ {red_pct:.1f}% energía',
                    xy=(t_p[-1], e_accum_p[-1]),
                    xytext=(-80, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=COLORS['proposed']),
                    color=COLORS['proposed'], fontweight='bold')

    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Energía acumulada (J)')
    ax.set_title('Energía acumulada')
    ax.legend()

    fig.suptitle('Comparativa energética: IBVS clásico vs propuesto',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig2_energy.pdf'))
    fig.savefig(os.path.join(out, 'fig2_energy.png'))
    plt.close(fig)
    print('  fig2_energy guardada')


def fig_torque_rms(df_b: pd.DataFrame, df_p: pd.DataFrame, out: str):
    """Fig 3: Torque RMS comparativo."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    t_b = df_b['timestamp_s'].values
    t_p = df_p['timestamp_s'].values
    tau_b = df_b['joint_torque_rms_Nm'].values
    tau_p = df_p['joint_torque_rms_Nm'].values
    win = 20

    ax.plot(t_b[win-1:], smooth(tau_b, win), color=COLORS['baseline'],
            label='Baseline')
    ax.plot(t_p[win-1:], smooth(tau_p, win), color=COLORS['proposed'],
            label='Propuesto')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('τ RMS (Nm)')
    ax.set_title('Torque articular RMS')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig3_torque_rms.pdf'))
    fig.savefig(os.path.join(out, 'fig3_torque_rms.png'))
    plt.close(fig)
    print('  fig3_torque_rms guardada')


def print_summary_table(df_b: pd.DataFrame, df_p: pd.DataFrame):
    """Tabla 1 de métricas numéricas para el paper."""
    print('\n' + '='*65)
    print('  TABLA 1: Resumen de métricas')
    print('='*65)
    print(f"{'Métrica':<35} {'Baseline':>12} {'Propuesto':>12} {'Δ%':>8}")
    print('-'*65)

    def row(name, col, fmt='.2f', lower_is_better=True):
        b = df_b[col].mean() if col in df_b else float('nan')
        p = df_p[col].mean() if col in df_p else float('nan')
        delta = (b - p) / (abs(b) + 1e-8) * 100
        sign = '↓' if (lower_is_better and delta > 0) else ('↑' if delta > 0 else '↓')
        print(f"  {name:<33} {b:>12{fmt}} {p:>12{fmt}} {sign}{abs(delta):>6.1f}%")

    row('Error imagen medio (px)',   'image_error_norm_px')
    row('Torque RMS medio (Nm)',     'joint_torque_rms_Nm')
    row('Potencia media (W)',        'instantaneous_power_W')
    row('Esfuerzo de control (r/s)', 'control_effort_rad_s')
    print('='*65 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Análisis de resultados VS')
    parser.add_argument('--baseline', required=True,
                        help='Patrón glob del CSV baseline')
    parser.add_argument('--proposed', required=True,
                        help='Patrón glob del CSV propuesto')
    parser.add_argument('--output', default=os.path.expanduser('~/vs_results/figures'),
                        help='Directorio de salida de figuras')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print('Cargando datos...')
    df_b = load_csv(args.baseline)
    df_p = load_csv(args.proposed)

    print('Generando figuras...')
    fig_convergence(df_b, df_p, args.output)
    fig_energy(df_b, df_p, args.output)
    fig_torque_rms(df_b, df_p, args.output)
    print_summary_table(df_b, df_p)

    print(f'\nFiguras guardadas en: {args.output}')


if __name__ == '__main__':
    main()
