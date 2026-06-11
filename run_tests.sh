#!/bin/bash
# =============================================================================
# run_tests.sh — Ejecuta tests unitarios y verifica el entorno
# =============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_ROOT/.venv"

# ── Activar entorno virtual ───────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "ERROR: Entorno virtual no encontrado. Ejecuta primero:"
    echo "    bash setup_venv.sh"
    exit 1
fi

source "$VENV/bin/activate"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   adaptive_visual_servo — Suite de Tests                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Tests unitarios ───────────────────────────────────────────────────────────
echo ">>> Ejecutando tests unitarios..."
pytest "$PROJECT_ROOT/tests/" \
    -v \
    --tb=short \
    --cov="$PROJECT_ROOT/src/adaptive_visual_servo/adaptive_visual_servo" \
    --cov-report=term-missing \
    --cov-report=html:"$PROJECT_ROOT/docs/coverage_html" \
    2>&1

echo ""
echo ">>> Análisis de sensibilidad (sin ROS2, solo Python)..."
python "$PROJECT_ROOT/scripts/sensitivity_analysis.py"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Tests completados. Reporte en docs/coverage_html/         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
