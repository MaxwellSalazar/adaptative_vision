#!/bin/bash
# =============================================================================
# setup_venv.sh — Crea y configura el entorno virtual Python del proyecto
# =============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo ">>> Creando entorno virtual en $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo ">>> Activando entorno virtual"
source "$VENV_DIR/bin/activate"

echo ">>> Actualizando pip"
pip install --upgrade pip setuptools wheel

echo ">>> Instalando dependencias del proyecto"
pip install -r "$PROJECT_ROOT/requirements.txt"

echo ">>> Instalando paquete en modo editable"
pip install -e "$PROJECT_ROOT/src/adaptive_visual_servo"

echo ""
echo "============================================================"
echo "  Entorno virtual listo."
echo "  Para activarlo:  source .venv/bin/activate"
echo "  Para VSCode:     seleccionar .venv/bin/python como intérprete"
echo "============================================================"
