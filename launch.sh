#!/usr/bin/env bash
# =============================================================================
# Запуск веб-интерфейса Hunyuan3D 2.1
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[ERR] Виртуальное окружение не найдено: $VENV_DIR"
    echo "      Сначала запусти ./install.sh"
    exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Оптимизации для Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export GRADIO_PORT="${GRADIO_PORT:-7860}"
# Уменьшает предварительный аллок на MPS (иногда спасает при 24 GB)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

echo "=============================================="
echo "  Hunyuan3D 2.1 — Web UI"
echo "  Проект: $PROJECT_DIR"
echo "  URL:    http://localhost:${GRADIO_PORT}"
echo "=============================================="

cd "$PROJECT_DIR"
exec python gradio_app.py
