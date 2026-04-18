#!/usr/bin/env bash
# =============================================================================
# Hunyuan3D 2.1 — автоматическая установка для macOS (Apple Silicon)
# Адаптировано под Mac с M1/M2/M3/M4 (рекомендуется >= 16 GB unified memory)
# =============================================================================
set -euo pipefail

# ---- Конфигурация ----------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${PROJECT_DIR}/Hunyuan3D-2.1"
VENV_DIR="${PROJECT_DIR}/venv"
MODELS_DIR="${PROJECT_DIR}/weights"
REPO_URL="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
HF_MODEL_ID="tencent/Hunyuan3D-2.1"
PYTHON_MIN_VERSION="3.10"

# ---- Цветной вывод ---------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR ]${NC}  $*" >&2; }

# ---- Проверки окружения ----------------------------------------------------
info "Проверка окружения macOS..."

if [[ "$(uname)" != "Darwin" ]]; then
    err "Скрипт предназначен для macOS. Обнаружена система: $(uname)"
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    warn "Обнаружена архитектура $ARCH. Apple Silicon (arm64) работает лучше всего."
fi
ok "macOS $(sw_vers -productVersion), архитектура: $ARCH"

# Xcode Command Line Tools
if ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools не установлены. Устанавливаю..."
    xcode-select --install || true
    err "После завершения установки Xcode CLT запусти скрипт заново."
    exit 1
fi
ok "Xcode Command Line Tools: $(xcode-select -p)"

# Homebrew
if ! command -v brew >/dev/null 2>&1; then
    err "Homebrew не найден. Установи: https://brew.sh"
    exit 1
fi
ok "Homebrew: $(brew --version | head -1)"

# Python 3.10+
find_python() {
    for v in 3.12 3.11 3.10; do
        if command -v "python${v}" >/dev/null 2>&1; then
            echo "python${v}"; return 0
        fi
    done
    return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    info "Python 3.10-3.12 не найден. Устанавливаю python@3.11 через Homebrew..."
    brew install python@3.11
    PYTHON_BIN="python3.11"
fi
ok "Python: $(${PYTHON_BIN} --version)"

# Git
command -v git >/dev/null 2>&1 || { err "git не найден. Установи: brew install git"; exit 1; }
ok "Git: $(git --version)"

# git-lfs (для больших весов модели)
if ! command -v git-lfs >/dev/null 2>&1; then
    info "Устанавливаю git-lfs..."
    brew install git-lfs
fi
git lfs install >/dev/null 2>&1 || true
ok "git-lfs готов"

# ---- Системная память ------------------------------------------------------
MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
info "Системная память: ${MEM_GB} GB"
if (( MEM_GB < 16 )); then
    warn "Рекомендуется минимум 16 GB unified memory. Возможны проблемы с запуском."
elif (( MEM_GB < 24 )); then
    warn "У тебя ${MEM_GB} GB. Возможно потребуется использовать fp16/низкое разрешение."
else
    ok "${MEM_GB} GB unified memory — достаточно для shape-генерации"
fi

# ---- Клонирование репозитория ---------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
    info "Клонирую Hunyuan3D 2.1 из $REPO_URL ..."
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    ok "Репозиторий склонирован в $REPO_DIR"
else
    info "Репозиторий уже существует, пропускаю клонирование."
    ( cd "$REPO_DIR" && git pull --ff-only ) || warn "Не удалось обновить, продолжаю с текущей версией."
fi

# ---- Виртуальное окружение -------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    info "Создаю виртуальное окружение: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
ok "venv активирован: $(python --version)"

python -m pip install --upgrade pip setuptools wheel

# ---- Установка PyTorch с MPS -----------------------------------------------
info "Устанавливаю PyTorch (nightly с поддержкой MPS для Apple Silicon)..."
# Используем stable для arm64 — поддержка MPS уже в 2.1+
python -m pip install --upgrade torch torchvision torchaudio

# Проверка MPS
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built:     {torch.backends.mps.is_built()}")
PY
ok "PyTorch установлен"

# ---- Остальные зависимости -------------------------------------------------
info "Устанавливаю Python-зависимости из requirements_mac.txt ..."
python -m pip install -r "${PROJECT_DIR}/requirements_mac.txt"
ok "Зависимости установлены"

# ---- Патчи для Mac: отключаем CUDA-only модули -----------------------------
info "Применяю Mac-патчи к коду Hunyuan3D..."
# Защитная обёртка: если в проекте встречается from ... import cuda_* — ок, пропускаем при ошибке.
# Настоящие патчи обычно не нужны, так как shape-модель работает через стандартный torch.

# Создаём mac_compat.py для подмены CUDA-вызовов на MPS
cat > "${REPO_DIR}/mac_compat.py" <<'PY'
"""Совместимость с Apple Silicon.
Импортируй в начале своего скрипта:
    import mac_compat
"""
import os
import torch

# Включаем fallback для операций, не реализованных в MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = best_device()
print(f"[mac_compat] Using device: {DEVICE}")
PY
ok "mac_compat.py добавлен"

# ---- Загрузка весов --------------------------------------------------------
mkdir -p "$MODELS_DIR"
info "Загружаю веса модели из Hugging Face ($HF_MODEL_ID)..."
info "(Это может занять 20-40 минут в зависимости от скорости интернета — модель ~20 GB)"

# Новый CLI от Hugging Face называется `hf`. Старый `huggingface-cli` deprecated.
if ! command -v hf >/dev/null 2>&1; then
    python -m pip install --upgrade "huggingface_hub[cli]"
fi

# Выбираем доступную команду (hf — новая, huggingface-cli — старая fallback)
HF_CMD=""
if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
else
    err "Не найден ни 'hf', ни 'huggingface-cli'"
    exit 1
fi
info "Использую CLI: $HF_CMD"

# Скачиваем только shape-модель по умолчанию (paint требует CUDA и ~10 GB дополнительно)
"$HF_CMD" download "$HF_MODEL_ID" \
    --local-dir "${MODELS_DIR}/Hunyuan3D-2.1" \
    --include "hunyuan3d-dit-v2-1/*" \
    --include "hunyuan3d-vae-v2-1/*" \
    --include "README*" \
    || { err "Не удалось скачать веса. Проверь соединение и HF_TOKEN, если модель gated."; exit 1; }

ok "Shape-модель скачана в ${MODELS_DIR}/Hunyuan3D-2.1"

# Paint-модель — опционально
echo ""
read -rp "Скачать paint-модель для текстурирования? На Mac она работает через медленный CPU-fallback (+10 GB, не рекомендуется) [y/N]: " ANSWER
ANSWER_LC="$(echo "${ANSWER:-}" | tr '[:upper:]' '[:lower:]')"
if [[ "$ANSWER_LC" == "y" || "$ANSWER_LC" == "yes" ]]; then
    "$HF_CMD" download "$HF_MODEL_ID" \
        --local-dir "${MODELS_DIR}/Hunyuan3D-2.1" \
        --include "hunyuan3d-paintpbr-v2-1/*" \
        || warn "Не удалось скачать paint-модель"
fi

# ---- Финальная проверка ----------------------------------------------------
info "Проверяю установку..."
python - <<'PY'
import sys, importlib
missing = []
for mod in ("torch", "diffusers", "transformers", "gradio", "trimesh", "PIL", "numpy"):
    try:
        importlib.import_module(mod)
    except ImportError as e:
        missing.append(f"{mod}: {e}")
if missing:
    print("Отсутствуют модули:")
    for m in missing:
        print("  -", m)
    sys.exit(1)
print("Все основные модули на месте.")
PY
ok "Установка завершена успешно"

echo ""
echo "============================================================"
echo "  Hunyuan3D 2.1 установлен в $PROJECT_DIR"
echo "  Запусти интерфейс: $PROJECT_DIR/launch.sh"
echo "  После запуска открой: http://localhost:7860"
echo "============================================================"
