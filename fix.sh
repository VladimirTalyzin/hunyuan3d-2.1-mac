#!/usr/bin/env bash
# =============================================================================
# Дозагрузка весов и починка путей для уже установленного Hunyuan3D 2.1.
# Запускай, если install.sh упал на этапе скачивания моделей или
# если интерфейс говорит "No module named 'hy3dshape.pipelines'".
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
MODELS_DIR="${PROJECT_DIR}/weights/Hunyuan3D-2.1"
REPO_DIR="${PROJECT_DIR}/Hunyuan3D-2.1"
HF_MODEL_ID="tencent/Hunyuan3D-2.1"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info(){ echo -e "${BLUE}[INFO]${NC}  $*"; }
ok(){   echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn(){ echo -e "${YELLOW}[WARN]${NC}  $*"; }
err(){  echo -e "${RED}[ERR ]${NC}  $*" >&2; }

# ---- Проверки -------------------------------------------------------------
[[ -d "$VENV_DIR" ]] || { err "venv не найден: $VENV_DIR. Запусти install.sh сначала."; exit 1; }
[[ -d "$REPO_DIR" ]] || { err "Репозиторий не найден: $REPO_DIR. Запусти install.sh сначала."; exit 1; }

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
ok "venv активирован: $(python --version)"

# ---- Проверка путей Python ------------------------------------------------
info "Проверяю структуру пакетов Hunyuan3D..."
if [[ ! -f "${REPO_DIR}/hy3dshape/hy3dshape/pipelines.py" ]]; then
    err "pipelines.py не найден по ожидаемому пути: ${REPO_DIR}/hy3dshape/hy3dshape/pipelines.py"
    err "Похоже, репозиторий не склонировался полностью. Запусти install.sh заново."
    exit 1
fi
ok "hy3dshape/hy3dshape/pipelines.py на месте"

# ---- Устанавливаем Mac-совместимые repo-зависимости ------------------------
info "Доустанавливаю недостающие зависимости из репозитория (Mac-подмножество)..."
# Из Hunyuan3D-2.1/requirements.txt берём всё, ЗА ИСКЛЮЧЕНИЕМ:
#   - cupy-cuda12x  (CUDA only)
#   - deepspeed     (CUDA only)
#   - bpy==4.0      (Blender, тяжело и не нужно для инференса)
#   - basicsr==1.4.2 / realesrgan (ломаные на современных torchvision, нужны только для апскейлинга)
#   - tb_nightly    (не критично)
#   - жёстких пинов ставить не будем, чтобы не сломать уже установленный torch
python -m pip install --upgrade \
    "pymeshlab" \
    "pygltflib" \
    "xatlas" \
    "omegaconf" \
    "configargparse" \
    "einops" \
    "opencv-python" \
    "imageio" \
    "scikit-image" \
    "trimesh" \
    "open3d; platform_machine == 'arm64'" \
    "pyyaml" \
    "psutil" \
    "torchmetrics" \
    "timm" \
    "torchdiffeq" \
    "pythreejs" \
    "pytorch-lightning" \
    "safetensors" \
    "accelerate" \
    "tqdm" \
    "rembg" \
    "onnxruntime" \
    || warn "Часть пакетов не установилась — смотри вывод выше"
ok "Зависимости обновлены"

# Быстрая проверка, что import работает с правильным PYTHONPATH.
# Если упадёт на конкретном модуле — цикл автодоустановки (до 5 попыток).
info "Проверяю import 'hy3dshape.pipelines' с исправленным PYTHONPATH..."
attempt=0
max_attempts=5
while (( attempt < max_attempts )); do
    attempt=$((attempt + 1))
    result=$(PYTHONPATH="${REPO_DIR}/hy3dshape:${REPO_DIR}/hy3dpaint:${REPO_DIR}" \
        python - <<'PY' 2>&1
import importlib, sys
try:
    mod = importlib.import_module("hy3dshape.pipelines")
    cls = getattr(mod, "Hunyuan3DDiTFlowMatchingPipeline", None)
    if cls is None:
        print("WARN: класс Hunyuan3DDiTFlowMatchingPipeline не найден в модуле")
        sys.exit(2)
    print(f"OK: {mod.__file__}")
    print(f"OK: класс {cls.__name__} доступен")
    sys.exit(0)
except ModuleNotFoundError as e:
    # Выдаём имя модуля для авто-установки
    print(f"MISSING_MODULE:{e.name}")
    sys.exit(10)
except Exception as e:
    print(f"IMPORT ERROR: {type(e).__name__}: {e}")
    sys.exit(3)
PY
)
    exit_code=$?
    echo "$result"
    if (( exit_code == 0 )); then
        ok "Импорты работают (попытка $attempt)"
        break
    elif (( exit_code == 10 )); then
        # Парсим имя модуля и пытаемся установить
        missing=$(echo "$result" | sed -n 's/^MISSING_MODULE:\(.*\)$/\1/p' | head -1)
        if [[ -n "$missing" ]]; then
            # Маппинг: имя модуля (import X) → имя PyPI-пакета
            case "$missing" in
                cv2)            pkg="opencv-python" ;;
                PIL)            pkg="Pillow" ;;
                sklearn)        pkg="scikit-learn" ;;
                skimage)        pkg="scikit-image" ;;
                yaml)           pkg="pyyaml" ;;
                *)              pkg="$missing" ;;
            esac
            warn "Не хватает модуля '$missing' — ставлю пакет '$pkg' (попытка $attempt/$max_attempts)"
            python -m pip install --upgrade "$pkg" || {
                err "Не удалось установить $pkg. Попробуй вручную:"
                err "  source venv/bin/activate && pip install $pkg"
                exit 1
            }
        else
            err "Не могу распарсить имя недостающего модуля"
            exit 1
        fi
    else
        err "Импорт падает по другой причине (exit=$exit_code). Смотри вывод выше."
        exit 1
    fi
done

if (( attempt >= max_attempts )); then
    err "Достигнут предел попыток авто-установки. Покажи мне последний вывод — разберём."
    exit 1
fi

# ---- Определяем CLI huggingface -------------------------------------------
HF_CMD=""
if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
    ok "Использую новый CLI: hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
    warn "Использую устаревший huggingface-cli"
else
    info "Устанавливаю huggingface_hub[cli]..."
    python -m pip install --upgrade "huggingface_hub[cli]"
    HF_CMD="hf"
fi

# ---- Проверка логина (если модель gated) ----------------------------------
info "Проверяю доступ к модели на Hugging Face..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    info "HF_TOKEN задан в окружении — будет использован"
else
    # Проверяем, есть ли сохранённый токен
    if python -c "from huggingface_hub import HfFolder; t=HfFolder.get_token(); exit(0 if t else 1)" 2>/dev/null; then
        ok "Сохранённый HF-токен обнаружен"
    else
        warn "HF-токен не найден. Если модель gated — надо залогиниться."
        echo "  Выполни: $HF_CMD auth login  (или)  hf auth login"
        echo "  Токен получить здесь: https://huggingface.co/settings/tokens"
        read -rp "Залогиниться сейчас? [y/N]: " ANSWER
        ANSWER_LC="$(echo "${ANSWER:-}" | tr '[:upper:]' '[:lower:]')"
        if [[ "$ANSWER_LC" == "y" || "$ANSWER_LC" == "yes" ]]; then
            "$HF_CMD" auth login
        fi
    fi
fi

# ---- Скачивание весов -----------------------------------------------------
mkdir -p "$MODELS_DIR"
info "Скачиваю shape-модель (DiT + VAE) ~15 GB из ${HF_MODEL_ID} ..."
info "(Можно прервать и перезапустить — скачивание резюмируется)"

"$HF_CMD" download "$HF_MODEL_ID" \
    --local-dir "$MODELS_DIR" \
    --include "hunyuan3d-dit-v2-1/*" \
    --include "hunyuan3d-vae-v2-1/*" \
    --include "README*" \
    || { err "Скачивание не удалось. Если это 401/403 — нужна авторизация. Детали: $HF_CMD auth whoami"; exit 1; }

ok "Shape-модель скачана в $MODELS_DIR"

# Paint-модель — опционально
echo ""
read -rp "Скачать paint-модель (+10 GB, на Mac почти бесполезна)? [y/N]: " ANSWER
ANSWER_LC="$(echo "${ANSWER:-}" | tr '[:upper:]' '[:lower:]')"
if [[ "$ANSWER_LC" == "y" || "$ANSWER_LC" == "yes" ]]; then
    "$HF_CMD" download "$HF_MODEL_ID" \
        --local-dir "$MODELS_DIR" \
        --include "hunyuan3d-paintpbr-v2-1/*" \
        || warn "Paint-модель не скачалась (на Mac это не критично)"
fi

# ---- Проверка, что все нужные файлы есть ----------------------------------
info "Проверяю локальные файлы..."
for sub in "hunyuan3d-dit-v2-1" "hunyuan3d-vae-v2-1"; do
    if [[ -d "${MODELS_DIR}/${sub}" ]]; then
        ok "${sub}: $(find "${MODELS_DIR}/${sub}" -type f | wc -l | tr -d ' ') файл(ов)"
    else
        warn "${sub}: отсутствует"
    fi
done

echo ""
echo "============================================================"
echo "  Починка завершена. Запусти: ./launch.sh"
echo "  Интерфейс: http://localhost:7860"
echo "============================================================"
