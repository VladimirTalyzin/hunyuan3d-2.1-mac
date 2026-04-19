#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hunyuan3D 2.1 — web interface for Apple Silicon.

Run:
    ./launch.sh   (or)   python gradio_app.py

This UI wraps the Hunyuan3D 2.1 shape pipeline (DiT + VAE), tuned for MPS
(Apple Silicon). Texturing (paint) on Mac is slow and disabled by default.

Console logs are always in English. UI is localised to English (default),
Chinese and Russian via the language radio at the top of the page.
"""
from __future__ import annotations

import os
import sys
import gc
import time
from pathlib import Path
from typing import Optional

# ==== Apple Silicon compatibility ===========================================
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import contextlib
import traceback
import numpy as np
import torch
import trimesh
import gradio as gr
from PIL import Image

# Lowpoly optimization algorithms (quadric / iso / planar / hybrid / auto)
# live in a standalone module that can be used from the debug CLI too.
import lowpoly  # noqa: E402

# ---- Runtime patches: redirect CUDA-only calls to MPS/CPU ------------------
# Hunyuan3D source hard-codes `device='cuda'` and `torch.autocast(device_type='cuda')`
# which crashes with "Torch not compiled with CUDA enabled" on Apple Silicon.
# We patch torch.autocast so `cuda` becomes no-op on CUDA-less systems.
def _install_mps_patches() -> None:
    if torch.cuda.is_available():
        return  # real CUDA present — leave everything alone

    # 1) torch.autocast(device_type="cuda", ...) → safe no-op context
    _orig_autocast = torch.autocast

    class _SafeAutocast:
        def __init__(self, device_type, *args, **kwargs):
            self.device_type = device_type
            self.args = args
            self.kwargs = kwargs
            self._ctx = None

        def __enter__(self):
            if self.device_type == "cuda":
                # MPS autocast is flaky; safest option is no-op.
                self._ctx = contextlib.nullcontext()
                return self._ctx.__enter__()
            self._ctx = _orig_autocast(self.device_type, *self.args, **self.kwargs)
            return self._ctx.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)

    torch.autocast = _SafeAutocast
    if hasattr(torch, "amp"):
        torch.amp.autocast = _SafeAutocast

    # 2) torch.cuda.synchronize / Event / empty_cache — stubs
    class _DummyEvent:
        def __init__(self, *a, **k): pass
        def record(self, *a, **k): pass
        def elapsed_time(self, other): return 0.0
        def synchronize(self): pass

    if not hasattr(torch.cuda, "_patched"):
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.Event = _DummyEvent
        torch.cuda.empty_cache = lambda: (
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        )
        torch.cuda._patched = True

    print("[mps_patches] torch.autocast and torch.cuda.* redirected to MPS/no-op")

_install_mps_patches()

# ---- Device usage tracker (MPS vs CPU) ------------------------------------
# Wraps torch operations through __torch_function__, counts which device
# each resulting tensor ends up on. Answers the question "how much actually
# ran on MPS vs how much fell back to CPU".
try:
    from torch.overrides import TorchFunctionMode as _TFMode
    _HAS_TFMODE = True
except Exception:
    _HAS_TFMODE = False

if _HAS_TFMODE:
    class DeviceUsageTracker(_TFMode):
        """Counts torch operations by output tensor device."""
        def __init__(self) -> None:
            super().__init__()
            self.counts: dict[str, int] = {}
            # per-op per-device counts — used to find the ops that leak to CPU
            self.op_per_dev: dict[str, dict[str, int]] = {}

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            result = func(*args, **kwargs)
            devices: set[str] = set()
            def _walk(x):
                if isinstance(x, torch.Tensor):
                    try:
                        devices.add(str(x.device).split(":")[0])
                    except Exception:
                        pass
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        _walk(y)
            _walk(result)
            op_name = getattr(func, "__name__", None) or str(func)
            for d in devices:
                self.counts[d] = self.counts.get(d, 0) + 1
                bucket = self.op_per_dev.setdefault(op_name, {})
                bucket[d] = bucket.get(d, 0) + 1
            return result

        def summary_text(self, lang: str = "en") -> str:
            total = sum(self.counts.values())
            if total == 0:
                return t(lang, "mps_none")
            lines = [t(lang, "mps_total", total=f"{total:,}".replace(",", " "))]
            for dev in sorted(self.counts, key=lambda d: -self.counts[d]):
                cnt = self.counts[dev]
                pct = 100.0 * cnt / total
                lines.append(t(
                    lang, "mps_device_line",
                    dev=dev,
                    cnt=f"{cnt:,}".replace(",", " "),
                    pct=pct,
                ))
            # Top-10 ops that bleed to CPU
            cpu_ops = []
            for op, dc in self.op_per_dev.items():
                cpu_cnt = dc.get("cpu", 0)
                if cpu_cnt > 0:
                    total_op = sum(dc.values())
                    cpu_ops.append((op, cpu_cnt, total_op))
            cpu_ops.sort(key=lambda x: -x[1])
            if cpu_ops:
                lines.append("")
                lines.append(t(lang, "mps_cpu_header"))
                for op, cpu_cnt, total_op in cpu_ops[:10]:
                    pct_op = 100.0 * cpu_cnt / total_op if total_op else 0.0
                    lines.append(t(
                        lang, "mps_cpu_line",
                        op=op,
                        cnt=f"{cpu_cnt:,}".replace(",", " "),
                        pct=pct_op,
                    ))
            return "\n".join(lines)
else:
    class DeviceUsageTracker:  # type: ignore
        """Fallback when torch.overrides.TorchFunctionMode is unavailable."""
        def __init__(self) -> None:
            self.counts = {}
            self.op_per_dev = {}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def summary_text(self, lang: str = "en") -> str:
            return t(lang, "mps_disabled")

# ---- Device selection ------------------------------------------------------
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR    = PROJECT_DIR / "Hunyuan3D-2.1"
WEIGHTS_DIR = PROJECT_DIR / "weights" / "Hunyuan3D-2.1"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add Hunyuan3D code to PYTHONPATH.
# Real layout: Hunyuan3D-2.1/hy3dshape/hy3dshape/pipelines.py (double-nested).
# We add both the root (for top-level utilities) and the subpackage dirs.
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR))
    for sub in ("hy3dshape", "hy3dpaint"):
        sub_path = REPO_DIR / sub
        if sub_path.exists():
            sys.path.insert(0, str(sub_path))

# ==== Internationalization ==================================================
# English is the default and the source-of-truth keyset. Zh/Ru are overlays;
# if a key is missing in zh/ru we silently fall back to English.
# All console prints (stdout/stderr) stay in English regardless of UI language.
LANG_CODES = ("en", "zh", "ru")
LANG_DISPLAY = {"en": "English", "zh": "中文", "ru": "Русский"}
LANG_BY_DISPLAY = {v: k for k, v in LANG_DISPLAY.items()}

I18N: dict[str, dict[str, str]] = {
    "en": {
        # ----- static UI strings -----
        "lang_label": "Interface language",
        "description": (
            "# Hunyuan3D 2.1 — local interface (Apple Silicon)\n\n"
            "Generate a 3D mesh from a single image using Tencent Hunyuan3D 2.1.\n\n"
            "**Device:** `{device}` • **dtype:** `{dtype}` • **Project:** `{project}`\n\n"
            "> On Mac only shape generation works. Texturing needs CUDA and "
            "is disabled by default. On an M4 Pro at octree=256 one mesh takes "
            "roughly **10-20 minutes** — MPS is used, but some operations fall "
            "back to CPU automatically (tick «Measure MPS vs CPU op share» to "
            "see the exact breakdown)."
        ),
        "input_image":    "Input image (object on a plain background)",
        "remove_bg":      "Automatically remove background (rembg)",
        "track_mps":      "Measure MPS vs CPU op share",
        "gen_params":     "Generation parameters",
        "num_steps":      "Diffusion steps (more = higher quality, slower)",
        "guidance":       "Guidance scale (classifier strength)",
        "octree":         "Octree resolution (mesh detail)",
        "seed":           "Seed (random seed)",
        "generate_btn":   "🚀 Generate 3D object",
        "log_label":      "Execution log",
        "preview_label":  "Result (preview)",
        "mps_stats_initial": "_MPS tracker is off. Tick the checkbox on the left to measure the MPS vs CPU share._",
        "mps_stats_header":  "### MPS vs CPU (operations during generation)\n",
        "mps_stats_off_note": "_Tracking was requested but `torch.overrides.TorchFunctionMode` is unavailable in this PyTorch version._",
        "postproc_header":  "### Post-processing (mesh decimation)",
        "opt_level_label":  "Optimization level (always applied to the original mesh)",
        "opt_mode_label":   "Algorithm",
        "opt_mode_help": (
            "- **Quadric** — universal, uniform reduction; best overall quality/size ratio.\n"
            "- **Iso** — uniform-edge remeshing with feature-edge preservation (CAD-like).\n"
            "- **Planar** — consolidates large flat regions into few big triangles "
            "(door panels, walls, boxes). Keeps curved regions as-is.\n"
            "- **Hybrid** — planar first, then quadric. Fewer triangles, keeps flat look.\n"
            "- **Auto** — planar for hard-surface meshes, quadric for organic ones.\n"
        ),
        "opt_btn":          "🛠️ Apply optimization",
        "opt_log":          "Post-processing log",
        "download_header":  "### Download mesh",
        "dl_format":        "Output format",
        "dl_btn":           "💾 Save and download",
        "dl_file":          "File to download",
        "tips": (
            "### Tips\n"
            "- Quick test: `octree=128`, `steps=30`\n"
            "- Maximum quality: `octree=320+`, `steps=80+` (slow!)\n"
            "- If MPS runs out of memory — close other apps or reduce octree\n"
            "- For CAD use: start with «Medium» optimization, check preview, tighten if needed\n"
            "- Output files are saved to `outputs/` next to this script\n"
        ),
        # ----- optimization level labels -----
        "opt_none":   "No optimization",
        "opt_weak":   "Light (≈ 30% of original)",
        "opt_medium": "Medium (≈ 8% of original)",
        "opt_strong": "Heavy (≈ 2% of original)",
        # ----- optimization mode labels -----
        "opt_mode_quadric": "Quadric (universal)",
        "opt_mode_iso":     "Iso (uniform triangles)",
        "opt_mode_planar":  "Planar (big flat triangles)",
        "opt_mode_hybrid":  "Hybrid (planar + quadric)",
        "opt_mode_auto":    "Auto (detect & pick)",
        # ----- dynamic log messages -----
        "log_device":          "Device: {device}, dtype: {dtype}",
        "log_preprocess":      "Preprocessing image (remove_bg={remove_bg})...",
        "log_after_size":      "Size after preprocessing: {size}",
        "log_loading_pipeline":"Loading pipeline (first run may take a few minutes)...",
        "log_pipeline_comps":  "Pipeline components:\n{diag}",
        "log_generation":      "Mesh generation: steps={steps}, guidance={guidance}, octree={octree}, seed={seed}, track_mps={track_mps}",
        "log_mesh_ready":      "Mesh ready: {verts} vertices, {faces} faces",
        "log_done":            "✅ Done in {elapsed:.1f} s",
        "log_preview_glb":     "Preview (GLB): {path}",
        "log_next_step":       "Pick a format on the right to download. Post-processing is also on the right.",
        "log_warn_no_image":   "⚠️ Please upload an input image",
        "log_err_model":       "❌ Model load error: {e}",
        "log_err_oom":         "❌ Out of memory or MPS failure: {msg}\n\nHint: reduce octree_resolution to 128 or num_steps to 30.\n\nTraceback:\n{tb}",
        "log_err_inference":   "❌ Inference error: {msg}\n\nTraceback:\n{tb}",
        "log_err_unknown":     "❌ Unknown error: {name}: {e}\n\nTraceback:\n{tb}",
        "log_no_mesh_yet":     "⚠️ Generate a mesh first, then optimize.",
        "log_shown_original":  "Showing original without optimization: {verts} vertices, {faces} faces.",
        "log_already_small":   "Mesh is already smaller than target ({faces} ≤ {target}). Optimization not needed.",
        "log_decimating":      "Decimating original ({faces} faces) → ~{target} faces (level: «{level}»)...",
        "log_optimizing":      "Optimizing: algorithm «{mode}», level «{level}», input {faces} faces...",
        "log_auto_pick":       "Auto classifier: large-planar area = {ratio:.0f}% → chose «{chosen}».",
        "log_decimation_done": "✅ Done in {elapsed:.1f} s: {v0}→{v1} vertices, {f0}→{f1} faces (reduced by {pct:.1f}%)",
        "log_preview":         "Preview: {path}",
        # ----- MPS stats block -----
        "mps_total":       "**Total tensor operations:** {total}",
        "mps_device_line": "- `{dev}`: {cnt} ({pct:.1f}%)",
        "mps_cpu_header":  "**Top CPU operations** (these are what hurts performance):",
        "mps_cpu_line":    "- `{op}`: {cnt} on CPU ({pct:.0f}% of this operation)",
        "mps_none":        "MPS tracker recorded no operations (probably disabled).",
        "mps_disabled":    "_torch.overrides.TorchFunctionMode is unavailable in this PyTorch version — tracker is off._",
    },
    "zh": {
        "lang_label": "界面语言",
        "description": (
            "# 混元3D 2.1 — 本地界面（Apple Silicon）\n\n"
            "使用腾讯 Hunyuan3D 2.1 从单张图像生成3D网格。\n\n"
            "**设备：** `{device}` • **精度：** `{dtype}` • **项目：** `{project}`\n\n"
            "> Mac 上仅支持形状生成。纹理生成需要 CUDA，默认禁用。"
            "M4 Pro 在 octree=256 时每个网格约需 **10-20 分钟** — MPS 已启用，"
            "但部分算子自动回退到 CPU（勾选「统计 MPS 与 CPU 操作占比」"
            "可查看具体分布）。"
        ),
        "input_image":    "输入图像（物体置于纯色背景）",
        "remove_bg":      "自动移除背景（rembg）",
        "track_mps":      "统计 MPS 与 CPU 操作占比",
        "gen_params":     "生成参数",
        "num_steps":      "扩散步数（越多质量越高，也越慢)",
        "guidance":       "引导强度（classifier-free guidance)",
        "octree":         "八叉树分辨率（网格细节)",
        "seed":           "随机种子",
        "generate_btn":   "🚀 生成 3D 对象",
        "log_label":      "运行日志",
        "preview_label":  "结果（预览）",
        "mps_stats_initial": "_MPS 跟踪已关闭。勾选左侧的复选框以测量 MPS 与 CPU 占比。_",
        "mps_stats_header":  "### MPS 与 CPU（生成期间的算子）\n",
        "mps_stats_off_note": "_已请求跟踪，但当前 PyTorch 版本不支持 `torch.overrides.TorchFunctionMode`。_",
        "postproc_header":  "### 后处理（网格简化）",
        "opt_level_label":  "优化级别（始终基于原始网格应用）",
        "opt_btn":          "🛠️ 应用优化",
        "opt_log":          "后处理日志",
        "download_header":  "### 下载网格",
        "dl_format":        "输出格式",
        "dl_btn":           "💾 保存并下载",
        "dl_file":          "待下载文件",
        "tips": (
            "### 提示\n"
            "- 快速测试：`octree=128`，`steps=30`\n"
            "- 最高质量：`octree=320+`，`steps=80+`（较慢！）\n"
            "- 如果 MPS 内存不足 — 关闭其他应用或降低 octree\n"
            "- CAD 用途：从「中度」优化开始，查看预览后再决定是否加强\n"
            "- 输出文件保存在脚本旁的 `outputs/` 目录\n"
        ),
        "opt_none":   "不优化",
        "opt_weak":   "轻度（约原始的 30%）",
        "opt_medium": "中度（约原始的 8%）",
        "opt_strong": "高度（约原始的 2%）",
        "opt_mode_label":   "算法",
        "opt_mode_help": (
            "- **Quadric** — 通用，均匀简化；综合质量/体积最佳。\n"
            "- **Iso** — 各向同性重新网格化，保留特征边（CAD 风格）。\n"
            "- **Planar** — 把大型平面区域合并成少量大三角形"
            "（门板、墙面、盒子）。曲面区域保持原样。\n"
            "- **Hybrid** — 先 Planar，再 Quadric。三角形更少，保留平面外观。\n"
            "- **Auto** — 硬表面网格用 Planar，有机造型用 Quadric。\n"
        ),
        "opt_mode_quadric": "Quadric（通用）",
        "opt_mode_iso":     "Iso（均匀三角形）",
        "opt_mode_planar":  "Planar（大平面三角形）",
        "opt_mode_hybrid":  "Hybrid（Planar + Quadric）",
        "opt_mode_auto":    "Auto（自动检测）",
        "log_device":          "设备：{device}，精度：{dtype}",
        "log_preprocess":      "预处理图像（remove_bg={remove_bg}）……",
        "log_after_size":      "预处理后尺寸：{size}",
        "log_loading_pipeline":"加载流水线（首次运行可能需要几分钟）……",
        "log_pipeline_comps":  "流水线组件：\n{diag}",
        "log_generation":      "网格生成：steps={steps}，guidance={guidance}，octree={octree}，seed={seed}，track_mps={track_mps}",
        "log_mesh_ready":      "网格已生成：{verts} 个顶点，{faces} 个面",
        "log_done":            "✅ 完成，用时 {elapsed:.1f} 秒",
        "log_preview_glb":     "预览（GLB）：{path}",
        "log_next_step":       "在右侧选择格式下载。后处理也在右侧。",
        "log_warn_no_image":   "⚠️ 请先上传输入图像",
        "log_err_model":       "❌ 模型加载错误：{e}",
        "log_err_oom":         "❌ 内存不足或 MPS 失败：{msg}\n\n提示：将 octree_resolution 降至 128 或 num_steps 降至 30。\n\nTraceback：\n{tb}",
        "log_err_inference":   "❌ 推理错误：{msg}\n\nTraceback：\n{tb}",
        "log_err_unknown":     "❌ 未知错误：{name}: {e}\n\nTraceback：\n{tb}",
        "log_no_mesh_yet":     "⚠️ 请先生成网格，然后再优化。",
        "log_shown_original":  "显示未优化的原始网格：{verts} 个顶点，{faces} 个面。",
        "log_already_small":   "网格已小于目标面数（{faces} ≤ {target}）。无需优化。",
        "log_decimating":      "对原始网格进行简化（{faces} 个面）→ 约 {target} 个面（级别：「{level}」）……",
        "log_optimizing":      "优化中：算法「{mode}」，级别「{level}」，输入 {faces} 个面……",
        "log_auto_pick":       "Auto 分类器：大平面面积占比 {ratio:.0f}% → 选择「{chosen}」。",
        "log_decimation_done": "✅ 完成，用时 {elapsed:.1f} 秒：{v0}→{v1} 顶点，{f0}→{f1} 面（减少 {pct:.1f}%)",
        "log_preview":         "预览：{path}",
        "mps_total":       "**张量操作总数：** {total}",
        "mps_device_line": "- `{dev}`: {cnt} ({pct:.1f}%)",
        "mps_cpu_header":  "**CPU 算子 Top 列表**（正是它们拖慢速度）：",
        "mps_cpu_line":    "- `{op}`: {cnt} 次在 CPU 上执行（占该算子的 {pct:.0f}%)",
        "mps_none":        "MPS 跟踪器未记录任何操作（可能已禁用）。",
        "mps_disabled":    "_当前 PyTorch 版本不支持 torch.overrides.TorchFunctionMode — 跟踪器已禁用。_",
    },
    "ru": {
        "lang_label": "Язык интерфейса",
        "description": (
            "# Hunyuan3D 2.1 — локальный интерфейс (Apple Silicon)\n\n"
            "Генерация 3D-меша из одного изображения с помощью Tencent Hunyuan3D 2.1.\n\n"
            "**Устройство:** `{device}` • **dtype:** `{dtype}` • **Проект:** `{project}`\n\n"
            "> На Mac работает только генерация формы (shape). Текстурирование "
            "требует CUDA и по умолчанию отключено. На M4 Pro при octree=256 "
            "один меш занимает ориентировочно **10-20 минут** — MPS "
            "используется, но часть операций автоматически падает на CPU "
            "(включи «Считать долю операций MPS vs CPU», чтобы увидеть точную "
            "раскладку)."
        ),
        "input_image":    "Входное изображение (объект на простом фоне)",
        "remove_bg":      "Автоматически удалять фон (rembg)",
        "track_mps":      "Считать долю операций MPS vs CPU",
        "gen_params":     "Параметры генерации",
        "num_steps":      "Шагов диффузии (больше = качественнее, дольше)",
        "guidance":       "Guidance scale (сила классификатора)",
        "octree":         "Octree resolution (детализация меша)",
        "seed":           "Seed (случайное зерно)",
        "generate_btn":   "🚀 Сгенерировать 3D-объект",
        "log_label":      "Лог выполнения",
        "preview_label":  "Результат (превью)",
        "mps_stats_initial": "_MPS-трекер выключен. Включи слева чек-бокс, чтобы увидеть долю MPS vs CPU._",
        "mps_stats_header":  "### MPS vs CPU (операции во время генерации)\n",
        "mps_stats_off_note": "_Трекинг запрошен, но `torch.overrides.TorchFunctionMode` недоступен в этой версии PyTorch._",
        "postproc_header":  "### Пост-обработка (декимация меша)",
        "opt_level_label":  "Уровень оптимизации (всегда применяется к исходному мешу)",
        "opt_mode_label":   "Алгоритм",
        "opt_mode_help": (
            "- **Quadric** — универсальный, равномерная децимация; "
            "лучшее соотношение качества и размера.\n"
            "- **Iso** — равномерные треугольники с сохранением характерных "
            "рёбер (CAD-вид).\n"
            "- **Planar** — сливает большие плоские регионы в несколько "
            "крупных треугольников (дверные панели, стены, коробки). "
            "Изогнутые участки остаются как есть.\n"
            "- **Hybrid** — сначала planar, затем quadric. Меньше треугольников, "
            "сохраняется «плоский» вид.\n"
            "- **Auto** — planar для hard-surface, quadric для органики.\n"
        ),
        "opt_btn":          "🛠️ Применить оптимизацию",
        "opt_log":          "Лог пост-обработки",
        "download_header":  "### Скачать меш",
        "dl_format":        "Формат выгрузки",
        "dl_btn":           "💾 Сохранить и скачать",
        "dl_file":          "Файл для скачивания",
        "tips": (
            "### Советы\n"
            "- Быстрый тест: `octree=128`, `steps=30`\n"
            "- Максимум качества: `octree=320+`, `steps=80+` (долго!)\n"
            "- Если MPS падает по памяти — закрой другие приложения или уменьши octree\n"
            "- Для CAD: начни со «Средней» оптимизации, глянь превью, при необходимости усиль\n"
            "- Выходные файлы сохраняются в `outputs/` рядом со скриптом\n"
        ),
        "opt_none":   "Без оптимизации",
        "opt_weak":   "Слабая (≈ 30% от исходного)",
        "opt_medium": "Средняя (≈ 8% от исходного)",
        "opt_strong": "Сильная (≈ 2% от исходного)",
        "opt_mode_quadric": "Quadric (универсальный)",
        "opt_mode_iso":     "Iso (равномерные треугольники)",
        "opt_mode_planar":  "Planar (крупные плоские треугольники)",
        "opt_mode_hybrid":  "Hybrid (planar + quadric)",
        "opt_mode_auto":    "Auto (автоподбор)",
        "log_device":          "Устройство: {device}, dtype: {dtype}",
        "log_preprocess":      "Предобработка изображения (remove_bg={remove_bg})...",
        "log_after_size":      "Размер после предобработки: {size}",
        "log_loading_pipeline":"Загружаю пайплайн (первый запуск может занять минуты)...",
        "log_pipeline_comps":  "Компоненты пайплайна:\n{diag}",
        "log_generation":      "Генерация меша: steps={steps}, guidance={guidance}, octree={octree}, seed={seed}, track_mps={track_mps}",
        "log_mesh_ready":      "Меш готов: {verts} вершин, {faces} граней",
        "log_done":            "✅ Готово за {elapsed:.1f} с",
        "log_preview_glb":     "Превью (GLB): {path}",
        "log_next_step":       "Выбери формат справа для скачивания. Пост-обработка — там же.",
        "log_warn_no_image":   "⚠️ Загрузите входное изображение",
        "log_err_model":       "❌ Ошибка загрузки модели: {e}",
        "log_err_oom":         "❌ Нехватка памяти или сбой MPS: {msg}\n\nСовет: уменьши octree_resolution до 128 или num_steps до 30.\n\nTraceback:\n{tb}",
        "log_err_inference":   "❌ Ошибка inference: {msg}\n\nTraceback:\n{tb}",
        "log_err_unknown":     "❌ Неизвестная ошибка: {name}: {e}\n\nTraceback:\n{tb}",
        "log_no_mesh_yet":     "⚠️ Сначала сгенерируй меш, потом оптимизируй.",
        "log_shown_original":  "Показан исходник без оптимизации: {verts} вершин, {faces} граней.",
        "log_already_small":   "Меш уже меньше целевого ({faces} ≤ {target}). Оптимизация не нужна.",
        "log_decimating":      "Декимация исходника ({faces} граней) → ~{target} граней (уровень: «{level}»)...",
        "log_optimizing":      "Оптимизация: алгоритм «{mode}», уровень «{level}», на входе {faces} граней...",
        "log_auto_pick":       "Auto-классификатор: площадь крупных плоских регионов {ratio:.0f}% → выбран «{chosen}».",
        "log_decimation_done": "✅ Готово за {elapsed:.1f} с: {v0}→{v1} вершин, {f0}→{f1} граней (уменьшено на {pct:.1f}%)",
        "log_preview":         "Превью: {path}",
        "mps_total":       "**Всего операций с тензорами:** {total}",
        "mps_device_line": "- `{dev}`: {cnt} ({pct:.1f}%)",
        "mps_cpu_header":  "**Топ CPU-операций** (именно эти ломают скорость):",
        "mps_cpu_line":    "- `{op}`: {cnt} на CPU ({pct:.0f}% от этой операции)",
        "mps_none":        "MPS-трекер не зафиксировал ни одной операции (возможно, отключён).",
        "mps_disabled":    "_torch.overrides.TorchFunctionMode недоступен в этой версии PyTorch — трекер отключён._",
    },
}


def t(lang: str, key: str, **kw) -> str:
    """Translate a key into the target language, falling back to English."""
    text = I18N.get(lang, {}).get(key)
    if text is None:
        text = I18N["en"].get(key, key)
    try:
        return text.format(**kw) if kw else text
    except Exception:
        return text


# ==== Lazy pipeline loading =================================================
_pipeline = None   # shape generation
_rembg_session = None

def _load_pipeline():
    """Lazy initialization of the Hunyuan3D shape pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print(f"[init] Loading Hunyuan3D shape pipeline on device: {DEVICE}, dtype={DTYPE}")

    # Hunyuan3D 2.1 ships Hunyuan3DDiTFlowMatchingPipeline in hy3dshape/pipelines.py.
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except Exception as e:
        raise RuntimeError(
            "Cannot import hy3dshape from the Hunyuan3D-2.1 repo. "
            f"Make sure it is cloned into {REPO_DIR}. Error: {e}"
        )

    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() else "tencent/Hunyuan3D-2.1"
    # Notes on the flags below:
    #   - weights ship as .ckpt (not .safetensors), file name is model.fp16.ckpt
    #     → use_safetensors=False  keeps the original extension
    #   - variant='fp16'           picks the fp16 checkpoint
    #   - device=DEVICE must be passed explicitly: Tencent default is
    #     device='cuda' which crashes with "Torch not compiled with CUDA enabled"
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
        variant="fp16",
        device=DEVICE,
        dtype=DTYPE,
    )
    # Safety net — move the pipeline again if something stayed on CPU.
    # IMPORTANT: Tencent's pipeline.to() does NOT return self (unlike standard
    # PyTorch). Never do `pipeline = pipeline.to(...)` — you'd get None.
    # We call .to() only for the side effect.
    if hasattr(pipeline, "to"):
        try:
            pipeline.to(DEVICE)
        except Exception as e:
            print(f"[init] pipeline.to({DEVICE}) failed: {e}. "
                  f"Moving components manually.", flush=True)
            for attr in ("model", "vae", "conditioner"):
                obj = getattr(pipeline, attr, None)
                if obj is not None and hasattr(obj, "to"):
                    obj.to(DEVICE)
    else:
        for attr in ("model", "vae", "conditioner"):
            obj = getattr(pipeline, attr, None)
            if obj is not None and hasattr(obj, "to"):
                obj.to(DEVICE)
    _pipeline = pipeline
    print("[init] Pipeline loaded")
    return _pipeline

def _load_rembg():
    """Lazy rembg session for background removal."""
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    try:
        from rembg import new_session
        _rembg_session = new_session("u2net")
        return _rembg_session
    except Exception as e:
        print(f"[warn] rembg unavailable: {e}")
        return None

# ==== Helpers ===============================================================
def preprocess_image(img: Image.Image, remove_bg: bool) -> Image.Image:
    """Return an RGBA 512x512 canvas with the object centered and bg removed."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if remove_bg:
        sess = _load_rembg()
        if sess is not None:
            from rembg import remove
            img = remove(img, session=sess)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2), img)
    return canvas.resize((512, 512), Image.LANCZOS)

def mesh_to_file(mesh: trimesh.Trimesh, fmt: str, suffix: str = "") -> str:
    """Save a trimesh to the requested format. Returns an absolute path."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    ext = fmt.lower()
    allowed = ("obj", "glb", "stl", "ply", "dae", "off", "3mf")
    if ext not in allowed:
        ext = "glb"
    name = f"hunyuan3d_{ts}{('_' + suffix) if suffix else ''}.{ext}"
    path = OUTPUT_DIR / name
    mesh.export(str(path))
    return str(path)

# ==== Post-processing: decimation ===========================================
# Optimization levels are language-agnostic: internally we use "none"/"weak"/
# "medium"/"strong". Display labels come from I18N.
OPT_PARAMS: dict[str, Optional[dict]] = {
    "none":   None,
    "weak":   dict(ratio=0.30, min_faces=3000),
    "medium": dict(ratio=0.08, min_faces=1500),
    "strong": dict(ratio=0.02, min_faces=400),
}
OPT_LEVEL_ORDER = ("none", "weak", "medium", "strong")


def _opt_level_choices(lang: str) -> list[tuple[str, str]]:
    """Return [(display_label, internal_key), ...] for a gr.Radio."""
    return [(t(lang, f"opt_{k}"), k) for k in OPT_LEVEL_ORDER]


# Order matters: the first entry is the default displayed at startup, and
# `quadric` matches the pre-existing behavior so opening the UI in a new
# session is indistinguishable from before.
OPT_MODE_ORDER = ("quadric", "iso", "planar", "hybrid", "auto")


def _opt_mode_choices(lang: str) -> list[tuple[str, str]]:
    return [(t(lang, f"opt_mode_{k}"), k) for k in OPT_MODE_ORDER]


def _preclean_mesh_pymeshlab(ms) -> None:
    """Clean-up series before decimation: duplicates, non-manifold, tiny components.

    A clean mesh decimates a lot better — the dangling triangles and T-vertices
    left by marching cubes confuse the quadric solver badly.
    """
    for step in (
        "meshing_remove_duplicate_vertices",
        "meshing_remove_duplicate_faces",
        "meshing_remove_unreferenced_vertices",
        "meshing_remove_null_faces",
        "meshing_remove_t_vertices",
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices",
    ):
        if hasattr(ms, step):
            try:
                getattr(ms, step)()
            except Exception:
                pass


def _taubin_smooth(ms, steps: int = 2) -> None:
    """Gentle Taubin smoothing WITHOUT shape shrinkage (unlike plain Laplacian).

    Good after decimation: removes stair-step noise on flat regions without
    eating sharp edges, provided you only run 1-2 iterations.
    """
    if hasattr(ms, "apply_coord_taubin_smoothing"):
        try:
            ms.apply_coord_taubin_smoothing(stepsmoothnum=int(steps))
            return
        except Exception:
            pass
    try:
        ms.apply_filter("apply_coord_taubin_smoothing", stepsmoothnum=int(steps))
    except Exception:
        pass


def decimate_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int,
    preserve_boundary: bool = True,
    feature_smooth: bool = True,
) -> trimesh.Trimesh:
    """Improved decimation: pre-clean → planar quadric → light Taubin smoothing.

    Uses pymeshlab with CAD-friendly parameters. Falls back to trimesh if
    pymeshlab is not available.
    """
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        pm = pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
        ms.add_mesh(pm, "orig")

        # 1) Pre-clean — critical for marching-cubes meshes.
        _preclean_mesh_pymeshlab(ms)

        # 2) Decimation proper.
        # Key params for CAD quality:
        #   planarquadric=True + planarweight small → preserves flat faces
        #   qualitythr=0.3 → forbids creating slivers that cause shading artefacts
        #   preservenormal=True → keeps shading smooth on existing faces
        #   preservetopology=True → won't create holes
        #   boundaryweight>1.0 → keeps silhouette sharp
        decimate_kwargs = dict(
            targetfacenum=int(target_faces),
            preserveboundary=preserve_boundary,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarquadric=True,
            planarweight=0.002,
            qualitythr=0.3,
            qualityweight=False,
            autoclean=True,
        )
        if preserve_boundary:
            decimate_kwargs["boundaryweight"] = 1.5

        filter_tried = False
        for filt_name in (
            "meshing_decimation_quadric_edge_collapse",
            "simplification_quadric_edge_collapse_decimation",
        ):
            if hasattr(ms, filt_name):
                try:
                    getattr(ms, filt_name)(**decimate_kwargs)
                    filter_tried = True
                    break
                except TypeError:
                    # Older pymeshlab: smaller accepted param set — fall back.
                    try:
                        getattr(ms, filt_name)(
                            targetfacenum=int(target_faces),
                            preserveboundary=preserve_boundary,
                            preservenormal=True,
                            preservetopology=True,
                            optimalplacement=True,
                            planarquadric=True,
                            autoclean=True,
                        )
                        filter_tried = True
                        break
                    except Exception:
                        continue
        if not filter_tried:
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse",
                **decimate_kwargs,
            )

        # 3) Light smoothing — flattens marching-cubes stair-steps but keeps
        # sharp corners if we use few iterations.
        if feature_smooth:
            _taubin_smooth(ms, steps=2)

        # 4) Final clean — duplicates can appear after decimation too.
        _preclean_mesh_pymeshlab(ms)

        out = ms.current_mesh()
        new_mesh = trimesh.Trimesh(
            vertices=out.vertex_matrix(),
            faces=out.face_matrix(),
            process=False,
        )
        if len(new_mesh.faces) < max(50, int(target_faces * 0.1)):
            print(
                f"[decimate] pymeshlab produced too few faces "
                f"({len(new_mesh.faces)} vs target {target_faces}) — "
                f"falling back to trimesh",
                flush=True,
            )
            raise RuntimeError("too few faces after decimation")
        return new_mesh
    except Exception as e:
        print(f"[decimate] pymeshlab failed ({e}); falling back to trimesh", flush=True)
        try:
            return mesh.simplify_quadric_decimation(int(target_faces))
        except Exception as e2:
            print(f"[decimate] trimesh fallback also failed: {e2}", flush=True)
            return mesh  # better than nothing


# ==== Main generation =======================================================
DOWNLOAD_FORMATS = ["glb", "obj", "stl", "ply", "dae", "off", "3mf"]


def generate_3d(
    image: Optional[Image.Image],
    remove_bg: bool,
    num_steps: int,
    guidance_scale: float,
    octree_resolution: int,
    seed: int,
    track_mps: bool,
    lang: str,
):
    """Generate a mesh from an image.

    Returns the tuple Gradio expects:
      (preview_glb_path, mesh_state, log_text, mps_stats_markdown)
    """
    if image is None:
        return None, None, t(lang, "log_warn_no_image"), ""

    log_lines: list[str] = []
    t0 = time.time()
    log_lines.append(t(lang, "log_device", device=DEVICE, dtype=DTYPE))
    log_lines.append(t(lang, "log_preprocess", remove_bg=remove_bg))
    image = preprocess_image(image, remove_bg)
    log_lines.append(t(lang, "log_after_size", size=image.size))

    log_lines.append(t(lang, "log_loading_pipeline"))
    try:
        pipe = _load_pipeline()
    except Exception as e:
        return None, None, t(lang, "log_err_model", e=e), ""

    # Diagnostics — catch None components BEFORE the call.
    diag = []
    for attr in ("vae", "model", "scheduler", "conditioner", "image_processor"):
        obj = getattr(pipe, attr, "__MISSING__")
        if obj == "__MISSING__":
            diag.append(f"  {attr}: <no attribute>")
        elif obj is None:
            diag.append(f"  {attr}: None  ← cause is here!")
        else:
            diag.append(f"  {attr}: {type(obj).__name__}")
    diag_str = "\n".join(diag)
    print(f"[diag] Pipeline state:\n{diag_str}", flush=True)
    log_lines.append(t(lang, "log_pipeline_comps", diag=diag_str))

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    log_lines.append(t(
        lang, "log_generation",
        steps=num_steps, guidance=guidance_scale,
        octree=octree_resolution, seed=seed, track_mps=track_mps,
    ))

    tracker: Optional[DeviceUsageTracker] = None
    mps_stats_md = ""
    try:
        if track_mps and _HAS_TFMODE:
            tracker = DeviceUsageTracker()
            tracker_ctx = tracker
        else:
            tracker_ctx = contextlib.nullcontext()

        with torch.inference_mode(), tracker_ctx:
            result = pipe(
                image=image,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                octree_resolution=int(octree_resolution),
                generator=gen,
            )
    except RuntimeError as e:
        msg = str(e)
        tb = traceback.format_exc()
        print(tb, flush=True)
        if "out of memory" in msg.lower() or "mps" in msg.lower():
            return None, None, t(lang, "log_err_oom", msg=msg, tb=tb), ""
        return None, None, t(lang, "log_err_inference", msg=msg, tb=tb), ""
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return None, None, t(lang, "log_err_unknown", name=type(e).__name__, e=e, tb=tb), ""
    finally:
        if DEVICE == "mps":
            torch.mps.empty_cache()
        gc.collect()

    mesh = result[0] if isinstance(result, (list, tuple)) else result
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = getattr(mesh, "mesh", mesh)

    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    log_lines.append(t(lang, "log_mesh_ready", verts=n_v, faces=n_f))

    glb_path = mesh_to_file(mesh, "glb", suffix="raw")

    elapsed = time.time() - t0
    log_lines.append(t(lang, "log_done", elapsed=elapsed))
    log_lines.append(t(lang, "log_preview_glb", path=glb_path))
    log_lines.append(t(lang, "log_next_step"))

    if tracker is not None:
        mps_stats_md = t(lang, "mps_stats_header") + "\n" + tracker.summary_text(lang=lang)
    elif track_mps:
        mps_stats_md = t(lang, "mps_stats_off_note")

    # mesh_state keeps two meshes:
    #   original — raw output of the model, IMMUTABLE
    #   mesh     — current preview (= original or an optimized version)
    # This matters so that switching «Strong → Medium» re-runs Medium on the
    # original, not on the already-shrunk mesh.
    mesh_state = {
        "original": mesh,
        "mesh": mesh,
        "original_faces": n_f,
        "original_vertices": n_v,
        "last_op": "original",
    }
    return glb_path, mesh_state, "\n".join(log_lines), mps_stats_md


# ==== Post-processing and download handlers ================================
def optimize_3d_handler(
    mesh_state: Optional[dict],
    mode_key: str,
    level_key: str,
    lang: str,
):
    """Apply the chosen algorithm + level to the ORIGINAL mesh.

    Returns (preview_glb, new_state, log).
    """
    if not mesh_state or "original" not in mesh_state:
        if mesh_state and "mesh" in mesh_state:
            mesh_state = dict(mesh_state)
            mesh_state["original"] = mesh_state["mesh"]
        else:
            return None, mesh_state, t(lang, "log_no_mesh_yet")

    original: trimesh.Trimesh = mesh_state["original"]
    n_v0, n_f0 = len(original.vertices), len(original.faces)

    # «None» is independent of algorithm — just surface the original mesh.
    if level_key == "none":
        preview = mesh_to_file(original, "glb", suffix="raw")
        new_state = dict(mesh_state)
        new_state["mesh"] = original
        new_state["last_op"] = "original"
        return preview, new_state, t(
            lang, "log_shown_original", verts=n_v0, faces=n_f0,
        )

    t0 = time.time()
    mode_label = t(lang, f"opt_mode_{mode_key}")
    level_label = t(lang, f"opt_{level_key}")
    log_lines = [t(
        lang, "log_optimizing",
        mode=mode_label, level=level_label, faces=n_f0,
    )]

    try:
        result = lowpoly.optimize(original, mode=mode_key, level=level_key)
    except Exception as e:
        tb = traceback.format_exc()
        log_lines.append(f"❌ {type(e).__name__}: {e}\n{tb}")
        return None, mesh_state, "\n".join(log_lines)

    new_mesh = result.mesh
    n_v1, n_f1 = len(new_mesh.vertices), len(new_mesh.faces)

    if mode_key == "auto" and result.classifier_ratio is not None:
        chosen_label = t(lang, f"opt_mode_{result.mode_used}")
        log_lines.append(t(
            lang, "log_auto_pick",
            ratio=result.classifier_ratio * 100.0, chosen=chosen_label,
        ))

    suffix = f"{mode_key}_{level_key}_{n_f1}"
    preview = mesh_to_file(new_mesh, "glb", suffix=suffix)
    elapsed = time.time() - t0
    saved_pct = 100.0 * (1.0 - n_f1 / max(n_f0, 1))
    log_lines.append(t(
        lang, "log_decimation_done",
        elapsed=elapsed, v0=n_v0, v1=n_v1, f0=n_f0, f1=n_f1, pct=saved_pct,
    ))
    log_lines.append(t(lang, "log_preview", path=preview))

    new_state = {
        "original": original,
        "mesh": new_mesh,
        "original_faces": n_f0,
        "original_vertices": n_v0,
        "last_op": f"{mode_key}_{level_key}",
    }
    return preview, new_state, "\n".join(log_lines)


def download_3d_handler(mesh_state: Optional[dict], fmt: str):
    """Export the current mesh in the chosen format. Returns the file path."""
    if not mesh_state or "mesh" not in mesh_state:
        return None
    mesh: trimesh.Trimesh = mesh_state["mesh"]
    suffix = mesh_state.get("last_op", "")
    safe_suffix = "".join(c if c.isalnum() else "_" for c in str(suffix))[:32]
    path = mesh_to_file(mesh, fmt, suffix=safe_suffix)
    return path


# ==== Gradio UI =============================================================
DEFAULT_LANG = "en"

with gr.Blocks(title="Hunyuan3D 2.1") as demo:
    # Language state — stored as an internal code ("en"/"zh"/"ru").
    lang_state = gr.State(value=DEFAULT_LANG)
    # Mesh state — a Python dict held in memory between calls.
    mesh_state = gr.State(value=None)

    # Language picker sits at the very top.
    lang_radio = gr.Radio(
        choices=[LANG_DISPLAY[k] for k in LANG_CODES],
        value=LANG_DISPLAY[DEFAULT_LANG],
        label=t(DEFAULT_LANG, "lang_label"),
    )

    desc_md = gr.Markdown(t(
        DEFAULT_LANG, "description",
        device=DEVICE, dtype=DTYPE, project=PROJECT_DIR,
    ))

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(
                label=t(DEFAULT_LANG, "input_image"),
                type="pil",
                sources=["upload", "clipboard"],
                height=360,
            )
            remove_bg = gr.Checkbox(
                label=t(DEFAULT_LANG, "remove_bg"),
                value=True,
            )
            track_mps = gr.Checkbox(
                label=t(DEFAULT_LANG, "track_mps"),
                value=False,
            )
            with gr.Accordion(t(DEFAULT_LANG, "gen_params"), open=False) as gen_acc:
                num_steps = gr.Slider(
                    minimum=10, maximum=100, value=50, step=5,
                    label=t(DEFAULT_LANG, "num_steps"),
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=15.0, value=5.0, step=0.5,
                    label=t(DEFAULT_LANG, "guidance"),
                )
                octree_resolution = gr.Dropdown(
                    choices=[128, 192, 256, 320, 384],
                    value=256,
                    label=t(DEFAULT_LANG, "octree"),
                )
                seed = gr.Number(
                    value=42, precision=0, label=t(DEFAULT_LANG, "seed"),
                )
            btn = gr.Button(t(DEFAULT_LANG, "generate_btn"), variant="primary")
            log = gr.Textbox(label=t(DEFAULT_LANG, "log_label"), lines=14, max_lines=30)

        with gr.Column(scale=1):
            model3d = gr.Model3D(
                label=t(DEFAULT_LANG, "preview_label"),
                clear_color=[0.0, 0.0, 0.0, 0.0],
                height=420,
            )

            mps_stats = gr.Markdown(value=t(DEFAULT_LANG, "mps_stats_initial"))

            with gr.Group():
                postproc_md = gr.Markdown(t(DEFAULT_LANG, "postproc_header"))
                opt_mode = gr.Radio(
                    choices=_opt_mode_choices(DEFAULT_LANG),
                    value="quadric",
                    label=t(DEFAULT_LANG, "opt_mode_label"),
                )
                opt_mode_help = gr.Markdown(t(DEFAULT_LANG, "opt_mode_help"))
                opt_level = gr.Radio(
                    choices=_opt_level_choices(DEFAULT_LANG),
                    value="medium",
                    label=t(DEFAULT_LANG, "opt_level_label"),
                )
                opt_btn = gr.Button(t(DEFAULT_LANG, "opt_btn"), variant="secondary")
                opt_log = gr.Textbox(
                    label=t(DEFAULT_LANG, "opt_log"),
                    lines=4,
                    max_lines=10,
                    interactive=False,
                )

            with gr.Group():
                download_md = gr.Markdown(t(DEFAULT_LANG, "download_header"))
                dl_format = gr.Radio(
                    choices=DOWNLOAD_FORMATS,
                    value="glb",
                    label=t(DEFAULT_LANG, "dl_format"),
                )
                dl_btn = gr.Button(t(DEFAULT_LANG, "dl_btn"), variant="secondary")
                dl_file = gr.File(label=t(DEFAULT_LANG, "dl_file"))

    tips_md = gr.Markdown(t(DEFAULT_LANG, "tips"))

    # --- language switching logic ------------------------------------------
    def apply_language(lang_display: str):
        """Return a dict of component updates for the chosen language."""
        lang = LANG_BY_DISPLAY.get(lang_display, "en")
        return (
            lang,  # lang_state
            gr.update(label=t(lang, "lang_label")),                          # lang_radio
            gr.update(value=t(lang, "description",
                              device=DEVICE, dtype=DTYPE, project=PROJECT_DIR)),  # desc_md
            gr.update(label=t(lang, "input_image")),                         # img_in
            gr.update(label=t(lang, "remove_bg")),                           # remove_bg
            gr.update(label=t(lang, "track_mps")),                           # track_mps
            gr.update(label=t(lang, "gen_params")),                          # gen_acc
            gr.update(label=t(lang, "num_steps")),                           # num_steps
            gr.update(label=t(lang, "guidance")),                            # guidance_scale
            gr.update(label=t(lang, "octree")),                              # octree_resolution
            gr.update(label=t(lang, "seed")),                                # seed
            gr.update(value=t(lang, "generate_btn")),                        # btn
            gr.update(label=t(lang, "log_label")),                           # log
            gr.update(label=t(lang, "preview_label")),                       # model3d
            gr.update(value=t(lang, "mps_stats_initial")),                   # mps_stats
            gr.update(value=t(lang, "postproc_header")),                     # postproc_md
            gr.update(
                choices=_opt_mode_choices(lang),
                label=t(lang, "opt_mode_label"),
            ),                                                               # opt_mode
            gr.update(value=t(lang, "opt_mode_help")),                       # opt_mode_help
            gr.update(
                choices=_opt_level_choices(lang),
                label=t(lang, "opt_level_label"),
            ),                                                               # opt_level
            gr.update(value=t(lang, "opt_btn")),                             # opt_btn
            gr.update(label=t(lang, "opt_log")),                             # opt_log
            gr.update(value=t(lang, "download_header")),                     # download_md
            gr.update(label=t(lang, "dl_format")),                           # dl_format
            gr.update(value=t(lang, "dl_btn")),                              # dl_btn
            gr.update(label=t(lang, "dl_file")),                             # dl_file
            gr.update(value=t(lang, "tips")),                                # tips_md
        )

    lang_radio.change(
        fn=apply_language,
        inputs=[lang_radio],
        outputs=[
            lang_state,
            lang_radio, desc_md,
            img_in, remove_bg, track_mps,
            gen_acc, num_steps, guidance_scale, octree_resolution, seed,
            btn, log, model3d, mps_stats,
            postproc_md, opt_mode, opt_mode_help, opt_level, opt_btn, opt_log,
            download_md, dl_format, dl_btn, dl_file,
            tips_md,
        ],
    )

    btn.click(
        fn=generate_3d,
        inputs=[img_in, remove_bg, num_steps, guidance_scale,
                octree_resolution, seed, track_mps, lang_state],
        outputs=[model3d, mesh_state, log, mps_stats],
    )

    opt_btn.click(
        fn=optimize_3d_handler,
        inputs=[mesh_state, opt_mode, opt_level, lang_state],
        outputs=[model3d, mesh_state, opt_log],
    )

    dl_btn.click(
        fn=download_3d_handler,
        inputs=[mesh_state, dl_format],
        outputs=[dl_file],
    )

# ==== Entry point ===========================================================
if __name__ == "__main__":
    print(f"[launch] Device: {DEVICE}, Weights: {WEIGHTS_DIR}")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
