# Hunyuan3D 2.1 for Mac / Apple Silicon (M1, M2, M3, M4)

**Run Tencent's Hunyuan3D 2.1 image-to-3D model on Apple Silicon Macs — natively via Metal Performance Shaders (MPS).** No CUDA, no Linux VM, no cloud GPU required.

This repository provides a turnkey Mac installer, a CUDA-to-MPS compatibility patch, and a localized Gradio web UI (English / 中文 / Русский) around the upstream [Tencent Hunyuan 3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) project. Take a single PNG/JPG image, get a clean `.obj` / `.glb` / `.ply` / `.stl` / `.fbx` / `.dae` / `.3mf` mesh — running entirely on your Mac's GPU.

**Keywords:** Hunyuan3D, Hunyuan 3D 2.1, image to 3D, Mac, macOS, Apple Silicon, M1, M2, M3, M4, Metal Performance Shaders, MPS, PyTorch MPS, 3D generation, AI 3D model, mesh generation, image-to-mesh, DiT, Tencent.

---

<img width="1874" height="861" alt="image" src="https://github.com/user-attachments/assets/95433b87-9cdd-4d05-a99e-509723cd8ac0" />

<img width="925" height="505" alt="image" src="https://github.com/user-attachments/assets/29d60e3a-3f12-4be3-a4a5-ab47e5d8bff0" />


## Why this repo exists

The upstream Hunyuan 3D 2.1 release targets NVIDIA CUDA GPUs on Linux. On a Mac you'll hit three blockers out of the box:

1. **Hardcoded `cuda` device strings** throughout the shape pipeline.
2. **`.safetensors`-only weight loader**, while Hugging Face actually ships the 2.1 checkpoint as `.ckpt`.
3. **`xformers` / `flash-attn` / `bitsandbytes` / `onnxruntime-gpu`** dependencies that don't build on Apple Silicon.

This project fixes all three at install time and wraps everything in a clean UI.

---

## What you get

- One-command installer (`./install.sh`) that clones the upstream repo, creates a venv, installs Mac-compatible dependencies, downloads the weights from Hugging Face, and applies the MPS patches.
- Gradio web UI with trilingual interface — **English (default), 中文, Русский** — and English-only console logs.
- Live **MPS / CPU operation tracker** so you can see what fraction of tensor operations actually ran on the GPU.
- Built-in **mesh post-processing** via pymeshlab (Quadric Edge Collapse Decimation) with four quality presets — Minimal (~90k tris) / Low (~30k) / Medium (~10k) / High (~5k).
- **Seven export formats** selected at download time: OBJ, GLB, PLY, STL, FBX, DAE, 3MF.
- Upstream source and weights are kept outside the repo (`.gitignore`d) — they are cloned and downloaded fresh by the installer.

---

## System requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Hardware  | Apple Silicon Mac (M1 / M2 / M3 / M4, any variant) | M-series Pro / Max / Ultra |
| Unified memory | 16 GB | 24 GB+ |
| Free disk | ~15 GB (repo + venv + weights) | 25 GB |
| macOS | 13 Ventura | 14 Sonoma or later |
| Python | 3.10 – 3.11 | 3.12 |
| Homebrew | any recent version | latest |

Tested on an M4 Pro with 24 GB unified memory.

> Intel Macs are **not** supported — the MPS backend requires Apple Silicon.

---

## Install

```bash
# 1. Clone this wrapper repo
git clone https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac.git
cd hunyuan3d-2.1-mac

# 2. Run the installer (takes 10–20 min; ~12 GB of weights)
chmod +x install.sh
./install.sh
```

The installer will:

1. Verify you're on Apple Silicon + Python 3.10/3.11.
2. `git clone` the upstream Tencent repo into `./Hunyuan3D-2.1/`.
3. Create a venv in `./venv/`.
4. Install PyTorch with MPS, Gradio 5.x, trimesh, pymeshlab and the rest of `requirements_mac.txt`.
5. Download the 2.1 weights from Hugging Face into `./weights/`.
6. Run `fix.sh` to apply the CUDA→MPS patches and the `.ckpt` loader patch against your local clone.

If anything goes wrong later (e.g. you upgraded macOS and some dep broke), just re-run:

```bash
./fix.sh
```

## Run

```bash
./launch.sh
```

Open the URL Gradio prints (usually http://127.0.0.1:7860). Upload an image, hit **Generate**, wait 2–5 minutes, preview the mesh, optionally post-process, pick a format, download.

---

## Features in the UI

- **Language switcher** — English / 中文 / Русский. Affects the UI only; console logs stay in English.
- **Background removal** — optional, via rembg.
- **Inference parameters** — steps, guidance scale, octree resolution, seed.
- **MPS usage tracker** — toggle to measure the actual share of tensor operations that run on the GPU vs CPU. Uses `torch.overrides.TorchFunctionMode` to intercept every tensor op; the summary appears in the status area (e.g. `MPS: 100.0% (154920 ops) / CPU: 0.0% (22 ops)`).
- **Post-processing** — 4 decimation presets powered by pymeshlab's Quadric Edge Collapse. Always starts from the original generated mesh (idempotent — switching presets does not compound error).
- **Download** — pick your format at download time, so you don't have to regenerate the mesh to switch from `.obj` to `.glb`.

---

## Known limitations on Mac

- **No texturing / painting.** The upstream texture pipeline relies on CUDA-only components (xformers, a CUDA-compiled differentiable renderer). This port exposes shape generation only. Bring your textured mesh into Blender / Substance / Maya for materials.
- **Speed.** Expect roughly 2–5 minutes per mesh on an M4 Pro with default settings (50 steps, octree 256). That's slower than a high-end NVIDIA GPU but fully usable locally.
- **Memory pressure.** On 16 GB Macs you may need to lower `octree_resolution` to 192 and close other apps.
- **First run warmup.** The first generation of a session is noticeably slower while MPS kernels compile.

---

## Project layout

```
hunyuan3d-2.1-mac/
├── install.sh            ← one-shot installer
├── fix.sh                ← re-applies CUDA→MPS and .ckpt patches
├── launch.sh             ← activates venv and starts the UI
├── gradio_app.py         ← the trilingual Gradio app
├── requirements_mac.txt  ← Mac-safe pinned deps
├── LICENSE               ← Tencent Hunyuan 3D 2.1 Community License (governs model + derivatives)
├── LICENSE-WRAPPERS      ← MIT (governs the wrapper scripts above)
├── NOTICE                ← Tencent Notice.txt + Mac-port addendum
├── .gitignore
└── README.md
(after install)
├── Hunyuan3D-2.1/        ← upstream clone, not tracked
├── venv/                 ← Python environment, not tracked
├── weights/              ← HF model weights, not tracked
└── outputs/              ← your generated meshes, not tracked
```

---

## Credits and upstream

Model and original research are by **Tencent Hunyuan**. This Mac port adds no model capability; it exists purely to make Hunyuan 3D 2.1 run on Apple Silicon.

- Upstream repo: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- Weights on Hugging Face: https://huggingface.co/tencent/Hunyuan3D-2.1
- Paper / project page: see the upstream README.

If you use this project in research or production, please cite the Tencent Hunyuan 3D 2.1 paper alongside acknowledging upstream Tencent.

---

## License — please read before using

This repository contains **two separately-licensed layers**:

1. The Tencent Hunyuan 3D 2.1 model, weights and any mesh you generate with them are governed by the **TENCENT HUNYUAN 3D 2.1 COMMUNITY LICENSE AGREEMENT** — see [`LICENSE`](./LICENSE).

   Important points of that license:

   - **Territory:** the license does **not** apply in the **European Union, the United Kingdom, or South Korea.** If you are in one of those jurisdictions you may not use the model under this license.
   - **Commercial threshold:** if your product or service has more than **100 million monthly active users**, you need a separate commercial license from Tencent.
   - **Attribution:** downstream distributions must preserve Tencent's copyright and license text.
   - **State your changes:** this port documents its modifications in [`NOTICE`](./NOTICE) (CUDA→MPS patch, `.ckpt` loader patch, both applied to a local clone — not redistributed).

2. The Mac wrapper scripts I wrote (`install.sh`, `fix.sh`, `launch.sh`, `gradio_app.py`, `requirements_mac.txt`, `README.md`) are **MIT-licensed** — see [`LICENSE-WRAPPERS`](./LICENSE-WRAPPERS).

This project is **not** affiliated with, endorsed by, or sponsored by Tencent. "Hunyuan" is a trademark of Tencent.

---

## Troubleshooting

- **"No module named hy3dshape"** — the upstream repo wasn't cloned or wasn't patched. Run `./fix.sh`.
- **"RuntimeError: Placeholder storage has not been allocated on MPS device"** — a tensor slipped onto the wrong device. Make sure you ran `./fix.sh` after any `git pull` inside `Hunyuan3D-2.1/`.
- **Gradio shows a warning about themes** — harmless; the theme is applied via `.launch()` as Gradio 5.x recommends.
- **`pymeshlab` fails to import on macOS 14+** — install via `pip install pymeshlab --no-cache-dir` inside the venv.
- **Out of memory during sampling** — lower `octree_resolution` to 192 or 128, or reduce `num_steps`.

---

## Contributing

Issues and PRs welcome — especially for:

- Texturing on Mac (even a slow CPU fallback would be valuable).
- Further MPS speed-ups.
- Additional language translations for the UI.

Please keep PRs scoped to the wrapper files. Upstream Tencent code should be fixed upstream, not vendored here.

---

Made with care on an M4 Pro. If this saved you a weekend, star the repo.
