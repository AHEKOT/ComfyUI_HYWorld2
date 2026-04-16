# ComfyUI HY-World 2.0 — WorldMirror 3D

ComfyUI custom nodes for 3D scene reconstruction from a single image or panorama using [HY-World 2.0](https://huggingface.co/tencent/HY-World-2.0) (Tencent).

---

**If you find my project useful, please consider supporting it! I work on it completely on my own, and your support will allow me to continue maintaining it and adding even more cool features!**

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20rtx%203090&emoji=☕&slug=MIUProject&button_colour=FFDD00&font_colour=000000&font_family=Comic&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/MIUProject)
---

## Nodes

**Category: VNCCS/3D**

| Node | Description |
|------|-------------|
| `VNCCS_LoadWorldMirrorModel` | Download and load WorldMirror V1 model |
| `VNCCS_WorldMirror3D` | V1 inference — outputs PLY point cloud, depth, normals, Gaussian splat |
| `VNCCS_LoadWorldMirrorV2Model` | Download and load WorldMirror V2 model |
| `VNCCS_WorldMirrorV2_3D` | V2 inference — outputs PLY point cloud, depth, normals, Gaussian splat |
| `VNCCS_PLYSceneRenderer` | Render PLY scene from arbitrary camera angles |
| `VNCCS_SplatRefiner` | Refine Gaussian splat data |
| `VNCCS_DecomposePLYData` | Extract XYZ / RGB / normals / opacity tensors from PLY |
| `VNCCS_SavePLY` | Save PLY file to disk |
| `VNCCS_BackgroundPreview` | Preview 3D background renders |
| `VNCCS_Equirect360ToViews` | Extract perspective views from equirectangular panorama |
| `VNCCS_PanoramaMapper` | Map panorama to wall / floor / ceiling projections |

---

## Installation

### Via ComfyUI Manager (recommended)

Search for **HY-World 2.0** and click Install. `requirements.txt` and `install.py` run automatically.

`install.py` will attempt to install `gsplat` — first from a pre-built wheel, then by compiling from source if no wheel is available for your platform.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AHEKOT/ComfyUI_HYWorld2
cd ComfyUI_HYWorld2
pip install -r requirements.txt
python install.py
```

---

## gsplat — Requirements for Compilation

`gsplat` is a CUDA extension and cannot always be installed from a pre-built wheel. If your Python version, PyTorch version, or CUDA version is not covered by an official wheel, it must be compiled from source.

The `install.py` script handles this automatically, but the following tools must be present on the system.

### Required

**1. CUDA Toolkit**

Install the CUDA Toolkit version that matches your PyTorch build:

```
PyTorch CUDA version  →  Required CUDA Toolkit
cu118                 →  CUDA 11.8
cu121                 →  CUDA 12.1
cu124                 →  CUDA 12.4
```

Download: https://developer.nvidia.com/cuda-toolkit-archive

After installation, verify:
```
nvcc --version
```

**2. MSVC C++ Compiler (Windows only)**

Install **Visual Studio Build Tools 2019 or 2022** with the **"Desktop development with C++"** workload.

Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

After installation, verify (from a Developer Command Prompt):
```
cl
```

If neither system MSVC nor a portable compiler is found, the build script will attempt to download a portable MSVC automatically (~600 MB).

**3. Git**

Required to clone the gsplat source repository if no pre-built wheel is available.

```
git --version
```

### Optional

- **ninja** — speeds up compilation significantly. Installed automatically by the build script if missing.

### Pre-built wheels (no compilation needed)

If a wheel exists for your exact combination of Python / PyTorch / CUDA, the build script will use it instead of compiling. Check availability at:

https://docs.gsplat.studio/whl/

### Manual build

To run the gsplat build independently:

```bash
# Windows
scripts\pipinstall.bat

# Any platform
python scripts/build_gsplat.py
```

---

## Workflows

Example workflows are in the `workflows/` directory:

- `World-single-image.json` — single image to 3D scene
- `World-Mirror-panorama.json` — equirectangular panorama to 3D scene

---

## Requirements

- Python 3.10+
- PyTorch with CUDA
- NVIDIA GPU (gsplat rendering requires CUDA)
- Flash Attention — required for HY-World 2.0 model inference. Install via:
  ```
  pip install flash-attn --no-build-isolation
  ```
