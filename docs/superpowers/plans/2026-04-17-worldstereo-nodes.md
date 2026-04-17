# WorldStereo ComfyUI Nodes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three ComfyUI nodes that run WorldStereo camera-guided video generation from a single image and wire its output directly into the existing WorldMirror V2 3D reconstruction node.

**Architecture:** Clone FuchengSu/WorldStereo into `worldstereo/` at project root and add it to sys.path. Three nodes: `VNCCS_LoadWorldStereoModel` (downloads + loads pipeline), `VNCCS_CameraTrajectoryBuilder` (generates C2W poses using worldstereo's own `get_c2w`), `VNCCS_WorldStereoGenerate` (MoGe depth → point cloud render → pipeline call → frames + poses). Outputs `video_frames / camera_poses / camera_intrinsics` that plug directly into the existing `VNCCS_WorldMirrorV2_3D` node.

**Tech Stack:** PyTorch, diffusers ≥ 0.36, optimum-quanto (fp8/fp4), bitsandbytes ≥ 0.43, pytorch3d (point rendering), MoGe (depth), huggingface_hub

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `worldstereo/` | Create (git clone) | WorldStereo source — models, pipelines, camera utils, pointcloud |
| `nodes/world_stereo.py` | Create | All 3 nodes + download helpers + preprocessing |
| `nodes/__init__.py` | Modify | Import WorldStereo node mappings |
| `requirements.txt` | Modify | Add diffusers, optimum-quanto, bitsandbytes, imageio, kornia |
| `tests/test_world_stereo_nodes.py` | Create | Unit tests for trajectory math and tensor contracts |

---

## Task 1: Clone WorldStereo and update dependencies

**Files:**
- Create: `worldstereo/` (git clone)
- Modify: `requirements.txt`

- [ ] **Step 1.1: Clone the repo**

```bash
cd /Users/ahekot/Documents/Development/ComfyUI_HYWorld2
git clone https://github.com/FuchengSu/WorldStereo.git worldstereo
```

Expected: `worldstereo/` directory with `models/`, `src/`, `run_camera_control.py`, etc.

- [ ] **Step 1.2: Verify required source files exist**

```bash
ls worldstereo/models/worldstereo_wrapper.py
ls worldstereo/models/camera.py
ls worldstereo/src/pointcloud.py
ls worldstereo/src/camera_utils.py
ls worldstereo/src/data_utils.py
```

Expected: all 5 files present, no errors.

- [ ] **Step 1.3: Add dependencies to requirements.txt**

In `requirements.txt`, after the `# CPU offloading` block, add:

```
# WorldStereo
diffusers>=0.36.0
transformers>=5.2.0
bitsandbytes>=0.43.0
optimum-quanto>=0.2.4
imageio[ffmpeg]
kornia
```

- [ ] **Step 1.4: Note PyTorch3D installation**

PyTorch3D requires a CUDA-compiled wheel. Add a comment to `requirements.txt` above the WorldStereo block:

```
# PyTorch3D (required for WorldStereo point rendering) — install manually:
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

- [ ] **Step 1.5: Note MoGe installation**

Add after PyTorch3D comment:

```
# MoGe depth estimator (required for WorldStereo) — install manually:
# pip install "git+https://github.com/microsoft/MoGe.git"
```

- [ ] **Step 1.6: Commit**

```bash
git add worldstereo/ requirements.txt
git commit -m "feat: clone WorldStereo repo and add dependencies"
```

---

## Task 2: Unit tests for trajectory math

**Files:**
- Create: `tests/test_world_stereo_nodes.py`

Write the tests before the implementation so they drive the interface.

- [ ] **Step 2.1: Create the test file**

Create `tests/test_world_stereo_nodes.py`:

```python
import math
import sys
import os

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "worldstereo"))


class TestBuildIntrinsics:
    def test_shape(self):
        from nodes.world_stereo import _build_intrinsics
        K = _build_intrinsics(70.0, 768, 480)
        assert K.shape == (3, 3)

    def test_principal_point_at_center(self):
        from nodes.world_stereo import _build_intrinsics
        K = _build_intrinsics(70.0, 768, 480)
        assert K[0, 2].item() == pytest.approx(384.0)
        assert K[1, 2].item() == pytest.approx(240.0)

    def test_focal_from_fov(self):
        from nodes.world_stereo import _build_intrinsics
        K = _build_intrinsics(90.0, 200, 100)
        # fov=90 → fx = (200/2)/tan(45°) = 100
        assert K[0, 0].item() == pytest.approx(100.0, abs=1e-3)


class TestBuildTrajectory:
    def test_circular_shape(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("circular", 16, 1.0, 0.0, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (16, 4, 4)
        assert intrs.shape == (16, 3, 3)

    def test_forward_shape(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("forward", 8, 1.0, 0.5, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (8, 4, 4)

    def test_zoom_in_shape(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("zoom_in", 12, 1.0, 0.0, 0.0, 70.0, 768, 480)
        assert c2ws.shape == (12, 4, 4)

    def test_all_c2ws_are_valid_se3(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, _ = _build_trajectory("circular", 8, 1.0, 0.0, 0.0, 70.0, 768, 480)
        # Bottom row must be [0,0,0,1]
        for i in range(8):
            bottom = c2ws[i, 3, :]
            assert bottom[3].item() == pytest.approx(1.0, abs=1e-5)
            assert bottom[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_intrinsics_replicated_per_frame(self):
        from nodes.world_stereo import _build_trajectory
        c2ws, intrs = _build_trajectory("circular", 5, 1.0, 0.0, 0.0, 70.0, 768, 480)
        # All intrinsics frames must be identical
        for i in range(1, 5):
            assert torch.allclose(intrs[0], intrs[i])


class TestC2WToW2C:
    def test_identity_roundtrip(self):
        from nodes.world_stereo import _c2w_to_w2c
        c2ws = torch.eye(4).unsqueeze(0).repeat(4, 1, 1)
        w2cs = _c2w_to_w2c(c2ws)
        assert torch.allclose(w2cs, c2ws, atol=1e-5)

    def test_shape_preserved(self):
        from nodes.world_stereo import _c2w_to_w2c
        c2ws = torch.randn(10, 4, 4)
        # Make valid SE3
        c2ws[:, 3, :] = torch.tensor([0., 0., 0., 1.])
        w2cs = _c2w_to_w2c(c2ws)
        assert w2cs.shape == (10, 4, 4)
```

- [ ] **Step 2.2: Run tests — expect ImportError (functions don't exist yet)**

```bash
cd /Users/ahekot/Documents/Development/ComfyUI_HYWorld2
python -m pytest tests/test_world_stereo_nodes.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name '_build_intrinsics' from 'nodes.world_stereo'` — confirms tests are wired correctly.

- [ ] **Step 2.3: Commit test file**

```bash
git add tests/test_world_stereo_nodes.py
git commit -m "test: add WorldStereo trajectory unit tests (failing)"
```

---

## Task 3: Implement helper functions + CameraTrajectoryBuilder

**Files:**
- Create: `nodes/world_stereo.py` (initial version with helpers + trajectory node only)

- [ ] **Step 3.1: Create nodes/world_stereo.py with helpers and trajectory node**

Create `nodes/world_stereo.py`:

```python
"""
WorldStereo ComfyUI nodes.

Nodes:
  - VNCCS_LoadWorldStereoModel       — download + load WorldStereo pipeline
  - VNCCS_CameraTrajectoryBuilder    — generate C2W camera trajectories
  - VNCCS_WorldStereoGenerate        — run inference, output frames + poses
"""

import os
import sys
import math
import json
import tempfile

import numpy as np
import torch

# ── project root + worldstereo source on path ─────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDSTEREO_ROOT = os.path.join(PROJECT_ROOT, "worldstereo")
for _p in [PROJECT_ROOT, WORLDSTEREO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers — testable without GPU
# ─────────────────────────────────────────────────────────────────────────────

def _build_intrinsics(fov_deg: float, width: int, height: int) -> torch.Tensor:
    """Build a 3×3 intrinsics matrix from FOV (degrees) and image dimensions."""
    fx = fy = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx, cy = width / 2.0, height / 2.0
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    return K


def _c2w_to_w2c(c2ws: torch.Tensor) -> torch.Tensor:
    """Convert batch of C2W [N,4,4] matrices to W2C via batch inverse."""
    return torch.linalg.inv(c2ws)


def _build_trajectory(
    preset: str,
    num_frames: int,
    radius: float,
    speed: float,
    elevation_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    median_depth: float = 1.0,
    custom_json: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build C2W trajectory [N,4,4] and replicated intrinsics [N,3,3].

    Uses worldstereo's own get_c2w() for all built-in presets.
    custom_json: JSON list of N 4×4 lists (row-major).
    """
    from src.camera_utils import get_c2w

    K = _build_intrinsics(fov_deg, width, height)
    intrs = K.unsqueeze(0).repeat(num_frames, 1, 1)  # [N,3,3]

    c2w_start = torch.eye(4, dtype=torch.float32).numpy()

    if preset == "custom":
        data = json.loads(custom_json)
        c2ws = torch.tensor(data, dtype=torch.float32)  # [N,4,4]
        return c2ws, intrs

    # Map presets to get_c2w motion types + params
    if preset == "circular":
        # eloop: elliptical orbit around scene center
        c2ws_np = get_c2w(
            c2w_start, move="eloop",
            median_depth=median_depth,
            nframe=num_frames,
            r_x=radius, r_y=radius,
        )
    elif preset == "forward":
        c2ws_np = get_c2w(
            c2w_start, move="normal",
            median_depth=median_depth,
            nframe=num_frames,
            z=speed,         # forward translation per frame
            x=0.0, y=0.0, phi=0.0, theta=0.0,
        )
    elif preset == "zoom_in":
        c2ws_np = get_c2w(
            c2w_start, move="normal",
            median_depth=median_depth,
            nframe=num_frames,
            z=-radius / num_frames,  # negative = zoom toward scene
            x=0.0, y=0.0, phi=0.0, theta=0.0,
        )
    elif preset == "zoom_out":
        c2ws_np = get_c2w(
            c2w_start, move="normal",
            median_depth=median_depth,
            nframe=num_frames,
            z=radius / num_frames,
            x=0.0, y=0.0, phi=0.0, theta=0.0,
        )
    elif preset == "aerial":
        c2ws_np = get_c2w(
            c2w_start, move="aerial",
            median_depth=median_depth,
            nframe=num_frames,
            theta=math.radians(elevation_deg),
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    c2ws = torch.from_numpy(np.array(c2ws_np)).float()  # [N,4,4]
    return c2ws, intrs


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_CameraTrajectoryBuilder
# ─────────────────────────────────────────────────────────────────────────────

class VNCCS_CameraTrajectoryBuilder:
    """Build camera trajectories for WorldStereo generation."""

    PRESETS = ["circular", "forward", "zoom_in", "zoom_out", "aerial", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "preset": (cls.PRESETS, {"default": "circular"}),
                "num_frames": ("INT", {"default": 25, "min": 4, "max": 81}),
                "radius": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Orbit radius (circular) or travel distance (zoom).",
                }),
                "speed": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Per-frame translation for forward preset.",
                }),
                "elevation_deg": ("FLOAT", {
                    "default": 15.0, "min": -90.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Camera elevation for aerial preset.",
                }),
                "median_depth": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Estimated scene depth — used as orbit center distance.",
                }),
                "fov_deg": ("FLOAT", {"default": 70.0, "min": 10.0, "max": 150.0, "step": 1.0}),
                "image_width":  ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
                "image_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 64}),
                "custom_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON list of N 4×4 C2W matrices. Used when preset=custom.",
                }),
            }
        }

    RETURN_TYPES  = ("CAMERA_TRAJECTORY",)
    RETURN_NAMES  = ("trajectory",)
    FUNCTION      = "build"
    CATEGORY      = "VNCCS/Video"

    def build(
        self,
        preset="circular",
        num_frames=25,
        radius=1.0,
        speed=0.05,
        elevation_deg=15.0,
        median_depth=1.0,
        fov_deg=70.0,
        image_width=768,
        image_height=480,
        custom_json="[]",
    ):
        c2ws, intrs = _build_trajectory(
            preset=preset,
            num_frames=num_frames,
            radius=radius,
            speed=speed,
            elevation_deg=elevation_deg,
            fov_deg=fov_deg,
            width=image_width,
            height=image_height,
            median_depth=median_depth,
            custom_json=custom_json,
        )
        trajectory = {
            "c2ws":   c2ws,    # [N,4,4] C2W
            "intrs":  intrs,   # [N,3,3]
            "width":  image_width,
            "height": image_height,
        }
        print(f"✅ [Trajectory] preset={preset}, frames={c2ws.shape[0]}, size={image_width}×{image_height}")
        return (trajectory,)


# Placeholder stubs so __init__.py import doesn't fail before Tasks 4+5
class VNCCS_LoadWorldStereoModel:
    @classmethod
    def INPUT_TYPES(cls): return {"optional": {}}
    RETURN_TYPES = ("WORLDSTEREO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/Video"
    def load_model(self): raise NotImplementedError("Task 4 not yet implemented")


class VNCCS_WorldStereoGenerate:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {}}
    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("video_frames", "camera_poses", "camera_intrinsics")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/Video"
    def generate(self): raise NotImplementedError("Task 5 not yet implemented")


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    VNCCS_LoadWorldStereoModel,
    "VNCCS_CameraTrajectoryBuilder": VNCCS_CameraTrajectoryBuilder,
    "VNCCS_WorldStereoGenerate":     VNCCS_WorldStereoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    "🎬 Load WorldStereo Model",
    "VNCCS_CameraTrajectoryBuilder": "🎬 Camera Trajectory Builder",
    "VNCCS_WorldStereoGenerate":     "🎬 WorldStereo Generate",
}
```

- [ ] **Step 3.2: Register in nodes/__init__.py**

In `nodes/__init__.py`, add after the panorama_mapper imports:

```python
from .world_stereo import NODE_CLASS_MAPPINGS as WS_MAPPINGS
from .world_stereo import NODE_DISPLAY_NAME_MAPPINGS as WS_DISPLAY_MAPPINGS
```

And update the merge line:

```python
NODE_CLASS_MAPPINGS = {**V1_MAPPINGS, **V2_MAPPINGS, **PANORAMA_MAPPER_MAPPINGS, **WS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**V1_DISPLAY_MAPPINGS, **V2_DISPLAY_MAPPINGS, **PANORAMA_MAPPER_DISPLAY_MAPPINGS, **WS_DISPLAY_MAPPINGS}
```

- [ ] **Step 3.3: Run tests — expect 5 passing**

```bash
python -m pytest tests/test_world_stereo_nodes.py -v
```

Expected output:
```
PASSED test_shape
PASSED test_principal_point_at_center
PASSED test_focal_from_fov
PASSED test_circular_shape
PASSED test_forward_shape
PASSED test_zoom_in_shape
PASSED test_all_c2ws_are_valid_se3
PASSED test_intrinsics_replicated_per_frame
PASSED test_identity_roundtrip
PASSED test_shape_preserved
10 passed
```

If `from src.camera_utils import get_c2w` fails, check that `get_c2w` is the correct function name by inspecting `worldstereo/src/camera_utils.py` and adjusting the import + call accordingly.

- [ ] **Step 3.4: Commit**

```bash
git add nodes/world_stereo.py nodes/__init__.py
git commit -m "feat: add CameraTrajectoryBuilder node with trajectory helpers"
```

---

## Task 4: Implement VNCCS_LoadWorldStereoModel

**Files:**
- Modify: `nodes/world_stereo.py` — replace `VNCCS_LoadWorldStereoModel` stub

- [ ] **Step 4.1: Add download helper function**

In `nodes/world_stereo.py`, add after `_c2w_to_w2c` and before the node classes:

```python
def _get_models_base() -> str:
    return (
        folder_paths.models_dir if FOLDER_PATHS_AVAILABLE
        else os.path.join(PROJECT_ROOT, "models")
    )


def _download_worldstereo_components(model_type: str) -> tuple[str, str, str]:
    """
    Download all required model components.

    Returns (transformer_dir, base_model_dir, moge_dir).
    """
    from huggingface_hub import snapshot_download

    base = _get_models_base()

    # 1. WorldStereo transformer weights
    transformer_dir = os.path.join(base, "WorldStereo", model_type)
    transformer_safetensors = os.path.join(transformer_dir, "model.safetensors")
    if not os.path.exists(transformer_safetensors):
        print(f"⬇️ [WorldStereo] Downloading transformer ({model_type}) …")
        tmp = os.path.join(base, "WorldStereo", "_tmp")
        snapshot_download(
            repo_id="hanshanxue/WorldStereo",
            allow_patterns=[f"{model_type}/**"],
            local_dir=tmp,
        )
        # snapshot_download nests files under model_type subfolder
        nested = os.path.join(tmp, model_type)
        if os.path.isdir(nested):
            import shutil
            os.makedirs(transformer_dir, exist_ok=True)
            for f in os.listdir(nested):
                shutil.move(os.path.join(nested, f), transformer_dir)
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"✅ [WorldStereo] Transformer cached: {transformer_dir}")
    else:
        print(f"✅ [WorldStereo] Transformer cached: {transformer_dir}")

    # 2. Wan2.1 base model (VAE, T5, CLIP)
    base_model_dir = os.path.join(base, "Wan2.1-I2V-14B-480P")
    wan_vae = os.path.join(base_model_dir, "vae", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(wan_vae):
        print(f"⬇️ [WorldStereo] Downloading Wan2.1-I2V-14B-480P base model (~40 GB) …")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            local_dir=base_model_dir,
        )
        print(f"✅ [WorldStereo] Base model cached: {base_model_dir}")
    else:
        print(f"✅ [WorldStereo] Base model cached: {base_model_dir}")

    # 3. MoGe depth estimator
    moge_dir = os.path.join(base, "MoGe")
    moge_config = os.path.join(moge_dir, "config.json")
    if not os.path.exists(moge_config):
        print(f"⬇️ [WorldStereo] Downloading MoGe depth estimator …")
        snapshot_download(
            repo_id="Ruicheng/moge-2-vitl-normal",
            local_dir=moge_dir,
        )
        print(f"✅ [WorldStereo] MoGe cached: {moge_dir}")
    else:
        print(f"✅ [WorldStereo] MoGe cached: {moge_dir}")

    # 4. Patch transformer config.json to point base_model at local dir
    config_path = os.path.join(transformer_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    if cfg.get("base_model") != base_model_dir:
        cfg["base_model"] = base_model_dir
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"✅ [WorldStereo] config.json patched → base_model={base_model_dir}")

    return transformer_dir, base_model_dir, moge_dir
```

- [ ] **Step 4.2: Replace LoadWorldStereoModel stub with real implementation**

Replace the stub class `VNCCS_LoadWorldStereoModel` with:

```python
class VNCCS_LoadWorldStereoModel:
    """Download and load the WorldStereo pipeline + MoGe depth estimator."""

    MODEL_TYPES = ["worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "model_type": (cls.MODEL_TYPES, {
                    "default": "worldstereo-camera",
                    "tooltip": (
                        "worldstereo-camera: 10.9 GB transformer, feasible on 16 GB VRAM with offloading. "
                        "worldstereo-memory: ~22 GB, requires 24+ GB VRAM. "
                        "worldstereo-memory-dmd: 34.9 GB distilled, requires 40+ GB VRAM."
                    ),
                }),
                "precision": (["bf16", "fp8", "fp4"], {
                    "default": "bf16",
                    "tooltip": (
                        "bf16: recommended. "
                        "fp8: transformer weight-only via optimum-quanto (~half bf16 VRAM). "
                        "fp4: transformer weight-only via optimum-quanto (~quarter bf16 VRAM, lower quality)."
                    ),
                }),
                "offload_mode": (["model_cpu_offload", "sequential_cpu_offload", "none"], {
                    "default": "model_cpu_offload",
                    "tooltip": (
                        "model_cpu_offload: move components to CPU between steps. Recommended for 16 GB. "
                        "sequential_cpu_offload: layer-by-layer offload, slower but less VRAM. "
                        "none: all components stay on GPU."
                    ),
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("WORLDSTEREO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/Video"

    def load_model(
        self,
        model_type="worldstereo-camera",
        precision="bf16",
        offload_mode="model_cpu_offload",
        device="cuda",
    ):
        from models.worldstereo_wrapper import WorldStereo
        from moge.model import MoGeModel

        transformer_dir, base_model_dir, moge_dir = _download_worldstereo_components(model_type)

        # ── Load WorldStereo pipeline ─────────────────────────────────────────
        print(f"🔄 [WorldStereo] Loading pipeline (model_type={model_type}, precision={precision}) …")
        worldstereo = WorldStereo.from_pretrained(transformer_dir, device=device)
        pipeline = worldstereo.pipeline

        # ── Apply precision to transformer only ───────────────────────────────
        if precision == "bf16":
            pipeline.transformer.to(torch.bfloat16)
            if hasattr(pipeline, "vae"):
                pipeline.vae.to(torch.bfloat16)
            # T5 and CLIP stay in float32 — they are offloaded to CPU anyway

        elif precision == "fp8":
            try:
                from optimum.quanto import quantize, freeze, qfloat8_e4m3fn
                pipeline.transformer.to(torch.bfloat16)
                quantize(pipeline.transformer, weights=qfloat8_e4m3fn)
                freeze(pipeline.transformer)
                print("✅ [WorldStereo] fp8 weight quantization applied")
            except ImportError:
                raise ImportError(
                    "optimum-quanto required for fp8. Install: pip install optimum-quanto"
                )

        elif precision == "fp4":
            try:
                from optimum.quanto import quantize, freeze, qint4
                pipeline.transformer.to(torch.bfloat16)
                quantize(pipeline.transformer, weights=qint4)
                freeze(pipeline.transformer)
                print("✅ [WorldStereo] fp4 (qint4) weight quantization applied")
            except ImportError:
                raise ImportError(
                    "optimum-quanto required for fp4. Install: pip install optimum-quanto"
                )

        # ── Apply offloading ──────────────────────────────────────────────────
        if device == "cuda":
            if offload_mode == "model_cpu_offload":
                pipeline.enable_model_cpu_offload()
                print("✅ [WorldStereo] model_cpu_offload enabled")
            elif offload_mode == "sequential_cpu_offload":
                pipeline.enable_sequential_cpu_offload()
                print("✅ [WorldStereo] sequential_cpu_offload enabled")

        # ── Load MoGe (stays on CPU until inference) ─────────────────────────
        print("🔄 [WorldStereo] Loading MoGe depth estimator …")
        moge_model = MoGeModel.from_pretrained(moge_dir).eval()
        print("✅ [WorldStereo] MoGe loaded (CPU)")

        print("✅ [WorldStereo] Pipeline ready")
        return ({
            "worldstereo": worldstereo,
            "pipeline":    pipeline,
            "moge":        moge_model,
            "device":      device,
            "model_type":  model_type,
        },)
```

- [ ] **Step 4.3: Verify import doesn't crash**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from nodes.world_stereo import VNCCS_LoadWorldStereoModel
print('INPUT_TYPES:', list(VNCCS_LoadWorldStereoModel.INPUT_TYPES()['optional'].keys()))
"
```

Expected: `INPUT_TYPES: ['model_type', 'precision', 'offload_mode', 'device']`

- [ ] **Step 4.4: Run unit tests (should still pass)**

```bash
python -m pytest tests/test_world_stereo_nodes.py -v
```

Expected: 10 passed (unchanged).

- [ ] **Step 4.5: Commit**

```bash
git add nodes/world_stereo.py
git commit -m "feat: implement VNCCS_LoadWorldStereoModel with download + quantization"
```

---

## Task 5: Implement VNCCS_WorldStereoGenerate

**Files:**
- Modify: `nodes/world_stereo.py` — add preprocessing helper + replace stub

- [ ] **Step 5.1: Add preprocessing helper `_prepare_pipeline_inputs`**

Add this function after `_download_worldstereo_components` in `nodes/world_stereo.py`:

```python
def _prepare_pipeline_inputs(
    image_pil,        # PIL Image, already resized to (width, height)
    c2ws: torch.Tensor,   # [N,4,4] C2W
    intrs: torch.Tensor,  # [N,3,3]
    moge_model,
    device: torch.device,
    width: int,
    height: int,
) -> dict:
    """
    Replicate WorldStereo's load_single_view_data() logic for arbitrary inputs.

    Returns dict ready to unpack as **kwargs into worldstereo.pipeline().
    """
    from src.pointcloud import get_points3d_and_colors, point_rendering
    from models.camera import get_camera_embedding
    import torchvision.transforms as T

    N = c2ws.shape[0]

    # 1. Image tensor [-1, 1] for pipeline conditioning
    img_tensor = T.ToTensor()(image_pil) * 2.0 - 1.0   # [3,H,W]
    img_tensor_device = img_tensor.to(device)

    # 2. MoGe depth (run on GPU for speed, move to CPU after)
    moge_device = torch.device(device)
    moge_model = moge_model.to(moge_device)
    with torch.no_grad():
        depth_output = moge_model.infer(img_tensor_device.unsqueeze(0))
    # MoGe returns dict; key is 'depth' with shape [1,H,W] or [H,W]
    depth_np = depth_output["depth"]
    if isinstance(depth_np, torch.Tensor):
        depth_np = depth_np.squeeze().cpu().numpy()
    moge_model.to("cpu")  # return to CPU
    torch.cuda.empty_cache()

    # 3. W2C matrices
    w2cs = _c2w_to_w2c(c2ws)  # [N,4,4]

    # 4. 3D point cloud from reference view (first camera = identity in scene)
    ref_w2c = w2cs[0:1]  # [1,4,4]
    ref_intr = intrs[0]  # [3,3]
    points3d, colors = get_points3d_and_colors(
        K=ref_intr.numpy(),
        w2cs=ref_w2c.numpy(),
        depth=depth_np,
        image=(img_tensor.permute(1, 2, 0).numpy() + 1.0) / 2.0,  # [H,W,3] in [0,1]
        device=device,
    )

    # 5. Render point cloud from all N camera views
    render_rgbs, render_masks = point_rendering(
        K=intrs.numpy(),      # [N,3,3]
        w2cs=w2cs.numpy(),    # [N,4,4]
        points=points3d,
        colors=colors,
        device=device,
        h=height,
        w=width,
    )
    # render_rgbs: [N,3,H,W] in [0,1] or [-1,1] — check and normalise to [-1,1]
    render_rgbs_t = torch.from_numpy(render_rgbs).float()
    if render_rgbs_t.max() > 1.5:  # already [-1,1]
        pass
    else:
        render_rgbs_t = render_rgbs_t * 2.0 - 1.0

    # Replace first frame with original image for quality reference
    render_rgbs_t[0] = img_tensor

    render_video = render_rgbs_t.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1,3,N,H,W]
    # render_masks: [N,1,H,W]
    render_masks_t = torch.from_numpy(render_masks).float()
    render_mask = render_masks_t.unsqueeze(0).permute(0, 2, 1, 3, 4)   # [1,1,N,H,W]

    render_video = render_video.to(device)
    render_mask  = render_mask.to(device)

    # 6. Camera embedding
    camera_emb = get_camera_embedding(
        intrinsic=intrs.unsqueeze(0).to(device),   # [1,N,3,3]
        extrinsic=w2cs.unsqueeze(0).to(device),    # [1,N,4,4]
        f=N, h=height, w=width,
        normalize=True,
        is_w2c=True,
    )  # [1,6,N,H,W] or [1,N,6,H,W] — verify shape at runtime

    return {
        "image":            image_pil,
        "render_video":     render_video,
        "render_mask":      render_mask,
        "camera_embedding": camera_emb,
        "extrinsics":       w2cs.unsqueeze(0).to(device),    # [1,N,4,4]
        "intrinsics":       intrs.unsqueeze(0).to(device),   # [1,N,3,3]
        "height":           height,
        "width":            width,
        "num_frames":       N,
    }
```

> **Implementation note:** `get_camera_embedding` signature and exact output shape must be verified by running the code. If the function name differs or parameters differ, inspect `worldstereo/models/camera.py` and adjust.
>
> Similarly, `point_rendering` may return `(render_rgbs, render_masks)` or a single dict — verify against `worldstereo/src/pointcloud.py` and adjust unpacking.
>
> `moge_model.infer()` output key may differ — inspect with `print(depth_output.keys())` if needed.

- [ ] **Step 5.2: Replace WorldStereoGenerate stub with real implementation**

Replace the stub class `VNCCS_WorldStereoGenerate` with:

```python
class VNCCS_WorldStereoGenerate:
    """
    WorldStereo video generation from a single image.

    Outputs video_frames + camera_poses + camera_intrinsics for direct
    connection to VNCCS_WorldMirrorV2_3D.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("WORLDSTEREO_MODEL",),
                "image":      ("IMAGE",),           # [1,H,W,3] ComfyUI tensor
                "trajectory": ("CAMERA_TRAJECTORY",),
            },
            "optional": {
                "num_inference_steps": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "0 = auto (4 for dmd, 20 for others).",
                }),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1,
                                  "tooltip": "-1 = random."}),
                "negative_prompt": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("IMAGE",  "TENSOR",         "TENSOR")
    RETURN_NAMES  = ("video_frames", "camera_poses", "camera_intrinsics")
    FUNCTION      = "generate"
    CATEGORY      = "VNCCS/Video"

    def generate(
        self,
        model,
        image,
        trajectory,
        num_inference_steps=0,
        guidance_scale=5.0,
        seed=-1,
        negative_prompt="",
    ):
        pipeline   = model["pipeline"]
        moge_model = model["moge"]
        device     = model["device"]
        model_type = model["model_type"]

        c2ws  = trajectory["c2ws"]    # [N,4,4]
        intrs = trajectory["intrs"]   # [N,3,3]
        W     = trajectory["width"]
        H     = trajectory["height"]
        N     = c2ws.shape[0]

        # ── Auto inference steps ──────────────────────────────────────────────
        if num_inference_steps == 0:
            num_inference_steps = 4 if "dmd" in model_type else 20

        # ── Preprocess: ComfyUI IMAGE [1,H,W,3] → PIL ────────────────────────
        img_np  = (image[0].cpu().numpy()[..., :3] * 255).astype(np.uint8)
        img_pil = PILImage.fromarray(img_np).resize((W, H), PILImage.Resampling.BICUBIC)

        # ── Prepare all pipeline inputs ───────────────────────────────────────
        print(f"🔄 [WorldStereo] Preprocessing: MoGe depth + point rendering …")
        pipeline_inputs = _prepare_pipeline_inputs(
            image_pil=img_pil,
            c2ws=c2ws,
            intrs=intrs,
            moge_model=moge_model,
            device=device,
            width=W,
            height=H,
        )

        # ── Generator ────────────────────────────────────────────────────────
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=device).manual_seed(seed)

        # ── Inference ────────────────────────────────────────────────────────
        print(f"🚀 [WorldStereo] Generating {N} frames ({num_inference_steps} steps) …")
        with torch.autocast(device, dtype=torch.bfloat16):
            output = pipeline(
                **pipeline_inputs,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
            )

        torch.cuda.empty_cache()

        # ── Decode frames: [N,C,H,W] float → ComfyUI [N,H,W,3] float ────────
        frames = output.frames[0].float().cpu()  # [N,3,H,W]
        # Clamp to [0,1] — diffusers may return slightly out-of-range values
        frames = frames.clamp(0.0, 1.0)
        video_frames = frames.permute(0, 2, 3, 1)  # [N,H,W,3]

        print(f"✅ [WorldStereo] Generated {video_frames.shape[0]} frames "
              f"({video_frames.shape[2]}×{video_frames.shape[1]})")

        # ── Camera outputs for WorldMirror V2 ────────────────────────────────
        # WorldMirror V2 expects camera_poses as-is (will use as conditioning)
        # Return C2W [N,4,4] — same format WorldMirror V2 camera_poses input
        camera_poses_out = c2ws.cpu().float()       # [N,4,4]
        camera_intrs_out = intrs.cpu().float()      # [N,3,3]

        return video_frames, camera_poses_out, camera_intrs_out
```

- [ ] **Step 5.3: Verify the file is syntactically valid**

```bash
python -c "import nodes.world_stereo; print('OK')"
```

Expected: `OK`

- [ ] **Step 5.4: Run all tests**

```bash
python -m pytest tests/test_world_stereo_nodes.py -v
```

Expected: 10 passed.

- [ ] **Step 5.5: Commit**

```bash
git add nodes/world_stereo.py
git commit -m "feat: implement VNCCS_WorldStereoGenerate with MoGe + point rendering + pipeline"
```

---

## Task 6: End-to-end smoke test

This task verifies the nodes load correctly in ComfyUI and the trajectory builder works without GPU.

- [ ] **Step 6.1: Verify all 3 nodes appear in node registry**

```bash
python -c "
from nodes import NODE_CLASS_MAPPINGS
ws_nodes = [k for k in NODE_CLASS_MAPPINGS if 'WorldStereo' in k or 'Trajectory' in k]
print('WorldStereo nodes registered:', ws_nodes)
assert len(ws_nodes) == 3, f'Expected 3, got {len(ws_nodes)}'
print('OK')
"
```

Expected:
```
WorldStereo nodes registered: ['VNCCS_LoadWorldStereoModel', 'VNCCS_CameraTrajectoryBuilder', 'VNCCS_WorldStereoGenerate']
OK
```

- [ ] **Step 6.2: Verify trajectory builder runs without GPU**

```bash
python -c "
from nodes.world_stereo import VNCCS_CameraTrajectoryBuilder
node = VNCCS_CameraTrajectoryBuilder()
result = node.build(preset='circular', num_frames=16, radius=1.0)
traj = result[0]
print('c2ws shape:', traj['c2ws'].shape)
print('intrs shape:', traj['intrs'].shape)
assert traj['c2ws'].shape == (16, 4, 4)
assert traj['intrs'].shape == (16, 3, 3)
print('Trajectory builder: OK')
"
```

Expected:
```
c2ws shape: torch.Size([16, 4, 4])
intrs shape: torch.Size([16, 3, 3])
Trajectory builder: OK
```

- [ ] **Step 6.3: Verify INPUT_TYPES on all 3 nodes**

```bash
python -c "
from nodes import NODE_CLASS_MAPPINGS as M
for name in ['VNCCS_LoadWorldStereoModel', 'VNCCS_CameraTrajectoryBuilder', 'VNCCS_WorldStereoGenerate']:
    cls = M[name]
    t = cls.INPUT_TYPES()
    print(f'{name}: inputs OK')
    print(f'  returns: {cls.RETURN_NAMES}')
"
```

Expected: 3 lines of `inputs OK` with return names printed.

- [ ] **Step 6.4: Commit final state**

```bash
git add -A
git commit -m "feat: WorldStereo ComfyUI nodes complete — LoadModel, TrajectoryBuilder, Generate"
```

---

## Implementation Notes

**If `get_c2w` signature differs:** Open `worldstereo/src/camera_utils.py` and read the actual signature. The parameters `nframe`, `r_x`, `r_y`, `z`, `phi`, `theta` may have different names. Adjust `_build_trajectory` calls accordingly.

**If `get_camera_embedding` shape is unexpected:** Print `camera_emb.shape` in `_prepare_pipeline_inputs` and compare to what `KFPCDControllerPipeline.__call__` expects. The pipeline may want `[B, N, 6, H, W]` or `[B, 6, N, H, W]` — rearrange with `.permute()` if needed.

**If `point_rendering` returns depth instead of mask:** The function signature shows `return_depth` flag. Call with `return_depth=False` to get masks, or read the actual return value and handle accordingly.

**If `moge_model.infer()` key is not `depth`:** Print `depth_output.keys()` and use the correct key. MoGe v2 likely returns `{'depth': ..., 'mask': ..., 'normal': ...}`.

**WorldMirror V2 camera_poses format:** If WorldMirror V2 reconstruction looks wrong (flipped/inverted geometry), try passing `_c2w_to_w2c(c2ws)` (W2C) instead of `c2ws` (C2W) as `camera_poses_out`. The existing WorldMirror V2 node code shows: `views["camera_poses"] = camera_poses.unsqueeze(0)` — check what it expects by looking at `worldmirror_v2.py:321`.
