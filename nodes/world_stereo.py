"""
WorldStereo ComfyUI nodes — camera-guided video generation.

Nodes:
  - VNCCS_LoadWorldStereoModel   — load WorldStereo model (Task 4 stub)
  - VNCCS_CameraTrajectoryBuilder — build camera trajectory tensors
  - VNCCS_WorldStereoGenerate    — run WorldStereo inference (Task 5 stub)
"""

import os
import sys
import json
import math
import numpy as np
import torch

# ── nodes/ -> repo root ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── worldstereo camera utils ───────────────────────────────────────────────────
_WORLDSTEREO_PATH = os.path.join(PROJECT_ROOT, "worldstereo")
if _WORLDSTEREO_PATH not in sys.path:
    sys.path.insert(0, _WORLDSTEREO_PATH)

try:
    from src.camera_utils import (
        camera_backward_forward,
        camera_left_right,
        camera_rotation,
        native_camera_rotation,
        interpolate_poses,
    )
    CAMERA_UTILS_AVAILABLE = True
except ImportError:
    CAMERA_UTILS_AVAILABLE = False

# ── pytorch3d (optional, needed for circular preset) ─────────────────────────
try:
    from pytorch3d.renderer.cameras import look_at_rotation
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _build_intrinsics(fov_deg: float, width: int, height: int) -> torch.Tensor:
    """Build a [3, 3] camera intrinsics matrix from field-of-view and image size."""
    fx = fy = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    K = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    return K


def _c2w_to_w2c(c2ws: torch.Tensor) -> torch.Tensor:
    """Batch-invert [N, 4, 4] camera-to-world matrices to world-to-camera."""
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
) -> tuple:
    """
    Build camera trajectory tensors.

    Returns:
        c2ws  : torch.Tensor [N, 4, 4] camera-to-world matrices
        intrs : torch.Tensor [N, 3, 3] intrinsics (same for every frame)
    """
    c2w_start = np.eye(4, dtype=np.float32)

    # ── per-preset trajectory construction ───────────────────────────────────
    if preset == "circular":
        if not PYTORCH3D_AVAILABLE:
            raise RuntimeError(
                "circular preset requires pytorch3d. "
                "Install it or choose a different preset."
            )
        look_at_point = np.array([0, 0, median_depth], dtype=np.float32)
        angles = np.linspace(0, 2 * math.pi, num_frames + 1)[1:]
        rx = radius * median_depth
        ry = radius * median_depth
        c2ws_np = []
        for angle in angles:
            cam_pos = np.array(
                [rx * np.sin(angle), ry * np.cos(angle) - ry, 0],
                dtype=np.float32,
            )
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = cam_pos
            R_new = look_at_rotation(
                cam_pos,
                at=(look_at_point.tolist(),),
                up=((0, 1, 0),),
                device="cpu",
            ).numpy()[0]
            c2w[:3, :3] = R_new
            c2w = c2w_start @ c2w
            c2ws_np.append(c2w)

    elif preset == "forward":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "forward preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        for j in range(1, num_frames + 1):
            c2w = c2w_start.copy()
            c2w = camera_backward_forward(c2w, -speed * j)  # negative = forward
            c2ws_np.append(c2w)

    elif preset == "zoom_in":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "zoom_in preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        for j in range(1, num_frames + 1):
            c2w = c2w_start.copy()
            c2w = camera_backward_forward(c2w, -radius * j / num_frames)
            c2ws_np.append(c2w)

    elif preset == "zoom_out":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "zoom_out preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        for j in range(1, num_frames + 1):
            c2w = c2w_start.copy()
            c2w = camera_backward_forward(c2w, radius * j / num_frames)
            c2ws_np.append(c2w)

    elif preset == "aerial":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "aerial preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        phi_total = math.radians(elevation_deg)
        theta_total = math.radians(elevation_deg * 0.5)  # half elevation for theta
        n_theta = max(1, num_frames // 2)
        n_phi = num_frames - n_theta
        for j in range(1, n_theta + 1):
            theta_j = theta_total * j / n_theta
            c2w = camera_rotation(c2w_start.copy(), median_depth, 0, theta_j)
            c2ws_np.append(c2w)
        c2w_mid = c2ws_np[-1].copy() if c2ws_np else c2w_start.copy()
        for j in range(1, n_phi + 1):
            phi_j = phi_total * j / n_phi
            c2w = camera_rotation(c2w_mid.copy(), median_depth, phi_j, 0)
            c2ws_np.append(c2w)

    elif preset == "custom":
        data = json.loads(custom_json)
        c2ws_np = [np.array(m, dtype=np.float32) for m in data]
        if len(c2ws_np) == 0:
            raise ValueError("custom_json contains no matrices.")
        for i, mat in enumerate(c2ws_np):
            if mat.shape != (4, 4):
                raise ValueError(f"custom_json matrix {i} has shape {mat.shape}, expected (4, 4).")

    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    # ── convert to torch ──────────────────────────────────────────────────────
    c2ws = torch.from_numpy(np.stack(c2ws_np)).float()  # [N, 4, 4]

    # ── build intrinsics (broadcast to all frames) ────────────────────────────
    K = _build_intrinsics(fov_deg, width, height)                # [3, 3]
    intrs = K.unsqueeze(0).expand(c2ws.shape[0], -1, -1).clone()  # [N, 3, 3]

    return c2ws, intrs


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_CameraTrajectoryBuilder
# ─────────────────────────────────────────────────────────────────────────────

class VNCCS_CameraTrajectoryBuilder:
    PRESETS = ["circular", "forward", "zoom_in", "zoom_out", "aerial", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "preset": (cls.PRESETS, {"default": "circular"}),
                "num_frames": ("INT", {"default": 25, "min": 4, "max": 81}),
                "radius": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Orbit radius (circular) or travel distance (zoom).",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Per-frame translation for forward preset.",
                    },
                ),
                "elevation_deg": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": -90.0,
                        "max": 90.0,
                        "step": 1.0,
                        "tooltip": "Camera elevation for aerial preset.",
                    },
                ),
                "median_depth": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Estimated scene depth — orbit center distance.",
                    },
                ),
                "fov_deg": (
                    "FLOAT",
                    {"default": 70.0, "min": 10.0, "max": 150.0, "step": 1.0},
                ),
                "image_width": (
                    "INT",
                    {"default": 768, "min": 64, "max": 2048, "step": 64},
                ),
                "image_height": (
                    "INT",
                    {"default": 480, "min": 64, "max": 2048, "step": 64},
                ),
                "custom_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "JSON list of N 4x4 C2W matrices. Used when preset=custom.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CAMERA_TRAJECTORY",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "build"
    CATEGORY = "VNCCS/Video"

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
            preset,
            num_frames,
            radius,
            speed,
            elevation_deg,
            fov_deg,
            image_width,
            image_height,
            median_depth,
            custom_json,
        )
        trajectory = {
            "c2ws": c2ws,
            "intrs": intrs,
            "width": image_width,
            "height": image_height,
        }
        print(
            f"[Trajectory] preset={preset}, frames={c2ws.shape[0]}, "
            f"size={image_width}x{image_height}"
        )
        return (trajectory,)


# ─────────────────────────────────────────────────────────────────────────────
# Stub nodes (Tasks 4 and 5 will replace these)
# ─────────────────────────────────────────────────────────────────────────────

class VNCCS_LoadWorldStereoModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {}}

    RETURN_TYPES = ("WORLDSTEREO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/Video"

    def load_model(self):
        raise NotImplementedError("Task 4 not yet implemented")


class VNCCS_WorldStereoGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("video_frames", "camera_poses", "camera_intrinsics")
    FUNCTION = "generate"
    CATEGORY = "VNCCS/Video"

    def generate(self):
        raise NotImplementedError("Task 5 not yet implemented")


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    VNCCS_LoadWorldStereoModel,
    "VNCCS_CameraTrajectoryBuilder": VNCCS_CameraTrajectoryBuilder,
    "VNCCS_WorldStereoGenerate":     VNCCS_WorldStereoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldStereoModel":    "Load WorldStereo Model",
    "VNCCS_CameraTrajectoryBuilder": "Camera Trajectory Builder",
    "VNCCS_WorldStereoGenerate":     "WorldStereo Generate",
}
