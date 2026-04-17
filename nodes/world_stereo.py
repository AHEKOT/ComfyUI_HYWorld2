"""
WorldStereo ComfyUI nodes — camera-guided video generation.

Nodes:
  - VNCCS_LoadWorldStereoModel   — download and load WorldStereo + MoGe models
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

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

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


def _get_models_base() -> str:
    return (
        folder_paths.models_dir if FOLDER_PATHS_AVAILABLE
        else os.path.join(PROJECT_ROOT, "models")
    )


def _download_worldstereo_components(model_type: str) -> tuple:
    """
    Download all required model components. Returns (transformer_dir, base_model_dir, moge_dir).
    """
    from huggingface_hub import snapshot_download

    base = _get_models_base()

    # 1. WorldStereo transformer weights
    transformer_dir = os.path.join(base, "WorldStereo", model_type)
    transformer_weights = os.path.join(transformer_dir, "model.safetensors")
    if not os.path.exists(transformer_weights):
        print(f"[WorldStereo] Downloading transformer ({model_type}) ...")
        tmp_dir = os.path.join(base, "WorldStereo", "_tmp")
        snapshot_download(
            repo_id="hanshanxude/WorldStereo",
            allow_patterns=[f"{model_type}/**"],
            local_dir=tmp_dir,
        )
        nested = os.path.join(tmp_dir, model_type)
        if os.path.isdir(nested):
            import shutil
            os.makedirs(transformer_dir, exist_ok=True)
            for f in os.listdir(nested):
                shutil.move(os.path.join(nested, f), transformer_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[WorldStereo] Transformer cached: {transformer_dir}")
    else:
        print(f"[WorldStereo] Transformer cached: {transformer_dir}")

    # 2. Wan2.1 base model (VAE, T5, CLIP)
    base_model_dir = os.path.join(base, "Wan2.1-I2V-14B-480P")
    wan_vae = os.path.join(base_model_dir, "vae", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(wan_vae):
        print(f"[WorldStereo] Downloading Wan2.1-I2V-14B-480P base model (~40 GB) ...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            local_dir=base_model_dir,
        )
        print(f"[WorldStereo] Base model cached: {base_model_dir}")
    else:
        print(f"[WorldStereo] Base model cached: {base_model_dir}")

    # 3. MoGe depth estimator
    moge_dir = os.path.join(base, "MoGe")
    moge_config = os.path.join(moge_dir, "config.json")
    if not os.path.exists(moge_config):
        print(f"[WorldStereo] Downloading MoGe depth estimator ...")
        snapshot_download(
            repo_id="Ruicheng/moge-2-vitl-normal",
            local_dir=moge_dir,
        )
        print(f"[WorldStereo] MoGe cached: {moge_dir}")
    else:
        print(f"[WorldStereo] MoGe cached: {moge_dir}")

    # 4. Patch transformer config.json to use local base_model path
    config_path = os.path.join(transformer_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as fh:
            cfg = json.load(fh)
        if cfg.get("base_model") != base_model_dir:
            cfg["base_model"] = base_model_dir
            with open(config_path, "w") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"[WorldStereo] config.json patched -> base_model={base_model_dir}")

    return transformer_dir, base_model_dir, moge_dir


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
                        "fp8: transformer weight-only via optimum-quanto. "
                        "fp4: transformer weight-only via optimum-quanto (lower quality)."
                    ),
                }),
                "offload_mode": (["model_cpu_offload", "sequential_cpu_offload", "none"], {
                    "default": "model_cpu_offload",
                    "tooltip": (
                        "model_cpu_offload: move components to CPU between steps. Recommended for 16 GB VRAM. "
                        "sequential_cpu_offload: layer-by-layer, slower but less VRAM. "
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

        transformer_dir, _base_model_dir, moge_dir = _download_worldstereo_components(model_type)

        # ── Load WorldStereo pipeline ─────────────────────────────────────────
        print(f"[WorldStereo] Loading pipeline (model_type={model_type}, precision={precision}) ...")
        worldstereo = WorldStereo.from_pretrained(transformer_dir, device=device)
        pipeline = worldstereo.pipeline

        # ── Apply precision to transformer ────────────────────────────────────
        if precision == "bf16":
            pipeline.transformer.to(torch.bfloat16)
            if hasattr(pipeline, "vae"):
                pipeline.vae.to(torch.bfloat16)

        elif precision == "fp8":
            try:
                from optimum.quanto import quantize, freeze, qfloat8_e4m3fn
                pipeline.transformer.to(torch.bfloat16)
                quantize(pipeline.transformer, weights=qfloat8_e4m3fn)
                freeze(pipeline.transformer)
                print("[WorldStereo] fp8 weight quantization applied")
            except ImportError:
                raise ImportError("optimum-quanto required for fp8: pip install optimum-quanto")

        elif precision == "fp4":
            try:
                from optimum.quanto import quantize, freeze, qint4
                pipeline.transformer.to(torch.bfloat16)
                quantize(pipeline.transformer, weights=qint4)
                freeze(pipeline.transformer)
                print("[WorldStereo] fp4 (qint4) weight quantization applied")
            except ImportError:
                raise ImportError("optimum-quanto required for fp4: pip install optimum-quanto")

        # ── Apply offloading ──────────────────────────────────────────────────
        if device == "cuda":
            if offload_mode == "model_cpu_offload":
                pipeline.enable_model_cpu_offload()
                print("[WorldStereo] model_cpu_offload enabled")
            elif offload_mode == "sequential_cpu_offload":
                pipeline.enable_sequential_cpu_offload()
                print("[WorldStereo] sequential_cpu_offload enabled")

        # ── Load MoGe on CPU ─────────────────────────────────────────────────
        print("[WorldStereo] Loading MoGe depth estimator ...")
        moge_model = MoGeModel.from_pretrained(moge_dir).eval()
        print("[WorldStereo] MoGe loaded (CPU)")

        print("[WorldStereo] Pipeline ready")
        return ({
            "worldstereo": worldstereo,
            "pipeline":    pipeline,
            "moge":        moge_model,
            "device":      device,
            "model_type":  model_type,
        },)


def _prepare_pipeline_inputs(
    image_pil,
    c2ws: torch.Tensor,
    intrs: torch.Tensor,
    moge_model,
    device: str,
    width: int,
    height: int,
) -> dict:
    """
    Build render_video, render_mask, camera_embedding from a single image + trajectory.
    Replicates WorldStereo's load_single_view_data() for arbitrary inputs.
    """
    import torchvision.transforms as T
    from src.pointcloud import get_points3d_and_colors, point_rendering
    from models.camera import get_camera_embedding

    N = c2ws.shape[0]

    # 1. Image tensor in [-1, 1] for pipeline
    img_tensor = T.ToTensor()(image_pil) * 2.0 - 1.0   # [3, H, W], range [-1, 1]
    img_np_01 = (img_tensor.permute(1, 2, 0).numpy() + 1.0) / 2.0  # [H, W, 3] in [0, 1]

    # 2. Depth via MoGe
    torch_device = torch.device(device)
    moge_model = moge_model.to(torch_device)
    with torch.no_grad():
        depth_output = moge_model.infer(
            img_tensor.unsqueeze(0).to(torch_device)
        )
    # MoGe returns dict; extract depth as numpy [H, W]
    depth_raw = depth_output["depth"]
    if isinstance(depth_raw, torch.Tensor):
        depth_np = depth_raw.squeeze().cpu().numpy()
    else:
        depth_np = np.squeeze(depth_raw)
    moge_model.to("cpu")
    torch.cuda.empty_cache()

    # 3. W2C matrices (numpy for pointcloud functions)
    w2cs_np = _c2w_to_w2c(c2ws).numpy()  # [N, 4, 4]
    intrs_np = intrs.numpy()              # [N, 3, 3]
    ref_w2c = w2cs_np[0:1]               # [1, 4, 4] reference view (identity scene)
    ref_K   = intrs_np[0]                # [3, 3]

    # 4. 3D point cloud from reference view
    points3d, colors = get_points3d_and_colors(
        K=ref_K,
        w2cs=ref_w2c,
        depth=depth_np,
        image=img_np_01,
        device=device,
    )

    # 5. Render point cloud from all N target views
    render_result = point_rendering(
        K=intrs_np,
        w2cs=w2cs_np,
        points=points3d,
        colors=colors,
        device=device,
        h=height,
        w=width,
    )
    # point_rendering returns (render_rgbs, render_masks) OR (render_rgbs, render_masks, render_depth)
    if isinstance(render_result, (tuple, list)):
        render_rgbs_raw, render_masks_raw = render_result[0], render_result[1]
    else:
        raise RuntimeError(f"Unexpected point_rendering return type: {type(render_result)}")

    # render_rgbs_raw: [N, 3, H, W] or [N, H, W, 3] — normalise to [-1, 1] tensor
    if isinstance(render_rgbs_raw, np.ndarray):
        render_rgbs_t = torch.from_numpy(render_rgbs_raw).float()
    else:
        render_rgbs_t = render_rgbs_raw.float()

    if render_rgbs_t.dim() == 4 and render_rgbs_t.shape[-1] == 3:
        render_rgbs_t = render_rgbs_t.permute(0, 3, 1, 2)  # [N, H, W, 3] → [N, 3, H, W]

    if render_rgbs_t.max() <= 1.5:   # [0, 1] range → convert to [-1, 1]
        render_rgbs_t = render_rgbs_t * 2.0 - 1.0
    render_rgbs_t[0] = img_tensor    # first frame = original image

    # render_masks_raw: [N, 1, H, W] or [N, H, W, 1] or [N, H, W]
    if isinstance(render_masks_raw, np.ndarray):
        render_masks_t = torch.from_numpy(render_masks_raw).float()
    else:
        render_masks_t = render_masks_raw.float()

    if render_masks_t.dim() == 3:
        render_masks_t = render_masks_t.unsqueeze(1)  # [N, H, W] → [N, 1, H, W]
    elif render_masks_t.dim() == 4 and render_masks_t.shape[-1] == 1:
        render_masks_t = render_masks_t.permute(0, 3, 1, 2)  # [N, H, W, 1] → [N, 1, H, W]

    # Reshape to [1, C, N, H, W] (batch=1)
    render_video = render_rgbs_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device)   # [1, 3, N, H, W]
    render_mask  = render_masks_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device)  # [1, 1, N, H, W]

    # 6. Camera embedding [1, 6, N, H, W]
    camera_emb = get_camera_embedding(
        intrinsic=intrs.to(torch_device),  # [N, 3, 3]
        extrinsic=c2ws.to(torch_device),   # [N, 4, 4] C2W (is_w2c=False)
        f=N, h=height, w=width,
        normalize=True,
        is_w2c=False,
    )

    return {
        "image":            image_pil,
        "render_video":     render_video,
        "render_mask":      render_mask,
        "camera_embedding": camera_emb,
        "extrinsics":       torch.from_numpy(w2cs_np).float().to(torch_device),  # [N, 4, 4] W2C
        "intrinsics":       intrs.to(torch_device),   # [N, 3, 3]
        "height":           height,
        "width":            width,
        "num_frames":       N,
    }


class VNCCS_WorldStereoGenerate:
    """
    WorldStereo camera-guided video generation from a single image.
    Outputs video_frames + camera_poses + camera_intrinsics for VNCCS_WorldMirrorV2_3D.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("WORLDSTEREO_MODEL",),
                "image":      ("IMAGE",),
                "trajectory": ("CAMERA_TRAJECTORY",),
            },
            "optional": {
                "num_inference_steps": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "0 = auto (4 for memory-dmd, 20 for others).",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1,
                    "tooltip": "-1 = random.",
                }),
                "negative_prompt": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("IMAGE",         "TENSOR",         "TENSOR")
    RETURN_NAMES  = ("video_frames",  "camera_poses",   "camera_intrinsics")
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

        c2ws  = trajectory["c2ws"]    # [N, 4, 4]
        intrs = trajectory["intrs"]   # [N, 3, 3]
        W     = trajectory["width"]
        H     = trajectory["height"]
        N     = c2ws.shape[0]

        if num_inference_steps == 0:
            num_inference_steps = 4 if "dmd" in model_type else 20

        # ── Preprocess: ComfyUI IMAGE [1, H, W, 3] → PIL ─────────────────────
        img_np  = (image[0].cpu().numpy()[..., :3] * 255).astype(np.uint8)
        img_pil = PILImage.fromarray(img_np).resize((W, H), PILImage.Resampling.BICUBIC)

        # ── Build pipeline inputs ─────────────────────────────────────────────
        print(f"[WorldStereo] Preprocessing: depth estimation + point rendering ...")
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
        print(f"[WorldStereo] Generating {N} frames ({num_inference_steps} steps, guidance={guidance_scale}) ...")
        with torch.autocast(device, dtype=torch.bfloat16):
            output = pipeline(
                **pipeline_inputs,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
            )

        torch.cuda.empty_cache()

        # ── Decode: [N, C, H, W] float → ComfyUI IMAGE [N, H, W, 3] ─────────
        frames = output.frames[0].float().cpu().clamp(0.0, 1.0)  # [N, 3, H, W]
        video_frames = frames.permute(0, 2, 3, 1)                # [N, H, W, 3]

        print(f"[WorldStereo] Done: {video_frames.shape[0]} frames @ {W}x{H}")

        # ── Camera outputs (for WorldMirror V2) ──────────────────────────────
        camera_poses_out = c2ws.cpu().float()   # [N, 4, 4]
        camera_intrs_out = intrs.cpu().float()  # [N, 3, 3]

        return video_frames, camera_poses_out, camera_intrs_out


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
