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
import hashlib
import importlib
from contextlib import contextmanager
import numpy as np
import torch

# Get the absolute path of ComfyUI-HYWorld2 and its worldstereo subdirectory
node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
worldstereo_dir = os.path.join(node_dir, "worldstereo")

# Inject the paths to the front of Python's search path to allow importing camera_utils
for path in [node_dir, worldstereo_dir, os.path.join(worldstereo_dir, "src")]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
# ─── PATH RESOLUTION PATCH END ───

# ── nodes/ -> repo root ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── worldstereo camera utils ───────────────────────────────────────────────────
_WORLDSTEREO_PATH = os.path.join(PROJECT_ROOT, "worldstereo")
if _WORLDSTEREO_PATH not in sys.path:
    sys.path.insert(0, _WORLDSTEREO_PATH)


def _prepend_package_search_path(package_name: str, search_path: str):
    """Allow WorldStereo's local namespace packages to coexist with ComfyUI modules."""
    if not os.path.isdir(search_path):
        return

    module = sys.modules.get(package_name)
    if module is None:
        import types
        module = types.ModuleType(package_name)
        sys.modules[package_name] = module

    search_path = os.path.abspath(search_path)
    existing_paths = list(getattr(module, "__path__", []) or [])
    normalized = {os.path.normcase(os.path.abspath(p)) for p in existing_paths}
    if os.path.normcase(search_path) not in normalized:
        existing_paths.insert(0, search_path)

    module.__path__ = existing_paths
    module.__package__ = package_name
    spec = getattr(module, "__spec__", None)
    if spec is not None:
        spec.submodule_search_locations = existing_paths


def _import_worldstereo_class(base_dir: str = PROJECT_ROOT):
    worldstereo_path = _prepare_worldstereo_import_paths(base_dir)
    try:
        from models.worldstereo_wrapper import WorldStereo
        return WorldStereo
    except ModuleNotFoundError as e:
        if e.name not in ("models.worldstereo_wrapper", "models"):
            raise

    wrapper_path = os.path.join(worldstereo_path, "models", "worldstereo_wrapper.py")
    if not os.path.exists(wrapper_path):
        raise FileNotFoundError(f"worldstereo_wrapper.py not found at {wrapper_path!r}")

    spec = importlib.util.spec_from_file_location("models.worldstereo_wrapper", wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {wrapper_path!r}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["models.worldstereo_wrapper"] = module
    spec.loader.exec_module(module)
    return module.WorldStereo


def _prepare_worldstereo_import_paths(base_dir: str = PROJECT_ROOT) -> str:
    worldstereo_path = None
    for name in os.listdir(base_dir):
        if name.lower() == "worldstereo":
            worldstereo_path = os.path.join(base_dir, name)
            break
    if worldstereo_path is None:
        worldstereo_path = os.path.join(base_dir, "worldstereo")

    world_models_path = os.path.join(worldstereo_path, "models")
    world_src_path = os.path.join(worldstereo_path, "src")

    for path in (base_dir, worldstereo_path, world_src_path):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    _prepend_package_search_path("models", world_models_path)
    _prepend_package_search_path("src", world_src_path)
    importlib.invalidate_caches()
    return worldstereo_path


_prepare_worldstereo_import_paths()

def _node_output_first(output):
    result = getattr(output, "result", output)
    if isinstance(result, (tuple, list)):
        if not result:
            raise RuntimeError("SeedVR2 node returned an empty result.")
        return result[0]
    return result


def _load_seedvr2_node_classes():
    class_names = ("SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler")

    try:
        import nodes as comfy_nodes
        mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
        classes = tuple(mappings.get(name) for name in class_names)
        if all(cls is not None for cls in classes):
            return classes
    except Exception:
        pass

    loaded = {}
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for name in class_names:
            if name not in loaded and hasattr(module, name):
                loaded[name] = getattr(module, name)
        if all(name in loaded for name in class_names):
            return tuple(loaded[name] for name in class_names)

    raise RuntimeError(
        "SeedVR2 classes are not registered in the running ComfyUI process. "
        "Install/enable SeedVR2 Video Upscaler and restart ComfyUI so these node classes are loaded: "
        + ", ".join(class_names)
    )


def _seedvr2_upscale_frames(
    frames: torch.Tensor,
    out_width: int,
    out_height: int,
    *,
    seed: int = 42,
    resolution: int = 0,
    max_resolution: int = 0,
    dit_model: str = "seedvr2_ema_3b-Q4_K_M.gguf",
    vae_model: str = "ema_vae_fp16.safetensors",
    device: str = "cuda:0",
    offload_device: str = "cpu",
    batch_size: int = 1,
    color_correction: str = "lab",
):
    if frames.shape[0] == 0:
        return frames
    if frames.shape[1] == out_height and frames.shape[2] == out_width:
        return frames.contiguous()

    SeedVR2LoadDiTModel, SeedVR2LoadVAEModel, SeedVR2VideoUpscaler = _load_seedvr2_node_classes()
    target_short_edge = int(resolution) if int(resolution) > 0 else min(int(out_width), int(out_height))
    target_max_edge = int(max_resolution) if int(max_resolution) > 0 else max(int(out_width), int(out_height))

    print(
        "[WorldStereo] SeedVR2 upscaling generated frames: "
        f"{frames.shape[2]}x{frames.shape[1]} -> {out_width}x{out_height}, frames={frames.shape[0]}"
    )
    try:
        dit = _node_output_first(SeedVR2LoadDiTModel.execute(
            model=dit_model,
            device=device,
            blocks_to_swap=0,
            swap_io_components=False,
            offload_device=offload_device,
            cache_model=True,
            attention_mode="sdpa",
            torch_compile_args=None,
        ))
        vae = _node_output_first(SeedVR2LoadVAEModel.execute(
            model=vae_model,
            device=device,
            encode_tiled=True,
            encode_tile_size=1024,
            encode_tile_overlap=128,
            decode_tiled=True,
            decode_tile_size=1024,
            decode_tile_overlap=128,
            tile_debug="false",
            offload_device=offload_device,
            cache_model=False,
            torch_compile_args=None,
        ))
    except Exception:
        raise

    if isinstance(dit, dict):
        dit = dict(dit)
        dit["node_id"] = f"worldstereo_seedvr2_dit_{dit_model}"
    if isinstance(vae, dict):
        vae = dict(vae)
        vae["node_id"] = f"worldstereo_seedvr2_vae_{vae_model}"

    output = SeedVR2VideoUpscaler.execute(
        image=frames.detach().cpu().float().clamp(0.0, 1.0).contiguous(),
        dit=dit,
        vae=vae,
        seed=int(seed),
        resolution=target_short_edge,
        max_resolution=target_max_edge,
        batch_size=max(1, int(batch_size)),
        uniform_batch_size=False,
        temporal_overlap=0,
        prepend_frames=0,
        color_correction=color_correction,
        input_noise_scale=0.0,
        latent_noise_scale=0.0,
        offload_device=offload_device,
        enable_debug=False,
    )
    upscaled = _node_output_first(output)
    if not isinstance(upscaled, torch.Tensor):
        raise RuntimeError(f"SeedVR2 returned {type(upscaled).__name__}, expected torch.Tensor.")
    return upscaled.float().cpu().clamp(0.0, 1.0).contiguous()


def _fallback_camera_backward_forward(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([0, 0, distance, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def _fallback_camera_left_right(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([distance, 0, 0, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def _camera_up_down(c2w, distance):
    c2w[:3, 3:4] = (c2w @ np.array([0, distance, 0, 1.0], dtype=np.float32).reshape(4, 1))[:3]
    return c2w


def _fallback_native_camera_rotation(c2w, medium_depth, phi, theta):
    R_elevation = np.array([[1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1]], dtype=np.float32)
    R_azimuth = np.array([[np.cos(phi), 0, np.sin(phi), 0],
                          [0, 1, 0, 0],
                          [-np.sin(phi), 0, np.cos(phi), 0],
                          [0, 0, 0, 1]], dtype=np.float32)
    dummy_c2w = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -medium_depth],
                          [0, 0, 0, 1]], dtype=np.float32)
    dummy_c2w = R_azimuth @ R_elevation @ dummy_c2w
    dummy_c2w[:3, 3] += np.array([0, 0, medium_depth], dtype=np.float32)
    return c2w @ dummy_c2w


def _fallback_axis_angle_to_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float32)


def _fallback_camera_rotation(c2w, medium_depth, phi, theta):
    z0 = c2w[2, 3]
    if z0 != 0 and phi != 0:
        axis_origin = c2w @ np.array([0, 0, medium_depth, 1], dtype=np.float32).reshape(4, 1)
        axis_origin = axis_origin[:3, 0]
        axis_origin[2] = 0
        R = np.eye(4, dtype=np.float32)
        R[:3, :3] = _fallback_axis_angle_to_matrix(np.array([0, 0, 1], dtype=np.float32), -phi)
        T1 = np.eye(4, dtype=np.float32)
        T2 = np.eye(4, dtype=np.float32)
        T1[:3, 3] = -axis_origin
        T2[:3, 3] = axis_origin
        return T2 @ R @ T1 @ c2w
    return _fallback_native_camera_rotation(c2w, medium_depth, phi, theta)


def _fallback_interpolate_poses(poses, M):
    if poses.shape[0] == M:
        return poses
    indices = np.linspace(0, poses.shape[0] - 1, M)
    nearest = np.clip(np.round(indices).astype(int), 0, poses.shape[0] - 1)
    return poses[nearest]


try:
    # 1. Try importing via the default package path
    from src.camera_utils import (
        camera_backward_forward,
        camera_left_right,
        camera_rotation,
        native_camera_rotation,
        interpolate_poses,
    )
    CAMERA_UTILS_AVAILABLE = True
except Exception as first_import_error:
    try:
        # 2. Fallback: Import directly to bypass the contested "src" namespace
        from camera_utils import (
            camera_backward_forward,
            camera_left_right,
            camera_rotation,
            native_camera_rotation,
            interpolate_poses,
        )
        CAMERA_UTILS_AVAILABLE = True
    except Exception as e:
        print("\n" + "="*80)
        print(f"[WorldStereo DEBUG] src.camera_utils import error: {first_import_error}")
        print(f"[WorldStereo DEBUG] Real camera_utils import error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        camera_backward_forward = _fallback_camera_backward_forward
        camera_left_right = _fallback_camera_left_right
        camera_rotation = _fallback_camera_rotation
        native_camera_rotation = _fallback_native_camera_rotation
        interpolate_poses = _fallback_interpolate_poses
        CAMERA_UTILS_AVAILABLE = True
        CAMERA_UTILS_FALLBACK = True
else:
    CAMERA_UTILS_FALLBACK = False

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    folder_paths = None
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


@contextmanager
def _temporary_worldstereo_runtime_patches(patch_diffusers=False):
    """Limit WorldStereo compatibility monkey-patches to the code block that needs them."""
    import torch.distributed as dist

    env_keys = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    original_env = {key: os.environ.get(key) for key in env_keys}
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    class DummyProcessGroup:
        def __init__(self):
            self.group_name = "dummy_group"

        def __getattr__(self, name):
            def _fallback_func(*args, **kwargs):
                return args[0] if args else None
            return _fallback_func

    def _dummy_dist_func(*args, **kwargs):
        return args[0] if args else None

    def _dummy_all_gather(tensor_list, tensor, *args, **kwargs):
        if tensor_list:
            tensor_list[0].copy_(tensor)
        return tensor_list

    _dummy_pg = DummyProcessGroup()
    attr_patchers = {
        "is_initialized": lambda: True,
        "get_rank": lambda *args, **kwargs: 0,
        "get_world_size": lambda *args, **kwargs: 1,
        "barrier": lambda *args, **kwargs: None,
        "new_group": lambda *args, **kwargs: _dummy_pg,
        "all_reduce": _dummy_dist_func,
        "broadcast": _dummy_dist_func,
        "all_gather": _dummy_all_gather,
        "reduce_scatter": lambda output, input_list, *args, **kwargs: output,
        "get_backend": lambda *args, **kwargs: "gloo",
        "init_process_group": lambda *args, **kwargs: None,
    }

    patched_attrs = []
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == "torch.distributed" or mod_name.startswith("torch.distributed."):
            if mod is None:
                continue
            for attr, patched in attr_patchers.items():
                if hasattr(mod, attr):
                    patched_attrs.append((mod, attr, getattr(mod, attr)))
                    try:
                        setattr(mod, attr, patched)
                    except Exception:
                        patched_attrs.pop()

    if dist not in [item[0] for item in patched_attrs]:
        for attr, patched in attr_patchers.items():
            if hasattr(dist, attr):
                patched_attrs.append((dist, attr, getattr(dist, attr)))
                setattr(dist, attr, patched)

    original_signature_keys = None
    pipeline_utils = None
    if patch_diffusers:
        import diffusers.pipelines.pipeline_utils as pipeline_utils
        original_signature_keys = pipeline_utils.DiffusionPipeline._get_signature_keys

        @staticmethod
        def patched_get_signature_keys(obj):
            expected_modules, optional_parameters = original_signature_keys(obj)

            if isinstance(expected_modules, list):
                expected_modules = [x for x in expected_modules if x != "device"]
            elif isinstance(expected_modules, set):
                expected_modules = expected_modules - {"device"}

            if isinstance(optional_parameters, list):
                if "device" not in optional_parameters:
                    optional_parameters.append("device")
            elif isinstance(optional_parameters, set):
                optional_parameters = optional_parameters | {"device"}

            return expected_modules, optional_parameters

        pipeline_utils.DiffusionPipeline._get_signature_keys = patched_get_signature_keys

    try:
        yield
    finally:
        if pipeline_utils is not None:
            pipeline_utils.DiffusionPipeline._get_signature_keys = original_signature_keys

        for mod, attr, original in reversed(patched_attrs):
            try:
                setattr(mod, attr, original)
            except Exception:
                pass

        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def _fallback_points_padding(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def _fallback_get_points3d_and_colors(K, w2cs, depth, image, device, contract=8.0):
    _, _, h, w = image.shape
    torch_device = torch.device(device)
    K = K.to(torch_device).float()
    w2cs = w2cs.to(torch_device).float()
    depth = depth.to(torch_device).float()
    image = image.to(torch_device).float()
    if depth.shape[1] == 3:
        depth = depth[:, 0:1]
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    valid_depth = depth[depth > 0]
    if valid_depth.numel() == 0:
        raise RuntimeError("Invalid depth map for fallback point rendering: no positive valid depth values.")

    mid_depth = torch.median(valid_depth.reshape(-1), dim=0)[0] * contract
    depth = depth.clone()
    far = depth > mid_depth
    depth[far] = (2 * mid_depth) - (mid_depth ** 2 / (depth[far] + 1e-6))

    ys, xs = torch.meshgrid(
        torch.arange(h, device=torch_device, dtype=torch.float32) + 0.5,
        torch.arange(w, device=torch_device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(h * w, device=torch_device)], dim=1)
    point_depth = depth[0, 0].reshape(-1, 1)
    cam_points = (torch.linalg.inv(K[0]) @ pix.T).T * point_depth
    c2w = torch.linalg.inv(w2cs[0])
    points3d = (c2w @ _fallback_points_padding(cam_points).T).T[:, :3]
    colors = image[0].permute(1, 2, 0).reshape(-1, 3)
    valid = point_depth.reshape(-1) > 0
    return points3d[valid], colors[valid]


def _fallback_point_rendering(K, w2cs, points, colors, device, h, w):
    torch_device = torch.device(device)
    K = K.to(torch_device).float()
    w2cs = w2cs.to(torch_device).float()
    points = points.to(torch_device).float()
    colors = colors.to(torch_device).float()
    nframe = w2cs.shape[0]
    points_h = _fallback_points_padding(points)
    render_rgbs = torch.zeros((nframe, 3, h, w), device=torch_device, dtype=torch.float32)
    render_masks = torch.ones((nframe, 1, h, w), device=torch_device, dtype=torch.float32)
    flat_size = h * w

    for frame_idx in range(nframe):
        cam = (w2cs[frame_idx] @ points_h.T).T[:, :3]
        z = cam[:, 2]
        valid = z > 1e-6
        if not valid.any():
            continue
        cam = cam[valid]
        z = z[valid]
        frame_colors = colors[valid]
        fx, fy = K[frame_idx, 0, 0], K[frame_idx, 1, 1]
        cx, cy = K[frame_idx, 0, 2], K[frame_idx, 1, 2]
        xs = torch.round((cam[:, 0] * fx / z) + cx).long()
        ys = torch.round((cam[:, 1] * fy / z) + cy).long()
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not inside.any():
            continue
        xs, ys, z, frame_colors = xs[inside], ys[inside], z[inside], frame_colors[inside]
        flat_idx = ys * w + xs
        if hasattr(torch.Tensor, "scatter_reduce_"):
            zbuf = torch.full((flat_size,), float("inf"), device=torch_device, dtype=torch.float32)
            zbuf.scatter_reduce_(0, flat_idx, z, reduce="amin", include_self=True)
            visible = z <= (zbuf[flat_idx] + 1e-5)
        else:
            order = torch.argsort(z, descending=True)
            flat_idx, frame_colors = flat_idx[order], frame_colors[order]
            visible = torch.ones_like(flat_idx, dtype=torch.bool)
        flat_img = render_rgbs[frame_idx].permute(1, 2, 0).reshape(flat_size, 3)
        flat_mask = render_masks[frame_idx, 0].reshape(flat_size)
        flat_img[flat_idx[visible]] = frame_colors[visible]
        flat_mask[flat_idx[visible]] = 0.0

    return render_rgbs, render_masks


def _opencv_look_at_c2w(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = target - cam_pos
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-8)

    world_down = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(world_down, z_axis))) > 0.99:
        world_down = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    x_axis = np.cross(world_down, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = cam_pos
    return c2w


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
            c2w = camera_backward_forward(c2w, speed * j)
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
            c2w = camera_backward_forward(c2w, radius * j / num_frames)
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
            c2w = camera_backward_forward(c2w, -radius * j / num_frames)
            c2ws_np.append(c2w)

    elif preset == "stereo_orbit":
        c2ws_np = []
        target = np.array([0.0, 0.0, median_depth], dtype=np.float32)
        orbit_radius = max(float(speed), 1e-4)
        angles = np.linspace(0.0, 2.0 * math.pi, num_frames, endpoint=False, dtype=np.float32)
        for angle in angles:
            cam_pos = np.array(
                [orbit_radius * math.cos(float(angle)), orbit_radius * math.sin(float(angle)), 0.0],
                dtype=np.float32,
            )
            c2ws_np.append(_opencv_look_at_c2w(cam_pos, target))

    elif preset == "forward_orbit":
        c2ws_np = []
        orbit_radius = max(float(speed), 1e-4)
        forward_distance = float(radius)
        angles = np.linspace(0.0, 2.0 * math.pi, num_frames, endpoint=False, dtype=np.float32)
        denom = max(1, num_frames - 1)
        for frame_idx, angle in enumerate(angles):
            t = float(frame_idx) / float(denom)
            forward_z = forward_distance * t
            cam_pos = np.array(
                [
                    orbit_radius * math.cos(float(angle)),
                    orbit_radius * math.sin(float(angle)),
                    forward_z,
                ],
                dtype=np.float32,
            )
            target = np.array([0.0, 0.0, forward_z + median_depth], dtype=np.float32)
            c2ws_np.append(_opencv_look_at_c2w(cam_pos, target))

    elif preset == "forward_lookaround":
        c2ws_np = []
        forward_distance = float(radius)
        denom = max(1, num_frames - 1)
        forward_phase = 0.40
        left_phase = 0.25
        for frame_idx in range(num_frames):
            t = float(frame_idx) / float(denom)
            if t <= forward_phase:
                forward_z = forward_distance * (t / forward_phase)
                yaw = 0.0
            else:
                forward_z = forward_distance
                turn_t = (t - forward_phase) / max(1e-6, 1.0 - forward_phase)
                if turn_t <= left_phase / (1.0 - forward_phase):
                    local_t = turn_t / max(1e-6, left_phase / (1.0 - forward_phase))
                    yaw = -0.5 * math.pi * local_t
                else:
                    local_t = (turn_t - left_phase / (1.0 - forward_phase)) / max(
                        1e-6,
                        1.0 - left_phase / (1.0 - forward_phase),
                    )
                    yaw = -0.5 * math.pi + math.pi * local_t

            cam_pos = np.array([0.0, 0.0, forward_z], dtype=np.float32)
            look_dir = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float32)
            target = cam_pos + look_dir * float(median_depth)
            c2ws_np.append(_opencv_look_at_c2w(cam_pos, target))

    elif preset == "left_right":
        if not CAMERA_UTILS_AVAILABLE:
            raise RuntimeError(
                "left_right preset requires worldstereo camera_utils. "
                "Ensure worldstereo/ is present in the repo root."
            )
        c2ws_np = []
        offsets = np.linspace(-radius, radius, num_frames, dtype=np.float32)
        for offset in offsets:
            c2w = camera_left_right(c2w_start.copy(), float(offset))
            c2ws_np.append(c2w)

    elif preset == "up_down":
        c2ws_np = []
        offsets = np.linspace(-radius, radius, num_frames, dtype=np.float32)
        for offset in offsets:
            c2w = _camera_up_down(c2w_start.copy(), float(offset))
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


def _get_comfy_root() -> str:
    return (
        folder_paths.base_path if FOLDER_PATHS_AVAILABLE
        else os.path.dirname(PROJECT_ROOT)
    )


WORLDSTEREO_TURBO_LORAS = {
    "none": None,
    "CausVid 14B rank32 v2": {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        "adapter_name": "causvid_14b_rank32_v2",
    },
    "AccVid I2V 480P 14B rank32": {
        "repo_id": "Kijai/WanVideo_comfy",
        "filename": "Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors",
        "adapter_name": "accvid_i2v_480p_14b_rank32",
    },
}


def _download_worldstereo_turbo_lora(lora_name: str, models_base: str | None = None):
    spec = WORLDSTEREO_TURBO_LORAS.get(lora_name)
    if spec is None:
        return None, None

    from huggingface_hub import hf_hub_download

    lora_dir = os.path.join(models_base or _get_models_base(), "loras", "wan")
    os.makedirs(lora_dir, exist_ok=True)
    lora_path = os.path.join(lora_dir, spec["filename"])

    if not os.path.exists(lora_path):
        print(f"[WorldStereo] Downloading turbo LoRA: {lora_name} ...")
        lora_path = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=lora_dir,
        )
    else:
        print(f"[WorldStereo] Turbo LoRA cached: {lora_path}")

    return lora_path, spec


def _call_lora_method(method_name: str, calls):
    last_error = None
    for call in calls:
        try:
            return call()
        except TypeError as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise AttributeError(method_name)


def _patch_worldstereo_lora_adapter_scaling():
    try:
        from diffusers.loaders import peft as diffusers_peft
        mapping = getattr(diffusers_peft, "_SET_ADAPTER_SCALE_FN_MAPPING", None)
    except Exception as e:
        print(f"[WorldStereo] LoRA adapter scale patch skipped ({type(e).__name__}: {e})")
        return

    if not isinstance(mapping, dict):
        return

    wan_scale_fn = mapping.get("WanTransformer3DModel")
    if wan_scale_fn is None:
        return

    patched = []
    for class_name in ("WorldStereoModel", "WorldStereoRefSModel"):
        if class_name not in mapping:
            mapping[class_name] = wan_scale_fn
            patched.append(class_name)
    if patched:
        print(f"[WorldStereo] LoRA adapter scaling patched for: {', '.join(patched)}")


def _apply_worldstereo_turbo_lora(pipeline, lora_name: str, lora_strength: float, models_base: str | None = None):
    lora_path, spec = _download_worldstereo_turbo_lora(lora_name, models_base=models_base)
    if spec is None:
        return None

    adapter_name = spec["adapter_name"]
    lora_strength = float(lora_strength)
    half_dtype = _worldstereo_half_dtype()

    # LoRA injection needs regular module weights; run it before fp8 quant/freeze.
    for name in ("text_encoder", "image_encoder", "transformer", "vae"):
        _move_module_to_half(getattr(pipeline, name, None), half_dtype)
    _patch_worldstereo_lora_adapter_scaling()

    try:
        _call_lora_method(
            "load_lora_weights",
            (
                lambda: pipeline.load_lora_weights(
                    os.path.dirname(lora_path),
                    weight_name=os.path.basename(lora_path),
                    adapter_name=adapter_name,
                ),
                lambda: pipeline.load_lora_weights(lora_path, adapter_name=adapter_name),
            ),
        )
        _call_lora_method(
            "set_adapters",
            (
                lambda: pipeline.set_adapters([adapter_name], adapter_weights=[lora_strength]),
                lambda: pipeline.set_adapters([adapter_name], weights=[lora_strength]),
                lambda: pipeline.set_adapters(adapter_name, adapter_weights=lora_strength),
            ),
        )
        _call_lora_method(
            "fuse_lora",
            (
                lambda: pipeline.fuse_lora(adapter_names=[adapter_name], lora_scale=1.0),
                lambda: pipeline.fuse_lora(lora_scale=1.0),
                lambda: pipeline.fuse_lora(),
            ),
        )
        if hasattr(pipeline, "unload_lora_weights"):
            try:
                pipeline.unload_lora_weights()
            except Exception as e:
                print(f"[WorldStereo] Turbo LoRA adapter cleanup skipped ({type(e).__name__}: {e})")
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply WorldStereo turbo LoRA {lora_name!r} from {lora_path}: {e}"
        ) from e

    print(f"[WorldStereo] Turbo LoRA applied and fused: {lora_name} (strength={lora_strength})")
    return lora_path


def _normalize_worldstereo_offload_mode(offload_mode: str, precision: str, prefix: str = "[WorldStereo]") -> str:
    if precision == "fp8" and offload_mode == "sequential_cpu_offload":
        print(
            f"{prefix} fp8 quanto tensors are incompatible with accelerate sequential_cpu_offload; "
            "using model_cpu_offload instead."
        )
        return "model_cpu_offload"
    return offload_mode


def _worldstereo_half_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.bfloat16


def _move_module_to_half(module, dtype: torch.dtype):
    if module is not None and hasattr(module, "to"):
        module.to(dtype=dtype)
    return module


def _metadata_lora_matches(metadata_turbo_lora: str, metadata_lora_strength: str | None, lora_name: str, lora_strength: float) -> bool:
    if not metadata_turbo_lora or metadata_turbo_lora == "none" or lora_name == "none":
        return False
    if metadata_turbo_lora != lora_name:
        return False
    try:
        return abs(float(metadata_lora_strength or 1.0) - float(lora_strength)) < 1e-6
    except (TypeError, ValueError):
        return False


def _apply_worldstereo_precision(pipeline, precision: str, *, skip_transformer: bool = False):
    """Apply non-fp32 precision to every heavy WorldStereo module."""
    half_dtype = _worldstereo_half_dtype()
    module_names = ("text_encoder", "image_encoder", "transformer", "vae")
    if skip_transformer:
        module_names = tuple(name for name in module_names if name != "transformer")

    for name in module_names:
        _move_module_to_half(getattr(pipeline, name, None), half_dtype)

    if precision == "bf16":
        suffix = " (transformer preserved from checkpoint)" if skip_transformer else ""
        print(f"[WorldStereo] bf16/fp16 runtime dtype applied: {half_dtype}{suffix}")
        return "bf16"

    if precision == "int4" and skip_transformer:
        print(f"[WorldStereo] bf16/fp16 runtime dtype applied to aux modules: {half_dtype} (int4 transformer preserved)")
        return "int4"

    if precision != "fp8":
        raise ValueError(f"Unsupported precision: {precision!r}")

    try:
        from optimum.quanto import freeze, qfloat8_e4m3fn, quantize
    except ImportError:
        if skip_transformer:
            print("[WorldStereo] optimum-quanto not installed; preserving checkpoint transformer precision")
            return "fp8"
        print("[WorldStereo] optimum-quanto not installed; falling back to bf16/fp16 weights")
        return "bf16"

    quantized = []
    skipped = []
    for name in module_names:
        module = getattr(pipeline, name, None)
        if module is None:
            continue
        try:
            quantize(module, weights=qfloat8_e4m3fn)
            freeze(module)
            quantized.append(name)
        except Exception as e:
            skipped.append(f"{name} ({type(e).__name__}: {e})")
            _move_module_to_half(module, half_dtype)

    if quantized:
        print(f"[WorldStereo] fp8 weight quantization applied: {', '.join(quantized)}")
    if skipped:
        print(f"[WorldStereo] fp8 unavailable for: {', '.join(skipped)}; kept at {half_dtype}")
    if skip_transformer:
        return "fp8"
    return "fp8" if quantized else "bf16"


def _apply_worldstereo_transformer_export_precision(pipeline, precision: str):
    half_dtype = _worldstereo_half_dtype()
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("WorldStereo pipeline has no transformer to export.")

    _move_module_to_half(transformer, half_dtype)
    if precision == "bf16":
        return "bf16"
    if precision != "fp8":
        raise ValueError(f"Unsupported export precision: {precision!r}")

    try:
        from optimum.quanto import freeze, qfloat8_e4m3fn, quantize
    except ImportError as e:
        raise ImportError("optimum-quanto required for fp8 export: pip install optimum-quanto") from e

    quantize(transformer, weights=qfloat8_e4m3fn)
    freeze(transformer)
    return "fp8"


def _apply_worldstereo_aux_export_precision(module, precision: str):
    half_dtype = _worldstereo_half_dtype()
    _move_module_to_half(module, half_dtype)
    if precision == "bf16":
        return "bf16"
    if precision == "int4":
        print("[WorldStereo Export] int4 is transformer-only; exporting auxiliary model as fp8 instead.")
        precision = "fp8"
    if precision != "fp8":
        raise ValueError(f"Unsupported export precision: {precision!r}")

    try:
        from optimum.quanto import freeze, qfloat8_e4m3fn, quantize
    except ImportError as e:
        raise ImportError("optimum-quanto required for fp8 export: pip install optimum-quanto") from e

    quantize(module, weights=qfloat8_e4m3fn)
    freeze(module)
    return "fp8"


def _comfy_scaled_fp8_state_dict_for_t5(module):
    half_dtype = _worldstereo_half_dtype()
    state = _tensor_state_dict_for_safetensors(module)
    state.pop("encoder.embed_tokens.weight", None)
    state = _dedupe_safetensors_state_dict(state)

    quantized = 0
    for key in list(state.keys()):
        if not key.endswith(".weight"):
            continue
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or not tensor.is_floating_point():
            continue
        if key == "shared.weight":
            continue

        source = tensor.detach().to(device="cpu", dtype=torch.float32)
        max_abs = source.abs().max()
        if max_abs.item() == 0:
            scale = torch.ones((), dtype=torch.float32)
        else:
            scale = (max_abs / 448.0).clamp(min=torch.finfo(torch.float32).tiny).to(torch.float32)
        state[key] = torch.clamp(source / scale, min=-448.0, max=448.0).to(torch.float8_e4m3fn).contiguous()
        state[f"{key[:-len('.weight')]}.scale_weight"] = scale.contiguous()
        quantized += 1

    state["scaled_fp8"] = torch.empty((0,), dtype=torch.float32)
    print(f"[WorldStereo Export] ComfyUI scaled fp8 T5 weights: {quantized}")
    return state, "fp8"


def _tensor_state_dict_for_safetensors(module):
    state = {}
    for key, value in module.state_dict().items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu().contiguous()
    return state


def _dedupe_safetensors_state_dict(state: dict) -> dict:
    seen = {}
    deduped = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            deduped[key] = value
            continue
        storage_key = (value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        if storage_key in seen:
            print(f"[WorldStereo Export] Dropping tied tensor duplicate: {key} -> {seen[storage_key]}")
            continue
        seen[storage_key] = key
        deduped[key] = value
    return deduped


def _export_worldstereo_aux_single_model(
    *,
    output_file: str,
    metadata_file: str,
    model_type: str,
    precision: str,
    device: str,
    export_models_base: str,
    write_metadata_json: bool,
):
    import gc
    from diffusers.models import AutoencoderKLWan
    from safetensors.torch import save_file
    from transformers import UMT5EncoderModel

    if model_type not in VNCCS_ExportWorldStereoSingleModel.AUX_MODEL_TYPES:
        raise ValueError(f"Unsupported auxiliary export model_type: {model_type!r}")

    base_model_dir = _resolve_wan_base_aux(
        ["text_encoder"] if model_type == "official-clip-umt5-encoder" else ["vae"],
        models_base=_get_models_base(),
    )
    base_model_dir = base_model_dir.replace("\\", "/")
    print(f"[WorldStereo Export] Loading official auxiliary model from {base_model_dir}")

    if model_type == "official-clip-umt5-encoder":
        module = UMT5EncoderModel.from_pretrained(
            base_model_dir,
            subfolder="text_encoder",
            torch_dtype=_worldstereo_half_dtype(),
            local_files_only=True,
        ).eval()
        module = module.to(device=device)
        _move_module_to_half(module, _worldstereo_half_dtype())
        if precision == "int4":
            print("[WorldStereo Export] int4 is transformer-only; exporting UMT5 as ComfyUI fp8 instead.")
            precision = "fp8"
        if precision == "fp8":
            state, actual_precision = _comfy_scaled_fp8_state_dict_for_t5(module)
        elif precision == "bf16":
            state = _tensor_state_dict_for_safetensors(module)
            state.pop("encoder.embed_tokens.weight", None)
            state = _dedupe_safetensors_state_dict(state)
            actual_precision = "bf16"
        else:
            raise ValueError(f"Unsupported UMT5 export precision: {precision!r}")
        spiece_path = os.path.join(base_model_dir, "tokenizer", "spiece.model")
        if not os.path.exists(spiece_path):
            raise FileNotFoundError(f"Wan tokenizer spiece.model not found: {spiece_path}")
        with open(spiece_path, "rb") as fh:
            state["spiece_model"] = torch.ByteTensor(list(fh.read()))
        export_format = "hyworld2_worldstereo_umt5_encoder_v1"
        note = "Official Wan UMT5 encoder exported from WorldStereo base_model."
    else:
        module = AutoencoderKLWan.from_pretrained(
            base_model_dir,
            subfolder="vae",
            torch_dtype=_worldstereo_half_dtype(),
            local_files_only=True,
        ).eval()
        module = module.to(device=device)
        if precision != "bf16":
            print("[WorldStereo Export] Single-file Wan VAE loader expects normal weights; exporting VAE as bf16.")
        actual_precision = "bf16"
        _move_module_to_half(module, _worldstereo_half_dtype())
        state = _tensor_state_dict_for_safetensors(module)
        state = _dedupe_safetensors_state_dict(state)
        export_format = "hyworld2_worldstereo_wan_vae_v1"
        note = "Official Wan VAE exported from WorldStereo base_model."

    total_params = sum(t.numel() for t in state.values())
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    dtype_summary = {}
    for tensor in state.values():
        dtype_summary[str(tensor.dtype)] = dtype_summary.get(str(tensor.dtype), 0) + tensor.numel()

    metadata = {
        "format": export_format,
        "model_type": model_type,
        "precision": actual_precision,
        "runtime_dtype": str(_worldstereo_half_dtype()),
        "base_model": base_model_dir,
        "num_tensors": str(len(state)),
        "num_params": str(total_params),
        "tensor_bytes": str(total_bytes),
        "dtype_summary": json.dumps(dtype_summary, sort_keys=True),
        "note": note,
    }

    print(f"[WorldStereo Export] Saving auxiliary checkpoint: {output_file}")
    save_file(state, output_file, metadata=metadata)
    if write_metadata_json:
        with open(metadata_file, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    del state
    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    info = (
        f"exported={output_file}\n"
        f"metadata={metadata_file if write_metadata_json else ''}\n"
        f"model_type={model_type}\n"
        f"precision={actual_precision}\n"
        f"tensors={metadata['num_tensors']}\n"
        f"params={metadata['num_params']}\n"
        f"bytes={metadata['tensor_bytes']}\n"
        f"dtypes={metadata['dtype_summary']}"
    )
    print(f"[WorldStereo Export] Done:\n{info}")
    return info


def _pack_int4_weight(weight: torch.Tensor, group_size: int = 128):
    weight = weight.detach().cpu().to(torch.float32).contiguous()
    if weight.ndim != 2:
        raise ValueError(f"int4 packing expects 2D linear weights, got {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    pad = (-in_features) % group_size
    if pad:
        weight = torch.nn.functional.pad(weight, (0, pad))
    padded_in = weight.shape[1]

    grouped = weight.view(out_features, padded_in // group_size, group_size)
    scale = grouped.abs().amax(dim=2).clamp(min=1e-8) / 7.0
    quant = torch.round(grouped / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16) + 8
    quant = quant.to(torch.uint8).view(out_features, padded_in)
    packed = quant[:, 0::2] | (quant[:, 1::2] << 4)
    shape = torch.tensor([out_features, in_features], dtype=torch.int64)
    return packed.contiguous(), scale.to(torch.float16).contiguous(), shape


def _int4_state_dict_for_safetensors(module, group_size: int = 128):
    state = {}
    linear_weight_keys = set()
    packed_params = 0
    packed_bytes = 0

    for module_name, child in module.named_modules():
        if isinstance(child, torch.nn.Linear):
            key = f"{module_name}.weight" if module_name else "weight"
            linear_weight_keys.add(key)
            packed, scale, shape = _pack_int4_weight(child.weight, group_size=group_size)
            state[f"{key}_packed"] = packed
            state[f"{key}_scale"] = scale
            state[f"{key}_shape"] = shape
            packed_params += child.weight.numel()
            packed_bytes += packed.numel() * packed.element_size() + scale.numel() * scale.element_size()

    for key, value in module.state_dict().items():
        if not isinstance(value, torch.Tensor) or key in linear_weight_keys:
            continue
        state[key] = value.detach().cpu().contiguous()

    return state, packed_params, packed_bytes, len(linear_weight_keys)


def _resolve_worldstereo_export_paths(export_path: str, model_type: str, precision: str, turbo_lora: str):
    export_path = os.path.abspath(os.path.expanduser(export_path.strip()))
    if export_path.lower().endswith(".safetensors"):
        output_file = export_path
    else:
        if model_type in VNCCS_ExportWorldStereoSingleModel.AUX_MODEL_TYPES:
            output_file = os.path.join(export_path, f"{model_type}-{precision}.safetensors")
        else:
            suffix = turbo_lora.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            suffix = suffix if suffix and suffix != "none" else "clean"
            output_file = os.path.join(export_path, f"{model_type}-{suffix}-{precision}.safetensors")
    metadata_file = os.path.splitext(output_file)[0] + ".json"
    return output_file, metadata_file


def _worldstereo_export_cache_dir(output_file: str) -> str:
    output_dir = os.path.dirname(os.path.abspath(output_file)) or os.getcwd()
    return os.path.join(output_dir, "_worldstereo_export_cache")


def _read_safetensors_metadata(path: str) -> dict:
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as fh:
            return dict(fh.metadata() or {})
    except Exception as e:
        print(f"[WorldStereo] Could not read safetensors metadata ({type(e).__name__}: {e})")
        return {}


WORLDSTEREO_LIGHT_REPO_ID = "MIUProject/VNCCS_WorldStereoLight"
WORLDSTEREO_LIGHT_TEXT_ENCODER_ASSET = {
    "filename": "clip/WorldStereo_umt5-xxl-encoder-fp8.safetensors",
    "local_subdir": "clip",
    "local_filename": "WorldStereo_umt5-xxl-encoder-fp8.safetensors",
    "label": "WorldStereo UMT5 encoder",
}
WORLDSTEREO_LIGHT_VAE_ASSET = {
    "filename": "vae/WorldStereo_wan_2.1_vae.safetensors",
    "local_subdir": "vae",
    "local_filename": "WorldStereo_wan_2.1_vae.safetensors",
    "label": "WorldStereo Wan VAE",
}
WORLDSTEREO_LIGHT_MODELS = {
    "Camera Light INT4": {
        "filename": "vnccs-worldstereo-camera-light-int4.safetensors",
        "metadata_filename": "vnccs-worldstereo-camera-light-int4.json",
        "fallback_model_type": "worldstereo-camera",
        "fallback_precision": "int4",
    },
    "Memory DMD Light INT4": {
        "filename": "vnccs-worldstereo-memory-dmd-int4.safetensors",
        "metadata_filename": "vnccs-worldstereo-memory-dmd-int4.json",
        "fallback_model_type": "worldstereo-memory-dmd",
        "fallback_precision": "int4",
    },
}


def _read_json_metadata(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WorldStereoLight] Could not read json metadata ({type(e).__name__}: {e})")
        return {}


def _download_worldstereo_light_model(model_name: str, models_base: str | None = None) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download
    import shutil

    if model_name not in WORLDSTEREO_LIGHT_MODELS:
        raise ValueError(f"Unknown WorldStereoLight model: {model_name!r}")

    spec = WORLDSTEREO_LIGHT_MODELS[model_name]
    light_dir = os.path.join(models_base or _get_models_base(), "WorldStereoLight")
    os.makedirs(light_dir, exist_ok=True)

    safetensors_filename = f"models/{spec['filename']}"
    json_filename = f"models/{spec['metadata_filename']}"
    local_safetensors = os.path.join(light_dir, spec["filename"])
    local_json = os.path.join(light_dir, spec["metadata_filename"])

    if not os.path.exists(local_safetensors):
        print(f"[WorldStereoLight] Downloading selected model: {model_name} ...")
        cached_safetensors = hf_hub_download(
            repo_id=WORLDSTEREO_LIGHT_REPO_ID,
            filename=safetensors_filename,
        )
        shutil.copy2(cached_safetensors, local_safetensors)
    else:
        print(f"[WorldStereoLight] Model cached: {local_safetensors}")

    if not os.path.exists(local_json):
        print(f"[WorldStereoLight] Downloading metadata: {spec['metadata_filename']} ...")
        cached_json = hf_hub_download(
            repo_id=WORLDSTEREO_LIGHT_REPO_ID,
            filename=json_filename,
        )
        shutil.copy2(cached_json, local_json)

    metadata = _read_json_metadata(local_json)
    metadata.update(_read_safetensors_metadata(local_safetensors))
    metadata.setdefault("model_type", spec["fallback_model_type"])
    metadata.setdefault("precision", spec["fallback_precision"])
    metadata.setdefault("turbo_lora", "none")
    metadata.setdefault("lora_strength", "1.0")

    return local_safetensors, metadata


def _download_worldstereo_light_asset(asset: dict, models_base: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    import shutil

    root = models_base or _get_models_base()
    local_dir = os.path.join(root, asset["local_subdir"])
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, asset["local_filename"])
    if os.path.exists(local_path):
        print(f"[WorldStereoLight] Using local {asset.get('label', 'asset')}: {local_path}")
        return local_path

    print(f"[WorldStereoLight] Downloading {asset.get('label', 'asset')}: {asset['filename']} ...")
    cached_path = hf_hub_download(
        repo_id=WORLDSTEREO_LIGHT_REPO_ID,
        filename=asset["filename"],
    )
    shutil.copy2(cached_path, local_path)
    print(f"[WorldStereoLight] Asset cached: {local_path}")
    return local_path


def _enable_worldstereo_pipeline_offload(pipeline, offload_mode: str, *, preserve_text_encoder: bool, prefix: str):
    text_encoder = getattr(pipeline, "text_encoder", None)
    if preserve_text_encoder and text_encoder is not None:
        print(f"{prefix} Keeping ComfyUI quantized text encoder out of accelerate CPU offload.")
        pipeline.text_encoder = None
    try:
        if offload_mode == "model_cpu_offload":
            print(f"{prefix} Enabling model_cpu_offload ...")
            pipeline.enable_model_cpu_offload()
            print(f"{prefix} model_cpu_offload enabled")
        elif offload_mode == "sequential_cpu_offload":
            print(f"{prefix} Enabling sequential_cpu_offload ...")
            pipeline.enable_sequential_cpu_offload()
            print(f"{prefix} sequential_cpu_offload enabled")
    finally:
        if preserve_text_encoder and text_encoder is not None:
            pipeline.text_encoder = text_encoder


def _download_hf_repo_missing(repo_id: str, local_dir: str, label: str, allow_patterns=None):
    from fnmatch import fnmatch
    from huggingface_hub import HfApi, hf_hub_download

    def _allowed(path: str) -> bool:
        if allow_patterns is None:
            return True
        return any(fnmatch(path, pattern) for pattern in allow_patterns)

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(repo_id, recursive=True))
        remote_files = [
            (entry.path, getattr(entry, "size", None))
            for entry in entries
            if getattr(entry, "path", None) and _allowed(entry.path) and getattr(entry, "size", None) is not None
        ]
    except Exception:
        remote_files = [(path, None) for path in api.list_repo_files(repo_id) if _allowed(path)]

    missing = []
    mismatched = []
    for rel_path, remote_size in remote_files:
        local_path = os.path.join(local_dir, *rel_path.split("/"))
        if not os.path.exists(local_path):
            missing.append((rel_path, remote_size))
        elif remote_size is not None and os.path.getsize(local_path) != remote_size:
            mismatched.append((rel_path, remote_size, os.path.getsize(local_path)))

    if not missing and not mismatched:
        print(f"[WorldStereo] {label} cached: {local_dir}")
        return

    os.makedirs(local_dir, exist_ok=True)
    total_size = sum(size or 0 for _, size in missing) + sum(size or 0 for _, size, _ in mismatched)
    size_msg = f", {total_size / (1024 ** 3):.2f} GB" if total_size else ""
    print(
        f"[WorldStereo] Resuming {label}: "
        f"{len(missing)} missing, {len(mismatched)} incomplete/corrupt files{size_msg}"
    )

    for rel_path, remote_size, local_size in mismatched:
        local_path = os.path.join(local_dir, *rel_path.split("/"))
        print(
            f"[WorldStereo] Re-fetching incomplete file: {rel_path} "
            f"({local_size} / {remote_size} bytes)"
        )
        try:
            os.remove(local_path)
        except OSError:
            pass

    files_to_fetch = [(path, size) for path, size in missing] + [(path, size) for path, size, _ in mismatched]
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(files_to_fetch, desc=f"[WorldStereo] {label}", unit="file")
    except Exception:
        iterator = files_to_fetch

    for index, (rel_path, _) in enumerate(iterator, start=1):
        print(f"[WorldStereo] [{index}/{len(files_to_fetch)}] downloading {rel_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            local_dir=local_dir,
        )

    print(f"[WorldStereo] {label} cached: {local_dir}")


def _wan_base_aux_ready(base_model_dir: str, components) -> bool:
    checks = [os.path.join(base_model_dir, "model_index.json")]
    if "text_encoder" in components:
        checks.extend([
            os.path.join(base_model_dir, "text_encoder", "config.json"),
            os.path.join(base_model_dir, "text_encoder", "model.safetensors.index.json"),
            os.path.join(base_model_dir, "tokenizer", "tokenizer_config.json"),
        ])
    if "vae" in components:
        checks.extend([
            os.path.join(base_model_dir, "vae", "config.json"),
            os.path.join(base_model_dir, "vae", "diffusion_pytorch_model.safetensors"),
        ])
    return all(os.path.exists(path) for path in checks)


def _hf_component_weights_ready(component_dir: str) -> bool:
    """Return True only when a local HF component has all indexed weight shards."""
    if not os.path.isfile(os.path.join(component_dir, "config.json")):
        return False

    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.bin.index.json",
    ):
        index_path = os.path.join(component_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        try:
            with open(index_path, "r", encoding="utf-8") as fh:
                weight_map = json.load(fh).get("weight_map", {})
            shards = set(weight_map.values())
            return bool(shards) and all(os.path.isfile(os.path.join(component_dir, shard)) for shard in shards)
        except Exception:
            return False

    return any(
        os.path.isfile(os.path.join(component_dir, filename))
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
        )
    )


def _worldstereo_light_aux_missing(base_model_dir: str, model_type: str) -> list[str]:
    """Validate files actually read by WorldStereoLight without consulting HF."""
    missing = []

    exact_files = {
        "transformer/config.json": os.path.join(base_model_dir, "transformer", "config.json"),
        "vae/config.json": os.path.join(base_model_dir, "vae", "config.json"),
        "image_processor/preprocessor_config.json": os.path.join(
            base_model_dir, "image_processor", "preprocessor_config.json"
        ),
        "tokenizer/tokenizer_config.json": os.path.join(base_model_dir, "tokenizer", "tokenizer_config.json"),
    }
    for label, path in exact_files.items():
        if not os.path.isfile(path):
            missing.append(label)

    tokenizer_dir = os.path.join(base_model_dir, "tokenizer")
    if not any(
        os.path.isfile(os.path.join(tokenizer_dir, filename))
        for filename in ("tokenizer.json", "spiece.model", "tokenizer.model")
    ):
        missing.append("tokenizer vocabulary")

    if not _hf_component_weights_ready(os.path.join(base_model_dir, "image_encoder")):
        missing.append("image_encoder weights/config")

    if model_type in ("worldstereo-camera", "worldstereo-memory"):
        scheduler_config = os.path.join(base_model_dir, "scheduler", "scheduler_config.json")
        if not os.path.isfile(scheduler_config):
            missing.append("scheduler/scheduler_config.json")

    return missing


def _resolve_wan_base_aux(components, models_base: str | None = None) -> str:
    base_model_dir = os.path.join(models_base or _get_models_base(), "Wan2.1-I2V-14B-480P")
    if _wan_base_aux_ready(base_model_dir, components):
        print(f"[WorldStereo Export] Using local Wan base aux files: {base_model_dir}")
        return base_model_dir

    patterns = ["model_index.json"]
    if "text_encoder" in components:
        patterns.extend(["tokenizer/**", "text_encoder/**"])
    if "vae" in components:
        patterns.append("vae/**")
    _download_hf_repo_missing(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        local_dir=base_model_dir,
        label=f"Wan2.1-I2V-14B-480P {'/'.join(components)} files",
        allow_patterns=patterns,
    )
    return base_model_dir


def _download_worldstereo_single_loader_components(
    model_type: str,
    include_moge: bool = True,
    models_base: str | None = None,
) -> tuple:
    """Download only files needed around a single transformer checkpoint."""
    from huggingface_hub import hf_hub_download, snapshot_download

    base = models_base or _get_models_base()

    transformer_dir = os.path.join(base, "WorldStereo", model_type)
    config_path = os.path.join(transformer_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[WorldStereo] Downloading WorldStereo config ({model_type}) ...")
        hf_hub_download(
            repo_id="hanshanxue/WorldStereo",
            filename=f"{model_type}/config.json",
            local_dir=os.path.join(base, "WorldStereo"),
        )
    else:
        print(f"[WorldStereo] WorldStereo config cached: {config_path}")

    base_model_dir = os.path.join(base, "Wan2.1-I2V-14B-480P")
    missing_aux = _worldstereo_light_aux_missing(base_model_dir, model_type)
    if not missing_aux:
        print(f"[WorldStereoLight] Using complete local Wan aux files: {base_model_dir}")
    else:
        print(f"[WorldStereoLight] Local Wan aux incomplete; missing: {', '.join(missing_aux)}")
        try:
            _download_hf_repo_missing(
                repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                local_dir=base_model_dir,
                label="Wan2.1-I2V-14B-480P aux files",
                allow_patterns=[
                    "model_index.json",
                    "scheduler/**",
                    "tokenizer/**",
                    "image_encoder/**",
                    "image_processor/**",
                    "vae/config.json",
                    "transformer/config.json",
                ],
            )
        except Exception as exc:
            raise RuntimeError(
                "WorldStereoLight cannot complete the local Wan auxiliary model while offline. "
                f"Missing before download: {', '.join(missing_aux)}. "
                f"Local directory: {base_model_dir}"
            ) from exc

    moge_dir = os.path.join(base, "MoGe")
    if include_moge:
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

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.get("base_model") != base_model_dir:
            cfg["base_model"] = base_model_dir
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=4)
            print(f"[WorldStereo] config.json patched -> base_model={base_model_dir}")

    return transformer_dir, base_model_dir, moge_dir


def _download_worldstereo_components(
    model_type: str,
    include_moge: bool = True,
    models_base: str | None = None,
    worldstereo_models_base: str | None = None,
    aux_models_base: str | None = None,
) -> tuple:
    """
    Download all required model components. Returns (transformer_dir, base_model_dir, moge_dir).
    """
    from huggingface_hub import snapshot_download

    base = models_base or _get_models_base()
    worldstereo_base = worldstereo_models_base or base
    aux_base = aux_models_base or base

    # 1. WorldStereo transformer weights
    transformer_dir = os.path.join(worldstereo_base, "WorldStereo", model_type)
    transformer_weights = os.path.join(transformer_dir, "model.safetensors")
    if not os.path.exists(transformer_weights):
        print(f"[WorldStereo] Downloading transformer ({model_type}) ...")
        tmp_dir = os.path.join(worldstereo_base, "WorldStereo", "_tmp")
        snapshot_download(
            repo_id="hanshanxue/WorldStereo",  
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
    base_model_dir = os.path.join(aux_base, "Wan2.1-I2V-14B-480P")
    _download_hf_repo_missing(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        local_dir=base_model_dir,
        label="Wan2.1-I2V-14B-480P base model",
        allow_patterns=[
            "model_index.json",
            "scheduler/**",
            "tokenizer/**",
            "text_encoder/**",
            "image_encoder/**",
            "image_processor/**",
            "vae/**",
            "transformer/**",
        ],
    )

    # 3. MoGe depth estimator
    moge_dir = os.path.join(aux_base, "MoGe")
    if include_moge:
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
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.get("base_model") != base_model_dir:
            cfg["base_model"] = base_model_dir
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=4)
            print(f"[WorldStereo] config.json patched -> base_model={base_model_dir}")

    return transformer_dir, base_model_dir, moge_dir


# ─────────────────────────────────────────────────────────────────────────────
# VNCCS_CameraTrajectoryBuilder
# ─────────────────────────────────────────────────────────────────────────────

class VNCCS_CameraTrajectoryBuilder:
    PRESETS = ["circular", "stereo_orbit", "forward_orbit", "forward_lookaround", "forward", "zoom_in", "zoom_out", "left_right", "up_down", "aerial", "custom"]
    PRESET_TOOLTIP = (
        "Trajectory preset:\n"
        "circular: classic orbit around the scene; radius controls orbit size, median_depth controls look-at distance.\n"
        "stereo_orbit: compact parallax orbit for stereo/detail capture; speed controls parallax radius.\n"
        "forward_orbit: moves forward while orbiting/parallaxing; radius controls forward distance, speed controls parallax radius.\n"
        "forward_lookaround: moves forward, then looks 90 deg left and sweeps to 90 deg right; radius controls forward distance.\n"
        "forward: straight forward camera move; speed controls per-frame movement.\n"
        "zoom_in: forward zoom over the whole path; radius controls total distance.\n"
        "zoom_out: backward zoom over the whole path; radius controls total distance.\n"
        "left_right: lateral slide from left to right; radius controls span.\n"
        "up_down: vertical slide from up to down; radius controls span.\n"
        "aerial: tilt/orbit upward camera motion; elevation_deg controls angle.\n"
        "custom: use custom_json list of 4x4 C2W camera matrices."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "preset": (cls.PRESETS, {
                    "default": "stereo_orbit",
                    "tooltip": cls.PRESET_TOOLTIP,
                }),
                "num_frames": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 21,
                    "tooltip": "Final WorldStereo frames. Dense conditioning frames are calculated internally.",
                }),
                "radius": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Orbit radius, travel distance, or lateral/up-down span depending on preset. For forward_orbit/forward_lookaround this is the forward distance.",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.001,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Per-frame translation for forward preset or parallax radius for stereo_orbit/forward_orbit.",
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
                    {
                        "default": 70.0,
                        "min": 10.0,
                        "max": 150.0,
                        "step": 1.0,
                        "tooltip": "Camera field of view in degrees for all generated trajectory frames.",
                    },
                ),
                "image_width": (
                    "INT",
                    {
                        "default": 768,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Generation/conditioning width used to build camera intrinsics.",
                    },
                ),
                "image_height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Generation/conditioning height used to build camera intrinsics.",
                    },
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
        num_frames=5,
        radius=1.0,
        speed=0.08,
        elevation_deg=15.0,
        median_depth=1.0,
        fov_deg=70.0,
        image_width=768,
        image_height=480,
        custom_json="[]",
    ):
        raw_num_frames = max(1, int(num_frames))
        if raw_num_frames > 21:
            requested_output_frames = max(1, (raw_num_frames - 1) // 4 + 1)
            print(
                "[Trajectory] legacy num_frames detected; interpreting "
                f"{raw_num_frames} conditioning frames as {requested_output_frames} final frames."
            )
        else:
            requested_output_frames = raw_num_frames
        conditioning_frames = requested_output_frames if preset == "custom" else (requested_output_frames - 1) * 4 + 1

        c2ws, intrs = _build_trajectory(
            preset,
            conditioning_frames,
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
            "preset": preset,
            "requested_output_frames": requested_output_frames,
            "conditioning_frames": int(c2ws.shape[0]),
        }
        print(
            f"[Trajectory] preset={preset}, final_frames={requested_output_frames}, "
            f"conditioning_frames={c2ws.shape[0]}, "
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
                "precision": (["fp8", "bf16"], {
                    "default": "fp8",
                    "tooltip": (
                        "fp8: quantize all supported heavy modules via optimum-quanto. "
                        "bf16: half precision fallback, never fp32."
                    ),
                }),
                "turbo_lora": (list(WORLDSTEREO_TURBO_LORAS.keys()), {
                    "default": "none",
                    "tooltip": (
                        "Optional Wan2.1 14B acceleration LoRA. It is downloaded, loaded, fused, "
                        "then the pipeline is quantized/offloaded."
                    ),
                }),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Scale for the selected turbo LoRA before fusing.",
                    },
                ),
                "offload_mode": (["sequential_cpu_offload", "model_cpu_offload", "none"], {
                    "default": "sequential_cpu_offload",
                    "tooltip": (
                        "sequential_cpu_offload: layer-by-layer, slower but less VRAM. "
                        "model_cpu_offload: move components to CPU between steps. Faster, but requires more VRAM. "
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
        precision="fp8",
        turbo_lora="none",
        lora_strength=1.0,
        offload_mode="sequential_cpu_offload",
        device="cuda",
    ):
        import sys
        import os

        # Resolve and patch WorldStereo import paths. ComfyUI may already own a
        # top-level "models" module, so make it a namespace package if needed.
        current_dir = os.path.dirname(os.path.dirname(__file__))
        _prepare_worldstereo_import_paths(current_dir)

        # ── Call download helper to resolve and prepare directory paths ───────
        transformer_dir, base_model_dir, moge_dir = _download_worldstereo_components(model_type)

        # Normalize Windows backslashes to forward slashes to prevent split('/') failures inside WorldStereo
        transformer_dir = transformer_dir.replace("\\", "/")
        base_model_dir = base_model_dir.replace("\\", "/")
        moge_dir = moge_dir.replace("\\", "/")

        with _temporary_worldstereo_runtime_patches(patch_diffusers=True):
            WorldStereo = _import_worldstereo_class(current_dir)

            # Resolve MoGeModel class import based on your installed MoGe library version
            try:
                from moge.model.v2 import MoGeModel
            except ImportError:
                try:
                    from moge.model.v1 import MoGeModel
                except ImportError:
                    from moge.model import MoGeModel

            # ── Load WorldStereo pipeline ─────────────────────────────────────
            print(f"[WorldStereo] Loading pipeline (model_type={model_type}, precision={precision}) ...")
            parent_dir = os.path.dirname(transformer_dir)
            model_device = "cpu" if device == "cuda" and offload_mode != "none" else device
            worldstereo = WorldStereo.from_pretrained(
                parent_dir,
                subfolder=model_type,
                device=device,
                model_device=model_device,
            )
        pipeline = worldstereo.pipeline

        turbo_lora_path = _apply_worldstereo_turbo_lora(pipeline, turbo_lora, lora_strength)
        precision = _apply_worldstereo_precision(pipeline, precision)
        offload_mode = _normalize_worldstereo_offload_mode(offload_mode, precision)

        # ── Apply offloading ──────────────────────────────────────────────────
        if device == "cuda":
            if offload_mode == "model_cpu_offload":
                pipeline.enable_model_cpu_offload()
                print("[WorldStereo] model_cpu_offload enabled")
            elif offload_mode == "sequential_cpu_offload":
                pipeline.enable_sequential_cpu_offload()
                print("[WorldStereo] sequential_cpu_offload enabled")
            elif offload_mode == "none":
                print("[WorldStereo Warning] CPU offload disabled; this is likely too large for normal VRAM.")

        # ── Load MoGe on CPU ─────────────────────────────────────────────────
        print("[WorldStereo] Loading MoGe depth estimator ...")
        
        # Point directly to the model.pt file inside the cached directory
        actual_moge_path = os.path.join(moge_dir, "model.pt")
        if not os.path.exists(actual_moge_path):
            actual_moge_path = moge_dir  # Fallback if structure varies
            
        moge_model = MoGeModel.from_pretrained(actual_moge_path).eval()
        try:
            moge_model.to(dtype=_worldstereo_half_dtype())
        except Exception as e:
            print(f"[WorldStereo] MoGe half precision cast skipped ({type(e).__name__}: {e})")
        print("[WorldStereo] MoGe loaded (CPU)")

        print("[WorldStereo] Pipeline ready")
        return ({
            "worldstereo": worldstereo,
            "pipeline":    pipeline,
            "moge":        moge_model,
            "device":      device,
            "model_type":  model_type,
            "precision":   precision,
            "offload_mode": offload_mode,
            "turbo_lora":  turbo_lora,
            "lora_strength": float(lora_strength),
            "turbo_lora_path": turbo_lora_path,
            "pretrained_path": parent_dir,
            "transformer_dir": transformer_dir,
            "base_model_dir": base_model_dir,
            "moge_dir": moge_dir,
            "loader_type": "worldstereo_full",
        },)


class VNCCS_LoadWorldStereoLightModel:
    """Download and load a selected WorldStereoLight single-transformer checkpoint."""

    MODELS = list(WORLDSTEREO_LIGHT_MODELS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
            },
            "optional": {
                "offload_mode": (["sequential_cpu_offload", "model_cpu_offload", "none"], {
                    "default": "sequential_cpu_offload",
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("WORLDSTEREO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/Video"

    def load_model(
        self,
        model,
        offload_mode="sequential_cpu_offload",
        device="cuda",
        use_light_aux_models=True,
    ):
        import os
        use_light_aux_models = True

        current_dir = os.path.dirname(os.path.dirname(__file__))
        _prepare_worldstereo_import_paths(current_dir)

        single_model_path, metadata = _download_worldstereo_light_model(model)
        metadata_model_type = metadata.get("model_type")
        metadata_precision = metadata.get("precision")
        model_type = metadata_model_type or WORLDSTEREO_LIGHT_MODELS[model]["fallback_model_type"]
        precision = metadata_precision or WORLDSTEREO_LIGHT_MODELS[model]["fallback_precision"]
        if model_type not in ("worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd"):
            raise ValueError(f"Unsupported WorldStereoLight model_type from metadata: {model_type!r}")
        if precision != "int4":
            raise ValueError(f"WorldStereoLight checkpoints are expected to be int4, got precision={precision!r}")

        metadata_turbo_lora = metadata.get("turbo_lora", "none")
        metadata_lora_strength = metadata.get("lora_strength")
        if metadata_turbo_lora and metadata_turbo_lora != "none":
            print(
                f"[WorldStereoLight] Checkpoint has fused turbo_lora={metadata_turbo_lora} "
                f"(strength={metadata_lora_strength or '1.0'})."
            )

        transformer_dir, base_model_dir, moge_dir = _download_worldstereo_single_loader_components(model_type)
        text_encoder_path = _download_worldstereo_light_asset(WORLDSTEREO_LIGHT_TEXT_ENCODER_ASSET)
        vae_path = _download_worldstereo_light_asset(WORLDSTEREO_LIGHT_VAE_ASSET)
        transformer_dir = transformer_dir.replace("\\", "/")
        base_model_dir = base_model_dir.replace("\\", "/")
        moge_dir = moge_dir.replace("\\", "/")
        if text_encoder_path:
            text_encoder_path = text_encoder_path.replace("\\", "/")
        if vae_path:
            vae_path = vae_path.replace("\\", "/")

        with _temporary_worldstereo_runtime_patches(patch_diffusers=True):
            WorldStereo = _import_worldstereo_class(current_dir)
            try:
                from moge.model.v2 import MoGeModel
            except ImportError:
                try:
                    from moge.model.v1 import MoGeModel
                except ImportError:
                    from moge.model import MoGeModel

            print(
                f"[WorldStereoLight] Loading pipeline "
                f"(selected={model}, model_type={model_type}, precision={precision}) ..."
            )
            parent_dir = os.path.dirname(transformer_dir)
            model_device = "cpu" if device == "cuda" and offload_mode != "none" else device
            worldstereo = WorldStereo.from_single_transformer(
                parent_dir,
                single_model_path,
                subfolder=model_type,
                local_files_only=True,
                device=device,
                model_device=model_device,
                text_encoder_path=text_encoder_path,
                vae_path=vae_path,
                comfy_root=_get_comfy_root(),
            )
        pipeline = worldstereo.pipeline

        checkpoint_transformer_ready = (
            metadata.get("format") == "hyworld2_worldstereo_single_transformer_v1"
            and metadata_precision == precision
        )
        precision = _apply_worldstereo_precision(
            pipeline,
            precision,
            skip_transformer=checkpoint_transformer_ready,
        )
        offload_mode = _normalize_worldstereo_offload_mode(offload_mode, precision, prefix="[WorldStereoLight]")

        if device == "cuda":
            if offload_mode in ("model_cpu_offload", "sequential_cpu_offload"):
                _enable_worldstereo_pipeline_offload(
                    pipeline,
                    offload_mode,
                    preserve_text_encoder=bool(use_light_aux_models),
                    prefix="[WorldStereoLight]",
                )
            elif offload_mode == "none":
                print("[WorldStereoLight Warning] CPU offload disabled; this is likely too large for normal VRAM.")

        print("[WorldStereoLight] Loading MoGe depth estimator ...")
        actual_moge_path = os.path.join(moge_dir, "model.pt")
        if not os.path.exists(actual_moge_path):
            actual_moge_path = moge_dir
        moge_model = MoGeModel.from_pretrained(actual_moge_path).eval()
        try:
            moge_model.to(dtype=_worldstereo_half_dtype())
        except Exception as e:
            print(f"[WorldStereoLight] MoGe half precision cast skipped ({type(e).__name__}: {e})")
        print("[WorldStereoLight] MoGe loaded (CPU)")

        print("[WorldStereoLight] Pipeline ready")
        return ({
            "worldstereo": worldstereo,
            "pipeline": pipeline,
            "moge": moge_model,
            "device": device,
            "model_type": model_type,
            "precision": precision,
            "offload_mode": offload_mode,
            "turbo_lora": metadata_turbo_lora,
            "lora_strength": float(metadata_lora_strength or 1.0),
            "turbo_lora_path": None,
            "single_model_path": single_model_path,
            "single_model_metadata": metadata,
            "pretrained_path": parent_dir,
            "transformer_dir": transformer_dir,
            "base_model_dir": base_model_dir,
            "moge_dir": moge_dir,
            "loader_type": "worldstereo_light",
            "selected_model": model,
            "use_light_aux_models": bool(use_light_aux_models),
        },)


def _prepare_pipeline_inputs(
    image_pil,
    c2ws: torch.Tensor,
    intrs: torch.Tensor,
    moge_model,
    device: str,
    width: int,
    height: int,
    conditioning_frame_indices: torch.Tensor | None = None,
) -> dict:
    """
    Build render_video, render_mask, camera_embedding from a single image + trajectory.
    Replicates WorldStereo's load_single_view_data() for arbitrary inputs.
    """
    import torchvision.transforms as T
    from models.camera import get_camera_embedding
    try:
        from src.pointcloud import get_points3d_and_colors, point_rendering
        use_pytorch3d_renderer = True
    except Exception as e:
        import traceback
        details = f"{type(e).__name__}: {e}\n{traceback.format_exc()}".lower()
        if "pytorch3d" not in details and "max_uint" not in details:
            raise
        print(f"[WorldStereo] PyTorch3D renderer unavailable ({type(e).__name__}: {e}); using torch fallback point renderer.")
        get_points3d_and_colors = _fallback_get_points3d_and_colors
        point_rendering = _fallback_point_rendering
        use_pytorch3d_renderer = False

    N = c2ws.shape[0]
    if conditioning_frame_indices is None:
        cond_indices = torch.arange(N, dtype=torch.long, device=c2ws.device)
    else:
        cond_indices = conditioning_frame_indices.to(device=c2ws.device, dtype=torch.long)
        if cond_indices.numel() == 0:
            cond_indices = torch.arange(N, dtype=torch.long, device=c2ws.device)
        cond_indices = cond_indices.clamp(0, N - 1)
    cond_c2ws = c2ws.index_select(0, cond_indices)
    cond_intrs = intrs.index_select(0, cond_indices)
    cond_N = cond_c2ws.shape[0]
    if cond_N != N:
        print(f"[WorldStereo] Conditioning frames prepared directly: {N} -> {cond_N}")

    # 1. Image tensor in [-1, 1] for pipeline
    img_tensor = T.ToTensor()(image_pil) * 2.0 - 1.0   # [3, H, W], range [-1, 1]
    
    # Create [1, 3, H, W] PyTorch tensor in [0, 1] for pointcloud functions
    img_tensor_01 = (img_tensor + 1.0).mul(0.5).unsqueeze(0).float() # [1, 3, H, W]

    # 2. Depth via MoGe
    torch_device = torch.device(device)
    moge_model = moge_model.to(torch_device)
    infer_dtype = _worldstereo_half_dtype()
    with torch.no_grad(), torch.autocast(device_type=torch_device.type, dtype=infer_dtype, enabled=torch_device.type == "cuda"):
        depth_output = moge_model.infer(
            img_tensor_01.to(torch_device)
        )
    # MoGe returns dict; extract depth as [1, 1, H, W] PyTorch tensor
    depth_raw = depth_output["depth"]
    if isinstance(depth_raw, torch.Tensor):
        depth_tensor = depth_raw.float()
    else:
        depth_tensor = torch.from_numpy(depth_raw).float()
    if depth_tensor.dim() == 2:
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    elif depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(0)
    if "mask" in depth_output:
        mask_raw = depth_output["mask"]
        if isinstance(mask_raw, torch.Tensor):
            depth_mask = mask_raw.bool()
        else:
            depth_mask = torch.from_numpy(mask_raw).bool()
        if depth_mask.dim() == 2:
            depth_mask = depth_mask.unsqueeze(0).unsqueeze(0)
        elif depth_mask.dim() == 3:
            depth_mask = depth_mask.unsqueeze(0)
        depth_mask = depth_mask.to(depth_tensor.device)
        if depth_mask.shape[-2:] == depth_tensor.shape[-2:]:
            depth_tensor = depth_tensor.masked_fill(~depth_mask, 0.0)
    depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    moge_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. W2C matrices as PyTorch Tensors
    w2cs_t = _c2w_to_w2c(cond_c2ws).float()   # [cond_N, 4, 4]
    intrs_t = cond_intrs.float()              # [cond_N, 3, 3]
    ref_w2c = w2cs_t[0:1]                # [1, 4, 4] reference view
    ref_K   = intrs_t[0:1]               # [1, 3, 3] reference view (preserves batch dim)

    # 4. 3D point cloud from reference view
    # Keep on CPU initially — get_points3d_and_colors will move it to target device
    points3d, colors = get_points3d_and_colors(
        K=ref_K,
        w2cs=ref_w2c,
        depth=depth_tensor.cpu(),
        image=img_tensor_01,
        device=device,
    )
    if points3d is None or colors is None:
        raise RuntimeError("Point cloud construction failed.")

    # 5. Render point cloud from selected conditioning target views
    render_result = point_rendering(
        K=intrs_t,
        w2cs=w2cs_t,
        points=points3d,
        colors=colors,
        device=device,
        h=height,
        w=width,
    )
    if not use_pytorch3d_renderer:
        print("[WorldStereo] Fallback point rendering complete.")
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
    conditioning_dtype = _worldstereo_half_dtype()
    render_video = render_rgbs_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device, dtype=conditioning_dtype)   # [1, 3, N, H, W]
    render_mask  = render_masks_t.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch_device, dtype=conditioning_dtype)  # [1, 1, N, H, W]

    # 6. Camera embedding [1, 6, N, H, W]
    camera_emb = get_camera_embedding(
        intrinsic=cond_intrs.to(torch_device),  # [cond_N, 3, 3]
        extrinsic=cond_c2ws.to(torch_device),   # [cond_N, 4, 4] C2W (is_w2c=False)
        f=cond_N, h=height, w=width,
        normalize=True,
        is_w2c=False,
    ).to(dtype=conditioning_dtype)

    return {
        "image":            image_pil,
        "render_video":     render_video,
        "render_mask":      render_mask,
        "camera_embedding": camera_emb,
        "extrinsics":       _c2w_to_w2c(c2ws).float().to(torch_device),  # [N, 4, 4] W2C
        "intrinsics":       intrs.to(torch_device),   # [N, 3, 3]
        "height":           height,
        "width":            width,
        "num_frames":       N,
    }


def _get_pipeline_execution_device(pipeline, fallback_device: str) -> torch.device:
    execution_device = getattr(pipeline, "_execution_device", None)
    if callable(execution_device):
        execution_device = execution_device()
    if execution_device is None:
        execution_device = fallback_device
    return torch.device(execution_device)


def _free_pipeline_offload_hooks(pipeline, context: str):
    if not hasattr(pipeline, "maybe_free_model_hooks"):
        return
    try:
        pipeline.maybe_free_model_hooks()
    except Exception as e:
        print(f"[WorldStereo] Offload hook cleanup skipped after {context} ({type(e).__name__}: {e})")


def _configure_vae_memory_mode(pipeline, mode: str, width: int, height: int) -> str:
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return "none"

    if mode == "auto":
        mode = "tiled+sliced" if int(width) * int(height) >= 768 * 480 else "off"

    if mode == "off":
        for method_name in ("disable_tiling", "disable_slicing"):
            method = getattr(vae, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as e:
                    print(f"[WorldStereo] VAE {method_name} skipped ({type(e).__name__}: {e})")
        return "off"

    enabled = []
    if "tiled" in mode:
        method = getattr(vae, "enable_tiling", None)
        if callable(method):
            try:
                method()
                enabled.append("tiled")
            except Exception as e:
                print(f"[WorldStereo] VAE enable_tiling skipped ({type(e).__name__}: {e})")
        else:
            print("[WorldStereo] VAE tiling not available in this diffusers AutoencoderKLWan.")

    if "sliced" in mode:
        method = getattr(vae, "enable_slicing", None)
        if callable(method):
            try:
                method()
                enabled.append("sliced")
            except Exception as e:
                print(f"[WorldStereo] VAE enable_slicing skipped ({type(e).__name__}: {e})")
        else:
            print("[WorldStereo] VAE slicing not available in this diffusers AutoencoderKLWan.")

    actual = "+".join(enabled) if enabled else "off"
    print(f"[WorldStereo] VAE memory mode: requested={mode}, active={actual}")
    return actual


def _encode_prompt_cache(
    pipeline,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    model_type: str,
    device: str,
):
    do_cfg = guidance_scale > 1.0 and "dmd" not in model_type
    execution_device = _get_pipeline_execution_device(pipeline, device)

    with torch.no_grad(), torch.autocast(
        execution_device.type,
        dtype=_worldstereo_half_dtype(),
        enabled=execution_device.type in ("cuda", "cpu"),
    ):
        embeds = pipeline.encode_prompt(
            prompt=prompt if prompt else "",
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=execution_device,
        )
    _free_pipeline_offload_hooks(pipeline, "prompt encode")
    return embeds


def _image_cache_key(image_pil) -> str:
    hasher = hashlib.sha1()
    hasher.update(str(image_pil.size).encode("ascii"))
    hasher.update(image_pil.mode.encode("ascii"))
    hasher.update(image_pil.tobytes())
    return hasher.hexdigest()


def _encode_image_cache(pipeline, image_pil, device: str):
    execution_device = _get_pipeline_execution_device(pipeline, device)
    with torch.no_grad(), torch.autocast(
        execution_device.type,
        dtype=_worldstereo_half_dtype(),
        enabled=execution_device.type in ("cuda", "cpu"),
    ):
        image_embeds = pipeline.encode_image(image_pil, execution_device)
    _free_pipeline_offload_hooks(pipeline, "image encode")
    return image_embeds


def _move_pipeline_inputs(pipeline_inputs: dict, device: str | torch.device):
    moved = {}
    for key, value in pipeline_inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _crop_generated_edges(frames: torch.Tensor, intrs: torch.Tensor, crop_percent: float):
    crop_percent = float(crop_percent)
    if crop_percent <= 0.0:
        return frames, intrs

    _, height, width, _ = frames.shape
    crop_x = int(round(width * crop_percent / 100.0))
    crop_y = int(round(height * crop_percent / 100.0))
    max_crop_x = max(0, (width - 16) // 2)
    max_crop_y = max(0, (height - 16) // 2)
    crop_x = min(crop_x, max_crop_x)
    crop_y = min(crop_y, max_crop_y)

    if crop_x == 0 and crop_y == 0:
        return frames, intrs

    cropped = frames[:, crop_y:height - crop_y, crop_x:width - crop_x, :]
    adjusted_intrs = intrs.clone()
    adjusted_intrs[:, 0, 2] -= crop_x
    adjusted_intrs[:, 1, 2] -= crop_y
    return cropped.contiguous(), adjusted_intrs


def _resize_frames_and_intrinsics(
    frames: torch.Tensor,
    intrs: torch.Tensor,
    out_width: int,
    out_height: int,
    upscale_mode: str = "bicubic",
    seedvr2_options: dict | None = None,
):
    if frames.shape[1] == out_height and frames.shape[2] == out_width:
        return frames.contiguous(), intrs

    in_height, in_width = frames.shape[1], frames.shape[2]
    if upscale_mode == "seedvr2":
        resized = _seedvr2_upscale_frames(
            frames,
            out_width,
            out_height,
            **(seedvr2_options or {}),
        )
        if resized.shape[1] != out_height or resized.shape[2] != out_width:
            print(
                "[WorldStereo] SeedVR2 output size differs from source target; "
                f"final bicubic fit {resized.shape[2]}x{resized.shape[1]} -> {out_width}x{out_height}"
            )
            resized = torch.nn.functional.interpolate(
                resized.permute(0, 3, 1, 2),
                size=(out_height, out_width),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
    else:
        resized = torch.nn.functional.interpolate(
            frames.permute(0, 3, 1, 2),
            size=(out_height, out_width),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

    adjusted_intrs = intrs.clone()
    adjusted_intrs[:, 0, :] *= float(out_width) / float(in_width)
    adjusted_intrs[:, 1, :] *= float(out_height) / float(in_height)
    return resized.contiguous(), adjusted_intrs


def _worldstereo_keyframe_indices(num_frames: int, device=None) -> torch.Tensor:
    keyframe_count = max(1, (int(num_frames) - 1) // 4 + 1)
    if keyframe_count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    indices = torch.linspace(0, int(num_frames) - 1, keyframe_count, device=device).round().long()
    return torch.unique_consecutive(indices.clamp(0, int(num_frames) - 1))


def _worldstereo_output_frame_indices(
    num_condition_frames: int,
    num_output_frames: int,
    device=None,
) -> torch.Tensor:
    if num_output_frames <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if num_output_frames == num_condition_frames:
        return torch.arange(num_condition_frames, dtype=torch.long, device=device)
    keyframe_indices = _worldstereo_keyframe_indices(num_condition_frames, device=device)
    if num_output_frames == keyframe_indices.numel():
        return keyframe_indices
    if num_output_frames == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.linspace(
        0,
        int(num_condition_frames) - 1,
        int(num_output_frames),
        device=device,
    ).round().long().clamp(0, int(num_condition_frames) - 1)


def _worldstereo_pose_pitch_deg(pose: torch.Tensor) -> float:
    R = pose[:3, :3].detach().cpu().float()
    forward = R @ torch.tensor([0.0, 0.0, 1.0])
    pitch = torch.asin(torch.clamp(-forward[1] / forward.norm().clamp_min(1e-8), -1.0, 1.0))
    return float(pitch.item() * 180.0 / math.pi)


def _drop_duplicate_first_frame(
    frames: torch.Tensor,
    poses: torch.Tensor,
    intrs: torch.Tensor,
    image_pil,
):
    if frames.shape[0] <= 1:
        return frames, poses, intrs, False
    if poses.shape[0] != frames.shape[0] or intrs.shape[0] != frames.shape[0]:
        raise RuntimeError(
            "WorldStereo frame/camera count mismatch before duplicate drop: "
            f"frames={frames.shape[0]}, poses={poses.shape[0]}, intrinsics={intrs.shape[0]}"
        )

    ref_np = np.asarray(image_pil.convert("RGB"), dtype=np.float32) / 255.0
    ref = torch.from_numpy(ref_np).to(dtype=frames.dtype, device=frames.device)
    if ref.shape != frames[0].shape:
        return frames, poses, intrs, False

    diff = (frames[0] - ref).abs()
    mean_diff = float(diff.mean().item())
    max_diff = float(diff.max().item())
    if mean_diff <= 0.006 and max_diff <= 0.04:
        print(
            "[WorldStereo] First frame matches source slice; "
            f"dropping duplicate (mean diff={mean_diff:.5f}, max diff={max_diff:.5f})"
        )
        return frames[1:].contiguous(), poses[1:].contiguous(), intrs[1:].contiguous(), True

    print(
        "[WorldStereo] First frame kept; source comparison differs "
        f"(mean diff={mean_diff:.5f}, max diff={max_diff:.5f})"
    )
    return frames, poses, intrs, False


@contextmanager
def _pipeline_conditioning_cache(pipeline, enabled: bool):
    old_enabled = getattr(pipeline, "_vnccs_cache_latent_condition", False)
    old_cache = getattr(pipeline, "_vnccs_latent_condition_cache", None)

    if enabled:
        pipeline._vnccs_cache_latent_condition = True
        pipeline._vnccs_latent_condition_cache = {}
    try:
        yield
    finally:
        if enabled:
            pipeline._vnccs_latent_condition_cache = old_cache
            pipeline._vnccs_cache_latent_condition = old_enabled


class VNCCS_WorldStereoGenerate:
    """
    WorldStereo camera-guided video generation from a single image or batch of sliced views.
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
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Text prompt to guide the inpainting/generation of occluded parts.",
                }),
                "num_inference_steps": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "0 = auto (4 for turbo LoRA or memory-dmd, 20 for others).",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Forced to 1.0 automatically when a turbo LoRA is loaded.",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1,
                    "tooltip": "-1 = random.",
                }),
                "negative_prompt": ("STRING", {"default": ""}),
                "cache_conditioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache T5 prompt embeds, CLIP image embeds, and VAE latent conditioning across slices.",
                }),
                "latent_condition_mode": (["auto", "full_vae", "first_frame_only"], {
                    "default": "auto",
                    "tooltip": "auto uses first_frame_only at high resolutions to reduce VAE encode VRAM.",
                }),
                "render_vae_mode": (["auto", "full", "keyframes"], {
                    "default": "auto",
                    "tooltip": "auto encodes keyframe render conditioning at high resolutions to reduce VAE encode VRAM.",
                }),
                "conditioning_frame_mode": (["auto", "full", "keyframes"], {
                    "default": "auto",
                    "tooltip": "auto renders only keyframe conditioning when render_vae_mode uses keyframes.",
                }),
                "vae_memory_mode": (["auto", "off", "tiled", "sliced", "tiled+sliced"], {
                    "default": "auto",
                    "tooltip": "Enable diffusers VAE tiling/slicing when available to reduce high-resolution VAE VRAM.",
                }),
                "crop_generated_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Crop generated borders from every output frame and adjust intrinsics.",
                }),
                "worldmirror_sequence_mode": (["stereo_only", "prepend_source_highres"], {
                    "default": "stereo_only",
                    "tooltip": "prepend_source_highres outputs the original high-res source view first, followed by generated frames resized to that source size.",
                }),
                "generated_upscale_mode": (["bicubic", "seedvr2"], {
                    "default": "bicubic",
                    "tooltip": "Upscaler used when prepend_source_highres needs generated video frames resized to the source image resolution.",
                }),
                "seedvr2_seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "SeedVR2 seed used only when generated_upscale_mode=seedvr2.",
                }),
                "seedvr2_resolution": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 2,
                    "tooltip": "0 = use the source image short edge for SeedVR2 resolution.",
                }),
                "seedvr2_max_resolution": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 2,
                    "tooltip": "0 = use the source image long edge as SeedVR2 max_resolution.",
                }),
                "video_view_filter": (["all_views", "zero_pitch_only"], {
                    "default": "all_views",
                    "tooltip": "zero_pitch_only runs WorldStereo video only for near-horizontal panorama views; other views pass through as high-res anchors.",
                }),
                "edge_crop_percent": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Percent to crop from each side when crop_generated_edges is enabled.",
                    },
                ),
                "base_camera_poses": ("TENSOR",),
                "base_camera_intrinsics": ("TENSOR",),
            },
        }

    RETURN_TYPES  = ("IMAGE",         "TENSOR",         "TENSOR")
    RETURN_NAMES  = ("video_frames",  "camera_poses",   "camera_intrinsics")
    FUNCTION      = "generate"
    CATEGORY = "VNCCS/Video"

    def generate(
        self,
        model,
        image,
        trajectory,
        prompt="",
        num_inference_steps=0,
        guidance_scale=5.0,
        seed=-1,
        negative_prompt="",
        cache_conditioning=True,
        latent_condition_mode="auto",
        render_vae_mode="auto",
        conditioning_frame_mode="auto",
        vae_memory_mode="auto",
        crop_generated_edges=False,
        worldmirror_sequence_mode="stereo_only",
        generated_upscale_mode="bicubic",
        seedvr2_seed=42,
        seedvr2_resolution=0,
        seedvr2_max_resolution=0,
        video_view_filter="all_views",
        edge_crop_percent=8.0,
        base_camera_poses=None,
        base_camera_intrinsics=None,
    ):
        pipeline   = model["pipeline"]
        moge_model = model["moge"]
        device     = model["device"]
        model_type = model["model_type"]
        turbo_lora = model.get("turbo_lora", "none")
        has_turbo_lora = turbo_lora not in (None, "", "none")

        c2ws  = trajectory["c2ws"]    # [N, 4, 4]
        intrs = trajectory["intrs"]   # [N, 3, 3]
        W     = trajectory["width"]
        H     = trajectory["height"]
        N     = c2ws.shape[0]
        trajectory_preset = trajectory.get("preset", "")
        high_res_pixels = int(W) * int(H)
        if latent_condition_mode == "auto":
            latent_condition_mode = "first_frame_only" if high_res_pixels >= 768 * 480 else "full_vae"
        if render_vae_mode == "auto":
            render_vae_mode = "keyframes" if high_res_pixels >= 768 * 480 else "full"
        if conditioning_frame_mode == "auto":
            conditioning_frame_mode = "keyframes" if render_vae_mode == "keyframes" else "full"
        print(
            f"[WorldStereo] VAE conditioning modes: latent_condition={latent_condition_mode}, "
            f"render_vae={render_vae_mode}, conditioning_frames={conditioning_frame_mode}"
        )
        _configure_vae_memory_mode(pipeline, vae_memory_mode, W, H)

        if num_inference_steps == 0:
            num_inference_steps = 4 if has_turbo_lora or "dmd" in model_type else 20
        if has_turbo_lora and guidance_scale != 1.0:
            print(
                f"[WorldStereo] Turbo LoRA '{turbo_lora}' loaded; "
                f"forcing guidance_scale {guidance_scale} -> 1.0"
            )
            guidance_scale = 1.0
        if trajectory_preset in ("stereo_orbit", "forward_orbit") and not crop_generated_edges:
            crop_generated_edges = True
            edge_crop_percent = max(float(edge_crop_percent), 6.0)
            print(
                f"[WorldStereo] {trajectory_preset} trajectory detected; "
                f"auto edge crop enabled ({edge_crop_percent:.1f}%)"
            )

        # Extract the total slice count from the input image batch
        B = image.shape[0] if isinstance(image, torch.Tensor) else len(image)
        print(f"[WorldStereo Batch Loop] Slices found: {B}. Initializing sequence generation loop...")

        # ── Safety and Format Checks for Poses and Intrinsics ──
        # Check if the user swapped camera_poses and intrinsics connections
        if base_camera_poses is not None and base_camera_intrinsics is not None:
            # Poses are typically [B, 4, 4] or [B, 3, 4], intrinsics are [B, 3, 3]
            # If poses has last dim 3, but intrinsics has last dim 4, they are swapped
            if base_camera_poses.shape[-1] == 3 and base_camera_intrinsics.shape[-1] == 4:
                print("[WorldStereo Warning] Swapped inputs detected! base_camera_poses and base_camera_intrinsics appear to be reversed. Swapping them back automatically.")
                base_camera_poses, base_camera_intrinsics = base_camera_intrinsics, base_camera_poses

        # If base_camera_poses was provided but looks like intrinsics [B, 3, 3]
        if base_camera_poses is not None and base_camera_poses.shape[-2] == 3 and base_camera_poses.shape[-1] == 3:
            print("[WorldStereo Warning] base_camera_poses has shape [3, 3], which looks like intrinsics. Treating it as base_camera_intrinsics instead.")
            if base_camera_intrinsics is None:
                base_camera_intrinsics = base_camera_poses
            base_camera_poses = None

        # Helper function to pad [3, 4] poses to [4, 4]
        def pad_pose_to_4x4(pose: torch.Tensor) -> torch.Tensor:
            if pose.shape[-2] == 3 and pose.shape[-1] == 4:
                if len(pose.shape) == 2:  # [3, 4]
                    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=pose.dtype, device=pose.device)
                    return torch.cat([pose, bottom], dim=0)  # [4, 4]
                elif len(pose.shape) == 3:  # [B, 3, 4]
                    B_dim = pose.shape[0]
                    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=pose.dtype, device=pose.device)
                    bottom = bottom.unsqueeze(0).repeat(B_dim, 1, 1)  # [B, 1, 4]
                    return torch.cat([pose, bottom], dim=1)  # [B, 4, 4]
            return pose

        # Pad base camera poses if they exist
        if base_camera_poses is not None:
            base_camera_poses = pad_pose_to_4x4(base_camera_poses)

        all_video_frames = []
        all_camera_poses = []
        all_camera_intrinsics = []
        cached_prompt_embeds = None
        cached_negative_prompt_embeds = None
        image_embeds_cache = {}
        latent_condition_cache = {}

        if cache_conditioning:
            print("[WorldStereo] Conditioning cache enabled: T5 prompt, CLIP image, VAE latent condition")
            cached_prompt_embeds, cached_negative_prompt_embeds = _encode_prompt_cache(
                pipeline=pipeline,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                model_type=model_type,
                device=device,
            )
            cached_prompt_embeds = cached_prompt_embeds.to("cpu")
            if cached_negative_prompt_embeds is not None:
                cached_negative_prompt_embeds = cached_negative_prompt_embeds.to("cpu")

        # Iterate through each panoramic view slice
        for b in range(B):
            print(f"[WorldStereo Batch Loop] Processing view slice {b + 1}/{B}...")
            
            # Extract slice image and convert to PIL
            img_slice = image[b] if isinstance(image, torch.Tensor) else image[b]
            source_h, source_w = int(img_slice.shape[0]), int(img_slice.shape[1])
            img_np  = (img_slice.cpu().numpy()[..., :3] * 255).astype(np.uint8)
            source_frame = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            source_pil = PILImage.fromarray(img_np)
            img_pil = source_pil.resize((W, H), PILImage.Resampling.BICUBIC)

            # Determine the base camera pose for this specific slice
            if base_camera_poses is not None:
                if len(base_camera_poses.shape) == 3:  # Shape [B, 4, 4]
                    idx_pose = b if b < base_camera_poses.shape[0] else 0
                    base_pose = base_camera_poses[idx_pose]
                else:  # Fallback to single [4, 4] matrix
                    base_pose = base_camera_poses
            else:
                # Identity matrix fallback if unconnected
                base_pose = torch.eye(4, dtype=torch.float32)

            # Determine the base intrinsics for this specific slice
            if base_camera_intrinsics is not None:
                if len(base_camera_intrinsics.shape) == 3:  # Shape [B, 3, 3]
                    idx_intr = b if b < base_camera_intrinsics.shape[0] else 0
                    base_K = base_camera_intrinsics[idx_intr]
                else:  # Fallback to single [3, 3] matrix
                    base_K = base_camera_intrinsics
            else:
                # Fallback to standard intrinsics from the trajectory
                base_K = intrs[0]

            base_K_raw = base_K.to(intrs.device, dtype=intrs.dtype)
            base_K_dev = base_K_raw.clone()
            base_K_source = base_K_raw.clone()
            if source_w != W or source_h != H:
                if base_camera_intrinsics is not None:
                    base_K_dev[0, :] *= float(W) / float(source_w)
                    base_K_dev[1, :] *= float(H) / float(source_h)
                else:
                    base_K_source[0, :] *= float(source_w) / float(W)
                    base_K_source[1, :] *= float(source_h) / float(H)

            base_pitch = _worldstereo_pose_pitch_deg(base_pose)
            should_video = video_view_filter != "zero_pitch_only" or abs(base_pitch) <= 1.0
            if not should_video:
                if worldmirror_sequence_mode != "prepend_source_highres":
                    print(
                        "[WorldStereo] video_view_filter=zero_pitch_only requires "
                        "worldmirror_sequence_mode=prepend_source_highres for skipped views; passing source anchor."
                    )
                print(
                    "[WorldStereo] Skipping stereo video for non-zero pitch view: "
                    f"slice={b + 1}/{B}, pitch={base_pitch:.2f}°"
                )
                all_video_frames.append(source_frame.unsqueeze(0).contiguous())
                all_camera_poses.append(base_pose.cpu().float().unsqueeze(0))
                all_camera_intrinsics.append(base_K_source.cpu().float().unsqueeze(0))
                continue

            # Compose absolute trajectory camera-to-world (c2w) matrices: T_abs = T_base * T_trajectory
            base_pose_dev = base_pose.to(c2ws.device, dtype=c2ws.dtype)
            # Use explicit batch expansion and bmm to avoid batch broadcasting issues in PyTorch
            base_pose_expanded = base_pose_dev.unsqueeze(0).expand(N, -1, -1)  # [N, 4, 4]
            c2ws_abs = torch.bmm(base_pose_expanded, c2ws)                     # [N, 4, 4]
            # Expand generation-resolution intrinsics across all frames: [N, 3, 3]
            intrs_abs = base_K_dev.unsqueeze(0).expand(N, -1, -1).clone()
            conditioning_frame_indices = None
            if conditioning_frame_mode == "keyframes":
                conditioning_frame_indices = _worldstereo_keyframe_indices(
                    N,
                    device=c2ws_abs.device,
                )

            # ── Preprocess Pipeline Inputs for the Slice ──
            print(f"[WorldStereo] Slice {b + 1} processing: depth estimation + point rendering...")
            with _temporary_worldstereo_runtime_patches():
                pipeline_inputs = _prepare_pipeline_inputs(
                    image_pil=img_pil,
                    c2ws=c2ws_abs,
                    intrs=intrs_abs,
                    moge_model=moge_model,
                    device=device,
                    width=W,
                    height=H,
                    conditioning_frame_indices=conditioning_frame_indices,
                )
            if device == "cuda":
                try:
                    moge_model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pipeline_inputs = _move_pipeline_inputs(pipeline_inputs, "cpu")
                except Exception as e:
                    print(f"[WorldStereo] MoGe CPU offload skipped ({type(e).__name__}: {e})")

            cached_image_embeds = None
            if cache_conditioning:
                image_key = _image_cache_key(img_pil)
                cached_image_embeds = image_embeds_cache.get(image_key)
                if cached_image_embeds is None:
                    cached_image_embeds = _encode_image_cache(pipeline, img_pil, device).to("cpu")
                    image_embeds_cache[image_key] = cached_image_embeds
                    print(f"[WorldStereo] Slice {b + 1} image embedding cached")
                else:
                    print(f"[WorldStereo] Slice {b + 1} image embedding reused")

            # ── Generator Setup ──
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=device).manual_seed(seed + b)

            # ── Diffusion Generation ──
            print(f"[WorldStereo] Slice {b + 1} executing inference for {N} camera frames...")
            autocast_device = torch.device(device).type
            with torch.autocast(autocast_device, dtype=_worldstereo_half_dtype()):
                if device == "cuda":
                    pipeline_inputs = _move_pipeline_inputs(pipeline_inputs, device)
                if render_vae_mode == "keyframes":
                    render_video = pipeline_inputs.get("render_video")
                    keyframe_indices = _worldstereo_keyframe_indices(
                        N,
                        device=render_video.device,
                    ) if isinstance(render_video, torch.Tensor) else None
                    if isinstance(render_video, torch.Tensor) and render_video.shape[2] != keyframe_indices.numel():
                        pipeline_inputs["render_video"] = render_video.index_select(2, keyframe_indices).contiguous()
                        render_mask = pipeline_inputs.get("render_mask")
                        if isinstance(render_mask, torch.Tensor) and render_mask.shape[2] == render_video.shape[2]:
                            pipeline_inputs["render_mask"] = render_mask.index_select(2, keyframe_indices.to(render_mask.device)).contiguous()
                        camera_embedding = pipeline_inputs.get("camera_embedding")
                        if isinstance(camera_embedding, torch.Tensor) and camera_embedding.shape[2] == render_video.shape[2]:
                            pipeline_inputs["camera_embedding"] = camera_embedding.index_select(2, keyframe_indices.to(camera_embedding.device)).contiguous()
                        print(
                            "[WorldStereo] Render VAE conditioning sliced to keyframes: "
                            f"{render_video.shape[2]} -> {pipeline_inputs['render_video'].shape[2]}"
                        )
                pipeline_kwargs = {
                    **pipeline_inputs,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                    "output_type": "pt",
                    "latent_cond_mode": latent_condition_mode,
                }
                if "dmd" in model_type:
                    pipeline_kwargs["mode"] = "test"
                if cache_conditioning:
                    pipeline_kwargs.update({
                        "prompt": None,
                        "negative_prompt": None,
                        "prompt_embeds": cached_prompt_embeds.to(device),
                        "negative_prompt_embeds": cached_negative_prompt_embeds.to(device) if cached_negative_prompt_embeds is not None else None,
                        "image_embeds": cached_image_embeds.to(device),
                    })
                else:
                    pipeline_kwargs.update({
                        "prompt": prompt if prompt else "",
                        "negative_prompt": negative_prompt if negative_prompt else None,
                    })
                with _temporary_worldstereo_runtime_patches():
                    with _pipeline_conditioning_cache(pipeline, cache_conditioning):
                        if cache_conditioning:
                            pipeline._vnccs_latent_condition_cache = latent_condition_cache
                        output = pipeline(**pipeline_kwargs)
                del pipeline_kwargs
                del pipeline_inputs

            # Clean cache immediately to keep low VRAM overhead on sequential slices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Extract Generated Frames ──
            frames = output.frames[0].float().cpu().clamp(0.0, 1.0)  # [N, 3, H, W]
            output_frame_count = int(frames.shape[0])
            output_frame_indices = _worldstereo_output_frame_indices(
                N,
                output_frame_count,
                device=c2ws_abs.device,
            )
            if output_frame_count != N:
                print(
                    "[WorldStereo] Aligning camera trajectory to decoded frames: "
                    f"conditioning={N}, decoded={output_frame_count}, "
                    f"indices={output_frame_indices.detach().cpu().tolist()}"
                )
            slice_video_frames = frames.permute(0, 2, 3, 1)          # [N, H, W, 3]
            slice_poses = c2ws_abs.index_select(0, output_frame_indices).cpu().float()
            slice_intrs = intrs_abs.index_select(0, output_frame_indices).cpu().float()
            slice_video_frames, slice_poses, slice_intrs, dropped_first = _drop_duplicate_first_frame(
                slice_video_frames,
                slice_poses,
                slice_intrs,
                img_pil,
            )
            if worldmirror_sequence_mode == "prepend_source_highres" and not dropped_first and slice_video_frames.shape[0] > 1:
                print("[WorldStereo] WorldMirror sequence mode: forcing generated first-frame removal; high-res source frame will be prepended.")
                slice_video_frames = slice_video_frames[1:].contiguous()
                slice_poses = slice_poses[1:].contiguous()
                slice_intrs = slice_intrs[1:].contiguous()
            if crop_generated_edges:
                slice_video_frames, slice_intrs = _crop_generated_edges(
                    slice_video_frames,
                    slice_intrs,
                    edge_crop_percent,
                )
            if worldmirror_sequence_mode == "prepend_source_highres":
                slice_video_frames, slice_intrs = _resize_frames_and_intrinsics(
                    slice_video_frames,
                    slice_intrs,
                    source_w,
                    source_h,
                    upscale_mode=generated_upscale_mode,
                    seedvr2_options={
                        "seed": seedvr2_seed,
                        "resolution": seedvr2_resolution,
                        "max_resolution": seedvr2_max_resolution,
                    } if generated_upscale_mode == "seedvr2" else None,
                )
                anchor_frame = source_frame.unsqueeze(0).to(slice_video_frames.dtype)
                anchor_pose = base_pose.cpu().float().unsqueeze(0)
                anchor_intr = base_K_source.cpu().float().unsqueeze(0)
                slice_video_frames = torch.cat([anchor_frame, slice_video_frames], dim=0)
                slice_poses = torch.cat([anchor_pose, slice_poses], dim=0)
                slice_intrs = torch.cat([anchor_intr, slice_intrs], dim=0)
                print(
                    "[WorldStereo] WorldMirror sequence prepared: "
                    f"anchor={source_w}x{source_h}, generated_resized={slice_video_frames.shape[0] - 1} frames"
                )

            all_video_frames.append(slice_video_frames)
            all_camera_poses.append(slice_poses)
            all_camera_intrinsics.append(slice_intrs)

        # Concatenate sequences of all processed slices into unified tensors
        video_frames     = torch.cat(all_video_frames, dim=0)       # [B * N, H, W, 3]
        camera_poses_out = torch.cat(all_camera_poses, dim=0)       # [B * N, 4, 4]
        camera_intrs_out = torch.cat(all_camera_intrinsics, dim=0)  # [B * N, 3, 3]
        if camera_poses_out.shape[0] != video_frames.shape[0] or camera_intrs_out.shape[0] != video_frames.shape[0]:
            raise RuntimeError(
                "WorldStereo output frame/camera count mismatch: "
                f"frames={video_frames.shape[0]}, poses={camera_poses_out.shape[0]}, "
                f"intrinsics={camera_intrs_out.shape[0]}"
            )

        print(f"[WorldStereo Batch Loop] Complete. Total of {video_frames.shape[0]} frames prepared for 3D fusion.")

        return video_frames, camera_poses_out, camera_intrs_out


class VNCCS_ExportWorldStereoSingleModel:
    """Experimental exporter for single-file WorldStereo transformer checkpoints."""

    AUX_MODEL_TYPES = ["official-clip-umt5-encoder", "official-wan-vae"]
    MODEL_TYPES = VNCCS_LoadWorldStereoModel.MODEL_TYPES + AUX_MODEL_TYPES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "export_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute file path ending in .safetensors, or a directory. Can be outside ComfyUI.",
                    },
                ),
            },
            "optional": {
                "model_type": (cls.MODEL_TYPES, {"default": "worldstereo-camera"}),
                "precision": (["bf16", "fp8", "int4"], {
                    "default": "bf16",
                    "tooltip": "int4 is experimental and packs Linear weights to 4-bit with bf16/fp16 fallbacks.",
                }),
                "turbo_lora": (list(WORLDSTEREO_TURBO_LORAS.keys()), {"default": "none"}),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "cpu avoids VRAM pressure but needs a lot of RAM. cuda may be faster but risky.",
                }),
                "overwrite": ("BOOLEAN", {"default": False}),
                "write_metadata_json": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("export_info",)
    FUNCTION = "export_model"
    CATEGORY = "VNCCS/Video"

    def export_model(
        self,
        export_path,
        model_type="worldstereo-camera",
        precision="bf16",
        turbo_lora="none",
        lora_strength=1.0,
        device="cpu",
        overwrite=False,
        write_metadata_json=True,
    ):
        import gc
        from safetensors.torch import save_file

        if not export_path or not export_path.strip():
            raise ValueError("export_path is required and may point outside the ComfyUI output folder.")
        if device == "cuda" and not torch.cuda.is_available():
            print("[WorldStereo Export] CUDA requested but unavailable; using CPU")
            device = "cpu"
        if model_type == "official-clip-umt5-encoder" and precision == "int4":
            print("[WorldStereo Export] int4 is transformer-only; UMT5 export will use ComfyUI fp8.")
            precision = "fp8"
        if model_type == "official-wan-vae" and precision != "bf16":
            print("[WorldStereo Export] Single-file Wan VAE export uses bf16 normal weights.")
            precision = "bf16"

        output_file, metadata_file = _resolve_worldstereo_export_paths(
            export_path,
            model_type,
            precision,
            turbo_lora,
        )
        if os.path.exists(output_file) and not overwrite:
            raise FileExistsError(f"Export file already exists: {output_file}")
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        export_models_base = _worldstereo_export_cache_dir(output_file)
        os.makedirs(export_models_base, exist_ok=True)
        print(f"[WorldStereo Export] Using export-local model cache: {export_models_base}")

        if model_type in self.AUX_MODEL_TYPES:
            if turbo_lora != "none":
                print("[WorldStereo Export] turbo_lora ignored for auxiliary model export.")
            info = _export_worldstereo_aux_single_model(
                output_file=output_file,
                metadata_file=metadata_file,
                model_type=model_type,
                precision=precision,
                device=device,
                export_models_base=export_models_base,
                write_metadata_json=write_metadata_json,
            )
            return (info,)

        transformer_dir, base_model_dir, _ = _download_worldstereo_components(
            model_type,
            include_moge=False,
            worldstereo_models_base=export_models_base,
            aux_models_base=_get_models_base(),
        )
        transformer_dir = transformer_dir.replace("\\", "/")

        current_dir = os.path.dirname(os.path.dirname(__file__))
        with _temporary_worldstereo_runtime_patches(patch_diffusers=True):
            WorldStereo = _import_worldstereo_class(current_dir)
            print(
                f"[WorldStereo Export] Loading {model_type} for single-file export "
                f"(precision={precision}, turbo_lora={turbo_lora}) ..."
            )
            transformer_only = turbo_lora == "none"
            if not transformer_only:
                print("[WorldStereo Export] Turbo LoRA fusion requires full pipeline loading.")
            worldstereo = WorldStereo.from_pretrained(
                os.path.dirname(transformer_dir),
                subfolder=model_type,
                device=device,
                model_device=device,
                transformer_only=transformer_only,
            )

        pipeline = worldstereo.pipeline
        turbo_lora_path = _apply_worldstereo_turbo_lora(
            pipeline,
            turbo_lora,
            lora_strength,
            models_base=export_models_base,
        )
        export_precision = "bf16" if precision == "int4" else precision
        actual_precision = _apply_worldstereo_transformer_export_precision(pipeline, export_precision)

        transformer = pipeline.transformer
        int4_group_size = 128
        if precision == "int4":
            print(f"[WorldStereo Export] Packing Linear weights to int4 (group_size={int4_group_size}) ...")
            state, int4_params, int4_bytes, int4_modules = _int4_state_dict_for_safetensors(
                transformer,
                group_size=int4_group_size,
            )
            actual_precision = "int4"
        else:
            state = _tensor_state_dict_for_safetensors(transformer)
            int4_params = 0
            int4_bytes = 0
            int4_modules = 0
        total_params = sum(t.numel() for t in state.values())
        total_bytes = sum(t.numel() * t.element_size() for t in state.values())
        dtype_summary = {}
        for tensor in state.values():
            dtype_summary[str(tensor.dtype)] = dtype_summary.get(str(tensor.dtype), 0) + tensor.numel()

        metadata = {
            "format": "hyworld2_worldstereo_single_transformer_v1",
            "model_type": model_type,
            "precision": actual_precision,
            "runtime_dtype": str(_worldstereo_half_dtype()),
            "transformer_prepared": "true",
            "recommended_loader_precision": "auto",
            "int4_group_size": str(int4_group_size if precision == "int4" else 0),
            "int4_linear_modules": str(int4_modules),
            "int4_packed_params": str(int4_params),
            "int4_packed_bytes": str(int4_bytes),
            "turbo_lora": turbo_lora,
            "lora_strength": str(float(lora_strength)),
            "num_tensors": str(len(state)),
            "num_params": str(total_params),
            "tensor_bytes": str(total_bytes),
            "dtype_summary": json.dumps(dtype_summary, sort_keys=True),
            "note": "Portable single WorldStereo transformer. T5/CLIP/VAE/MoGe are resolved by the loader and are not included.",
        }

        print(f"[WorldStereo Export] Saving single transformer checkpoint: {output_file}")
        save_file(state, output_file, metadata=metadata)
        if write_metadata_json:
            with open(metadata_file, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)

        del state
        del pipeline
        del worldstereo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        info = (
            f"exported={output_file}\n"
            f"metadata={metadata_file if write_metadata_json else ''}\n"
            f"precision={actual_precision}\n"
            f"turbo_lora={turbo_lora}\n"
            f"tensors={metadata['num_tensors']}\n"
            f"params={metadata['num_params']}\n"
            f"bytes={metadata['tensor_bytes']}\n"
            f"dtypes={metadata['dtype_summary']}"
        )
        print(f"[WorldStereo Export] Done:\n{info}")
        return (info,)


def _run_dependency_command(cmd, cwd=PROJECT_ROOT, env=None, label="command"):
    import subprocess

    print(f"[HYWorld2 Installer] Running {label}: {' '.join(cmd)}")
    process_env = os.environ.copy()
    process_env.setdefault("PYTHONUTF8", "1")
    process_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    if env:
        process_env.update(env)

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        print(f"[HYWorld2 Installer] {line}")
        lines.append(line)
        if len(lines) > 400:
            lines = lines[-400:]
    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(lines[-80:])
        raise RuntimeError(
            f"{label} failed with exit code {return_code}.\n"
            f"Last installer output:\n{tail}"
        )
    return "\n".join(lines[-120:])


def _write_filtered_requirements(source_path: str) -> str:
    import tempfile

    skip_packages = {"torch", "torchvision", "torchaudio"}
    filtered_lines = []
    with open(source_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            package_name = stripped.split(";", 1)[0].split("[", 1)[0]
            package_name = package_name.replace("==", ">=").replace("~=", ">=").replace("<=", ">=").replace(">", ">=").split(">=", 1)[0]
            if package_name.lower() in skip_packages:
                filtered_lines.append(f"# skipped by HYWorld2 installer: {line}")
            else:
                filtered_lines.append(line)

    tmp = tempfile.NamedTemporaryFile("w", suffix="_hyworld2_requirements.txt", delete=False, encoding="utf-8")
    try:
        tmp.writelines(filtered_lines)
        return tmp.name
    finally:
        tmp.close()


class VNCCS_InstallHYWorld2Dependencies:
    """One-shot dependency installer for HYWorld2 experimental dependencies."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "install": (["RUN_DEPENDENCY_INSTALL"], {
                    "default": "RUN_DEPENDENCY_INSTALL",
                    "tooltip": "Queue this node to install HYWorld2 dependencies, gsplat, and PyTorch3D.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("install_log",)
    FUNCTION = "install_dependencies"
    CATEGORY = "VNCCS/Install"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return str(kwargs.get("install", "RUN_DEPENDENCY_INSTALL"))

    def install_dependencies(self, install="RUN_DEPENDENCY_INSTALL"):
        import sys

        requirements_path = os.path.join(PROJECT_ROOT, "requirements.txt")
        build_script = os.path.join(PROJECT_ROOT, "scripts", "build_gsplat.py")

        if not os.path.exists(requirements_path):
            raise FileNotFoundError(f"requirements.txt not found: {requirements_path}")
        if not os.path.exists(build_script):
            raise FileNotFoundError(f"build_gsplat.py not found: {build_script}")

        print("[HYWorld2 Installer] Starting dependency installation.")
        filtered_requirements = _write_filtered_requirements(requirements_path)
        try:
            requirements_log = _run_dependency_command(
                [sys.executable, "-m", "pip", "install", "-r", filtered_requirements],
                cwd=PROJECT_ROOT,
                label="pip install filtered requirements.txt",
            )
        finally:
            try:
                os.remove(filtered_requirements)
            except OSError:
                pass
        gsplat_log = _run_dependency_command(
            [sys.executable, build_script],
            cwd=PROJECT_ROOT,
            label="build_gsplat.py (gsplat + PyTorch3D)",
        )

        summary = (
            "HYWorld2 dependency installation completed.\n\n"
            "Last requirements output:\n"
            f"{requirements_log}\n\n"
            "Last gsplat/PyTorch3D output:\n"
            f"{gsplat_log}"
        )
        print("[HYWorld2 Installer] Dependency installation completed.")
        return (summary,)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VNCCS_InstallHYWorld2Dependencies": VNCCS_InstallHYWorld2Dependencies,
    "VNCCS_LoadWorldStereoModel":    VNCCS_LoadWorldStereoModel,
    "VNCCS_LoadWorldStereoLightModel": VNCCS_LoadWorldStereoLightModel,
    "VNCCS_CameraTrajectoryBuilder": VNCCS_CameraTrajectoryBuilder,
    "VNCCS_WorldStereoGenerate":     VNCCS_WorldStereoGenerate,
    "VNCCS_ExportWorldStereoSingleModel": VNCCS_ExportWorldStereoSingleModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_InstallHYWorld2Dependencies": "Install HYWorld2 Dependencies",
    "VNCCS_LoadWorldStereoModel":    "Load WorldStereo Model",
    "VNCCS_LoadWorldStereoLightModel": "Load WorldStereoLight Model",
    "VNCCS_CameraTrajectoryBuilder": "Camera Trajectory Builder",
    "VNCCS_WorldStereoGenerate":     "WorldStereo Generate",
    "VNCCS_ExportWorldStereoSingleModel": "Export WorldStereo Single Model",
}
