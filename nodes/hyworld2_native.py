import contextlib
import gc
import hashlib
import json
import os
import re
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_model_management = None
import torch.nn.functional as F
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLDGEN_DIR = PROJECT_ROOT / "hyworld2" / "worldgen"

HYWORLD2_QWENVL_MODELS = {
    "Qwen3-VL-2B-Instruct": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct", "quantized": False},
    "Qwen3-VL-2B-Thinking": {"repo_id": "Qwen/Qwen3-VL-2B-Thinking", "quantized": False},
    "Qwen3-VL-2B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-2B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-2B-Thinking-FP8", "quantized": True},
    "Qwen3-VL-4B-Instruct": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct", "quantized": False},
    "Qwen3-VL-4B-Thinking": {"repo_id": "Qwen/Qwen3-VL-4B-Thinking", "quantized": False},
    "Qwen3-VL-4B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-4B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-4B-Thinking-FP8", "quantized": True},
    "Qwen3-VL-8B-Instruct": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct", "quantized": False},
    "Qwen3-VL-8B-Thinking": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking", "quantized": False},
    "Qwen3-VL-8B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-8B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking-FP8", "quantized": True},
    "Qwen3-VL-32B-Instruct": {"repo_id": "Qwen/Qwen3-VL-32B-Instruct", "quantized": False},
    "Qwen3-VL-32B-Thinking": {"repo_id": "Qwen/Qwen3-VL-32B-Thinking", "quantized": False},
    "Qwen3-VL-32B-Instruct-FP8": {"repo_id": "Qwen/Qwen3-VL-32B-Instruct-FP8", "quantized": True},
    "Qwen3-VL-32B-Thinking-FP8": {"repo_id": "Qwen/Qwen3-VL-32B-Thinking-FP8", "quantized": True},
    "Qwen2.5-VL-3B-Instruct": {"repo_id": "Qwen/Qwen2.5-VL-3B-Instruct", "quantized": False},
    "Qwen2.5-VL-7B-Instruct": {"repo_id": "Qwen/Qwen2.5-VL-7B-Instruct", "quantized": False},
}
HYWORLD2_QWENVL_DEFAULT = "Qwen3-VL-4B-Instruct"
HYWORLD2_QWENVL_DEFAULT_QUANTIZATION = "4-bit (VRAM-friendly)"
HYWORLD2_QWENVL_QUANTIZATION = ["None (FP16)", "8-bit (Balanced)", "4-bit (VRAM-friendly)"]
HYWORLD2_QWENVL_ATTENTION = ["auto", "sage", "flash_attention_2", "sdpa"]
HYWORLD2_QWENVL_MAX_IMAGE_EDGE = 768
HYWORLD2_SAM3_REPO_ID = "MIUProject/sam3"


def _ensure_worldgen_path():
    for path in (str(PROJECT_ROOT), str(WORLDGEN_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _output_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return PROJECT_ROOT / "output"


def _qwenvl_model_names():
    return list(HYWORLD2_QWENVL_MODELS.keys())


def _qwenvl_repo_id(model_name):
    name = str(model_name or "").strip()
    if name in HYWORLD2_QWENVL_MODELS:
        return HYWORLD2_QWENVL_MODELS[name]["repo_id"]
    if "/" in name:
        return name
    raise ValueError(f"Unsupported QwenVL model: {model_name}")


def _qwenvl_is_fp8(model_name):
    name = str(model_name or "")
    info = HYWORLD2_QWENVL_MODELS.get(name, {})
    return bool(info.get("quantized")) or "-fp8" in name.lower() or "_fp8" in name.lower()


def _qwenvl_models_dir():
    if folder_paths is not None:
        try:
            llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
            if llm_paths:
                return Path(llm_paths[0]) / "Qwen-VL"
        except Exception:
            pass
        try:
            return Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        except Exception:
            pass
    return PROJECT_ROOT / "models" / "LLM" / "Qwen-VL"


def _qwenvl_ensure_model(model_name):
    repo_id = _qwenvl_repo_id(model_name)
    target = _qwenvl_models_dir() / repo_id.split("/")[-1]
    if target.exists() and target.is_dir():
        if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
            print(f"[HYWorld2 QwenVL] Using local model '{model_name}' from {target}")
            return str(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ImportError("QwenVL auto-download requires huggingface_hub.") from exc
    print(f"[HYWorld2 QwenVL] Downloading model '{model_name}' from {repo_id} to {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=["*.md", ".git*"],
    )
    print(f"[HYWorld2 QwenVL] Model ready: {target}")
    return str(target)


def _qwenvl_normalize_device(device):
    device = str(device or "auto").strip()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device.isdigit():
        device = f"cuda:{int(device)}"
    if device == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
        except Exception:
            idx = 0
        if idx >= torch.cuda.device_count():
            idx = 0
        return f"cuda:{idx}"
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return "cpu"
    return device


def _qwenvl_flash_attn_available():
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def _qwenvl_sage_attn_available():
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False
        import sageattention  # noqa: F401
    except Exception:
        return False
    return True


def _qwenvl_resolve_attention(attention_mode, force_sdpa=False):
    mode = str(attention_mode or "auto")
    if force_sdpa or mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        return "flash_attention_2" if _qwenvl_flash_attn_available() else "sdpa"
    if mode == "sage":
        # We expose the selector, but only use kernels when the installed transformers stack supports it.
        return "sdpa" if not _qwenvl_sage_attn_available() else "sdpa"
    if _qwenvl_flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def _qwenvl_quantization_config(model_name, quantization, cpu_offload=False):
    if _qwenvl_is_fp8(model_name):
        return None, None, True
    quant = str(quantization or "None (FP16)")
    if quant == "4-bit (VRAM-friendly)":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise ImportError("QwenVL 4-bit quantization requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ), None, False
    if quant == "8-bit (Balanced)":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise ImportError("QwenVL 8-bit quantization requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=bool(cpu_offload)), None, False
    return None, torch.float16 if torch.cuda.is_available() else torch.float32, False


def _qwenvl_auto_max_memory():
    if not torch.cuda.is_available():
        return None
    max_memory = {}
    for idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        gpu_limit = max(1, int(max(1.0, total_gib - 3.0)))
        max_memory[idx] = f"{gpu_limit}GiB"
    max_memory["cpu"] = "64GiB"
    return max_memory


def _qwenvl_preview_image(image, max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE):
    if not isinstance(image, Image.Image):
        return image
    width, height = image.size
    max_edge = max(width, height)
    limit = max(64, int(max_image_edge or HYWORLD2_QWENVL_MAX_IMAGE_EDGE))
    if max_edge <= limit:
        return image.convert("RGB")
    scale = limit / float(max_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    resized = image.convert("RGB").resize(new_size, resampling)
    print(f"[HYWorld2 QwenVL] Resized VLM preview {width}x{height} -> {new_size[0]}x{new_size[1]}")
    return resized


def _sanitize_name(value, fallback="scene"):
    import re

    base = os.path.basename(str(value or fallback).replace("\\", "/"))
    base = os.path.splitext(base)[0]
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" ._")
    return base or fallback


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hyworld2_missing_memory_prerequisites(scene):
    scene = Path(scene)
    render_root = scene / "render_results"
    required_files = [
        scene / "meta_info.json",
        render_root / "global_pcd.ply",
        render_root / "sky_mask.png",
        render_root / "full_depth_prediction.pt",
        render_root / "pano_bank" / "cameras.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    pano_images = sorted((render_root / "pano_bank" / "images").glob("*.png"))
    pano_depths = sorted((render_root / "pano_bank" / "depths").glob("*.png"))
    if not pano_images:
        missing.append(str(render_root / "pano_bank" / "images" / "*.png"))
    if not pano_depths:
        missing.append(str(render_root / "pano_bank" / "depths" / "*.png"))
    return missing


def _hyworld2_file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"sha256": digest.hexdigest(), "size": stat.st_size}


def _hyworld2_trajectory_state_path(scene):
    return Path(scene) / "hyworld2_trajectories_state.json"


def _hyworld2_world_expansion_state_path(scene):
    return Path(scene) / "hyworld2_world_expansion_state.json"


def _hyworld2_klein_expansion_state_path(scene):
    return Path(scene) / "hyworld2_klein_expansion_state.json"


def _hyworld2_read_json_file(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _hyworld2_write_json_file(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _hyworld2_pil_pixel_fingerprint(image):
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.size[0]}x{image.size[1]}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return {"sha256": digest.hexdigest(), "width": image.size[0], "height": image.size[1], "mode": "RGB"}


def _hyworld2_image_file_pixel_fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return _hyworld2_pil_pixel_fingerprint(image)


def _hyworld2_image_tensor_fingerprint(image_tensor):
    if image_tensor is None:
        return None
    tensor = image_tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return {"sha256": digest.hexdigest(), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def _hyworld2_workspace_state_path(scene):
    return Path(scene) / "hyworld2_workspace_state.json"


def _hyworld2_clear_workspace_derivatives(scene):
    scene = Path(scene).resolve()
    render_root = scene / "render_results"
    if render_root.exists():
        resolved = render_root.resolve()
        if scene not in resolved.parents:
            raise RuntimeError(f"Refusing to clear unexpected render_results directory: {render_root}")
        shutil.rmtree(render_root)
    for path in (
        scene / "hyworld2_trajectories_state.json",
        scene / "hyworld2_qwenvl_scene.json",
        scene / "objects.json",
        scene / "detail_objects.json",
        scene / "meta_info.json",
        scene / "hyworld2_world_expansion_state.json",
    ):
        if path.exists():
            path.unlink()


def _hyworld2_trajectory_dirs(render_root):
    render_root = Path(render_root)
    if not render_root.is_dir():
        return []
    return sorted(path.parent for path in render_root.glob("**/render.mp4") if path.is_file())


def _hyworld2_trajectory_cache_status(
    scene,
    settings_signature,
    require_nav=False,
    require_detail=False,
    require_anchor=False,
    anchor_topk=0,
    workspace_cache_action=None,
):
    scene = Path(scene)
    render_root = scene / "render_results"
    state = _hyworld2_read_json_file(_hyworld2_trajectory_state_path(scene), default={}) or {}
    current_pano = _hyworld2_image_file_pixel_fingerprint(scene / "panorama.png")
    legacy_current_pano = _hyworld2_file_sha256(scene / "panorama.png")
    cached_pano = state.get("panorama")
    if not current_pano:
        return False, "panorama.png is missing", state
    workspace_says_unchanged = str(workspace_cache_action or "") == "panorama_unchanged"
    pano_matches = cached_pano in (current_pano, legacy_current_pano)
    if not pano_matches and not workspace_says_unchanged:
        return False, "panorama cache mismatch", state
    state_signature = state.get("settings_signature")
    repair_state = False
    if state_signature != settings_signature:
        # An unchanged panorama only validates geometry derived from the source
        # image. It must never bless cameras/renders produced with different
        # trajectory settings (resolution, FOV, frame count, anchor options...).
        return False, "trajectory settings changed", state
    if not pano_matches and workspace_says_unchanged:
        repair_state = True
    if repair_state:
        state["_repair_state"] = True

    missing_geometry = _hyworld2_missing_memory_prerequisites(scene)
    if missing_geometry:
        return False, "base geometry is incomplete", state

    trajectory_dirs = _hyworld2_trajectory_dirs(render_root)
    if not trajectory_dirs:
        return False, "no rendered trajectories found", state

    if require_nav:
        nav_dirs = [
            path for path in trajectory_dirs
            if any(part.startswith("target_") or part.startswith("reconstruct_") for part in path.parts)
        ]
        if not (scene / "objects.json").is_file():
            return False, "object navigation requested but objects.json is missing", state
        if not nav_dirs:
            return False, "object navigation requested but no object trajectory renders were found", state

    if require_detail:
        if not (scene / "detail_objects.json").is_file():
            return False, "extreme detail trajectories requested but detail_objects.json is missing", state
        detail_targets = []
        target_path = scene / "camera_trajectory" / "target_camera.json"
        if target_path.is_file():
            try:
                target_data = _hyworld2_read_json_file(target_path, default=[]) or []
                detail_targets = [item for item in target_data if isinstance(item, dict) and item.get("detail_pass")]
            except Exception:
                detail_targets = []
        if not detail_targets:
            return False, "extreme detail trajectories requested but no detail targets were found", state

    if require_anchor and int(anchor_topk) > 0:
        anchor_dirs = [path for path in trajectory_dirs if any(part.startswith("wonder_scan_") for part in path.parts)]
        if len(anchor_dirs) < int(anchor_topk):
            return False, f"anchor scan requested but only {len(anchor_dirs)}/{int(anchor_topk)} scan renders were found", state

    return True, "cached trajectory workspace is complete", state


def _release_model_memory(label="HYWorld2"):
    gc.collect()
    if comfy_model_management is not None:
        try:
            comfy_model_management.unload_all_models()
            comfy_model_management.cleanup_models_gc()
            comfy_model_management.soft_empty_cache(force=True)
        except Exception as exc:
            print(f"[{label}] Comfy model memory cleanup skipped ({type(exc).__name__}: {exc})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


class _SingleProcessDist:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_rank():
        return 0

    @staticmethod
    def get_world_size():
        return 1

    @staticmethod
    def barrier(*args, **kwargs):
        return None

    @staticmethod
    def all_gather_object(object_list, obj, *args, **kwargs):
        if object_list:
            object_list[0] = obj
        return None


def _ensure_single_process_dist(bank=None):
    if dist.is_available() and dist.is_initialized():
        return
    shim = _SingleProcessDist()
    module_names = {"hyworld2.worldgen.src.retrieval_wm", "src.retrieval_wm"}
    if bank is not None:
        module_names.add(bank.__class__.__module__)
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "dist"):
            module.dist = shim


def _reset_dir(path, label="directory"):
    import shutil

    path = Path(path)
    resolved = path.resolve()
    if resolved == resolved.anchor:
        raise ValueError(f"Refusing to reset drive/root {label}: {resolved}")
    protected = {PROJECT_ROOT.resolve(), WORLDGEN_DIR.resolve(), Path.home().resolve()}
    if resolved in protected:
        raise ValueError(f"Refusing to reset protected {label}: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _image_tensor_to_pil_list(images):
    if not isinstance(images, torch.Tensor):
        return []
    tensor = images.detach().cpu().float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
        tensor = tensor.permute(0, 2, 3, 1)
    result = []
    for frame in tensor:
        arr = (frame[..., :3].clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
        result.append(Image.fromarray(arr))
    return result


def _pil_list_to_image_tensor(images):
    frames = []
    for image in images:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr))
    if not frames:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)
    return torch.stack(frames, dim=0).contiguous()


def _save_rgb_image(path, image):
    arr = (image.detach().cpu().float().clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(path)


def _depth_tensor_to_numpy(depth_maps):
    if not isinstance(depth_maps, torch.Tensor):
        return []
    depth = depth_maps.detach().cpu().float()
    if depth.dim() == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.dim() == 4 and depth.shape[0] == 1 and depth.shape[-1] not in (1, 3, 4):
        depth = depth[0]
    if depth.dim() == 4 and depth.shape[-1] in (1, 3, 4):
        depth = depth[..., 0]
    elif depth.dim() == 4 and depth.shape[1] in (1, 3, 4):
        depth = depth[:, 0]
    if depth.dim() != 3:
        return []
    return [np.nan_to_num(d.numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) for d in depth]


def _depth_maps_to_numpy(depth_maps):
    return _depth_tensor_to_numpy(depth_maps)


def _raw_worldmirror_depths_to_numpy(raw_splats):
    if not isinstance(raw_splats, dict):
        return [], ""
    for key in ("gs_depth", "depth"):
        depths = _depth_tensor_to_numpy(raw_splats.get(key))
        if depths:
            return depths, f"raw_splats.{key}"
    return [], ""


def _first_existing_ply_path(*values):
    for value in values:
        if isinstance(value, dict):
            nested = _first_existing_ply_path(
                value.get("ply_path"),
                value.get("path"),
                value.get("file"),
                value.get("filepath"),
                value.get("gaussian_ply"),
                value.get("points_ply"),
            )
            if nested:
                return nested
        elif isinstance(value, (str, os.PathLike)) and str(value).lower().endswith(".ply"):
            path = Path(value)
            if path.exists():
                return str(path)
    return ""


def _first_splat_tensor(splats, key, dim):
    if not isinstance(splats, dict):
        return None
    value = splats.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, torch.Tensor):
        return None
    value = value.detach().cpu().float()
    if value.dim() >= 2 and value.shape[0] == 1 and value.shape[-1] == dim:
        value = value[0]
    if value.shape[-1] != dim:
        return None
    return value.reshape(-1, dim)


def _splat_tensor_any(splats, keys, dim):
    for key in keys:
        tensor = _first_splat_tensor(splats, key, dim)
        if tensor is not None:
            return tensor
    return None


def _normalize_bypass_points(points):
    if not isinstance(points, torch.Tensor):
        return None
    points = points.detach().cpu().float()
    if points.dim() == 5 and points.shape[0] == 1:
        points = points[0]
    if points.dim() == 4 and points.shape[-1] == 3:
        return points
    if points.dim() == 2 and points.shape[-1] == 3:
        return points
    return None


def _normalize_bypass_images(images):
    if not isinstance(images, torch.Tensor):
        return None
    images = images.detach().cpu().float()
    if images.dim() == 5 and images.shape[0] == 1:
        images = images[0]
    if images.dim() == 4 and images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = images.permute(0, 2, 3, 1)
    if images.dim() == 4 and images.shape[-1] in (1, 3, 4):
        return images[..., :3]
    return None


def _bypass_splat_points_and_colors(splats):
    means = _splat_tensor_any(splats, ("means", "xyz", "positions"), 3)
    if means is None:
        return None, None

    colors = _splat_tensor_any(splats, ("colors", "rgb", "rgbs"), 3)
    if colors is None:
        sh = _splat_tensor_any(splats, ("sh", "features_dc"), 3)
        if sh is not None:
            sh_c0 = 0.28209479177387814
            colors = (0.5 + sh_c0 * sh).clamp(0.0, 1.0)
    if colors is None or colors.shape[0] not in (1, means.shape[0]):
        colors = torch.full_like(means, 0.5)
    elif colors.shape[0] == 1 and means.shape[0] > 1:
        colors = colors.repeat(means.shape[0], 1)

    finite = torch.isfinite(means).all(dim=1)
    means = means[finite]
    colors = colors[finite]
    if means.numel() == 0:
        return None, None
    colors_u8 = (colors.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return means.numpy().astype(np.float32), colors_u8


def _bypass_points_and_colors(ply_data, raw_splats=None):
    for container in (ply_data, raw_splats):
        if isinstance(container, dict):
            points, colors = _bypass_splat_points_and_colors(container.get("splats"))
            if points is not None and colors is not None:
                return points, colors

    sources = []
    if isinstance(ply_data, dict):
        sources.extend(
            [
                ply_data.get("pts3d_filtered"),
                ply_data.get("pts3d"),
                ply_data.get("model_pts3d_filtered"),
                ply_data.get("model_pts3d"),
            ]
        )
    if isinstance(raw_splats, dict):
        sources.extend([raw_splats.get("pts3d_filtered"), raw_splats.get("pts3d")])

    points = None
    for source in sources:
        points = _normalize_bypass_points(source)
        if points is not None:
            break
    if points is None:
        return None, None

    images = None
    if isinstance(ply_data, dict):
        images = _normalize_bypass_images(ply_data.get("images"))
    if images is None and isinstance(raw_splats, dict):
        images = _normalize_bypass_images(raw_splats.get("images"))

    if points.dim() == 4:
        flat_points = points.reshape(-1, 3)
        if images is not None and images.shape[0] == points.shape[0] and images.shape[1:3] == points.shape[1:3]:
            flat_colors = images.reshape(-1, 3)
        else:
            flat_colors = torch.full_like(flat_points, 0.5)
    else:
        flat_points = points.reshape(-1, 3)
        flat_colors = torch.full_like(flat_points, 0.5)

    finite = torch.isfinite(flat_points).all(dim=1)
    flat_points = flat_points[finite]
    flat_colors = flat_colors[finite]
    if flat_points.numel() == 0:
        return None, None
    colors_u8 = (flat_colors.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return flat_points.numpy().astype(np.float32), colors_u8


def _write_bypass_point_ply(path, points, colors):
    from hyworld2.worldrecon.hyworldmirror.utils.save_utils import save_points_ply

    save_points_ply(Path(path), points, colors)
    return str(path)


def _export_bypass_memory_bank_pcds(bank, ply_data, raw_splats, downsampled_pts):
    points, colors = _bypass_points_and_colors(ply_data, raw_splats)
    if points is None or colors is None:
        raise ValueError(
            "HYWorld2 Memory Alignment bypass could not build aligned_pcd.ply: "
            "no Gaussian means or point geometry found in ply_data/raw_splats."
        )

    export_dir = Path(bank.root_path) / "render_results" / bank.results_path
    _ensure_dir(export_dir)
    bank.global_points = {
        "worldmirror_bypass": {
            "points": points,
            "colors": colors,
        }
    }
    bank.export_pcd(str(export_dir), N_points=max(1, int(downsampled_pts)))
    return str(export_dir / "aligned_pcd.ply"), int(points.shape[0])


def _save_worldmirror_ply_data_for_bypass(ply_data, output_path, raw_splats=None):
    existing = _first_existing_ply_path(ply_data, raw_splats)
    if existing:
        return existing

    if not isinstance(ply_data, dict):
        raise ValueError("HYWorld2 Memory Alignment bypass requires the WorldMirror PLY_DATA output connected to ply_data.")

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    splats = ply_data.get("splats")
    if not isinstance(splats, dict) and isinstance(raw_splats, dict):
        splats = raw_splats.get("splats")
    means = _splat_tensor_any(splats, ("means", "xyz", "positions"), 3)
    scales = _splat_tensor_any(splats, ("scales", "scale"), 3)
    quats = _splat_tensor_any(splats, ("quats", "rotations", "rotation", "rots"), 4)
    opacities = _splat_tensor_any(splats, ("opacities", "opacity"), 1)
    colors = _splat_tensor_any(splats, ("sh", "features_dc"), 3)
    if colors is None:
        colors = _splat_tensor_any(splats, ("colors", "rgb", "rgbs"), 3)
        if colors is not None:
            sh_c0 = 0.28209479177387814
            colors = (colors - 0.5) / sh_c0
    if colors is not None and means is not None and colors.shape[0] == 1 and means.shape[0] > 1:
        colors = colors.repeat(means.shape[0], 1)

    if all(t is not None for t in (means, scales, quats, opacities, colors)):
        from hyworld2.worldrecon.hyworldmirror.utils.save_utils import _build_gs_ply_data

        count = min(means.shape[0], scales.shape[0], quats.shape[0], opacities.shape[0], colors.shape[0])
        ply = _build_gs_ply_data(
            means[:count],
            scales[:count].clamp_min(1e-8),
            quats[:count],
            colors[:count],
            opacities[:count].reshape(-1),
            quantile_threshold=1.0,
        )
        ply.write(str(output_path))
        return str(output_path)

    try:
        from .world_mirror_v1 import extract_splat_params
    except Exception:
        extract_splat_params = None
    if extract_splat_params is not None:
        params = extract_splat_params(ply_data)
        if params:
            from hyworld2.worldrecon.hyworldmirror.utils.save_utils import _build_gs_ply_data

            means, scales, quats, rgb, opacities = params
            sh_c0 = 0.28209479177387814
            colors = (rgb.detach().cpu().float() - 0.5) / sh_c0
            ply = _build_gs_ply_data(
                means.detach().cpu().float(),
                scales.detach().cpu().float().clamp_min(1e-8),
                quats.detach().cpu().float(),
                colors,
                opacities.detach().cpu().float().reshape(-1),
                quantile_threshold=1.0,
            )
            ply.write(str(output_path))
            return str(output_path)

    points, colors = _bypass_points_and_colors(ply_data, raw_splats)
    if points is not None and colors is not None:
        return _write_bypass_point_ply(output_path, points, colors)

    keys = sorted(str(k) for k in ply_data.keys())
    raw_keys = sorted(str(k) for k in raw_splats.keys()) if isinstance(raw_splats, dict) else []
    raise ValueError(
        "HYWorld2 Memory Alignment bypass could not find Gaussian splats or point geometry "
        f"in connected ply_data/raw_splats. ply_data keys={keys}, raw_splats keys={raw_keys}"
    )


def _to_c2w(poses):
    if not isinstance(poses, torch.Tensor):
        return torch.empty((0, 4, 4), dtype=torch.float32)
    poses = poses.detach().cpu().float()
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.dim() == 2:
        poses = poses.unsqueeze(0)
    if poses.shape[-2:] == (3, 4):
        bottom = torch.tensor([0, 0, 0, 1], dtype=poses.dtype).view(1, 1, 4).repeat(poses.shape[0], 1, 1)
        poses = torch.cat([poses, bottom], dim=1)
    return poses


def _to_intrinsics(intrs):
    if not isinstance(intrs, torch.Tensor):
        return torch.empty((0, 3, 3), dtype=torch.float32)
    intrs = intrs.detach().cpu().float()
    if intrs.dim() == 4 and intrs.shape[0] == 1:
        intrs = intrs[0]
    if intrs.dim() == 2:
        intrs = intrs.unsqueeze(0)
    return intrs


_WORLDSTEREO_TO_WORLDMIRROR_BASIS = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def _worldstereo_w2c_to_worldmirror_c2w(w2c):
    """Convert WorldStereo Z-up W2C cameras to WorldMirror panorama C2W poses."""
    c2w = torch.linalg.inv(w2c.detach().cpu().float())
    return _worldstereo_c2w_to_worldmirror_c2w(c2w)


def _worldstereo_c2w_to_worldmirror_c2w(c2w):
    """Convert WorldStereo/worldgen Z-up C2W cameras to WorldMirror panorama C2W poses."""
    c2w = c2w.detach().cpu().float()
    basis = _WORLDSTEREO_TO_WORLDMIRROR_BASIS.to(dtype=c2w.dtype)
    return basis @ c2w


def _normalize_c2w_poses_to_first(poses):
    """Match worldrecon.pipeline prior-camera normalization: inv(first_pose) @ pose."""
    if not isinstance(poses, torch.Tensor) or poses.numel() == 0:
        return poses
    work = poses.detach().cpu().float()
    squeeze_batch = False
    trim_3x4 = False
    if work.dim() == 4 and work.shape[0] == 1:
        work = work[0]
        squeeze_batch = True
    if work.dim() != 3 or work.shape[-2:] not in ((3, 4), (4, 4)):
        return poses
    if work.shape[-2:] == (3, 4):
        trim_3x4 = True
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=work.dtype).view(1, 1, 4).repeat(work.shape[0], 1, 1)
        work = torch.cat([work, bottom], dim=1)
    try:
        inv_first = torch.linalg.inv(work[0])
        normalized = inv_first.unsqueeze(0) @ work
    except Exception:
        return poses
    if trim_3x4:
        normalized = normalized[:, :3, :]
    if squeeze_batch:
        normalized = normalized.unsqueeze(0)
    return normalized.to(dtype=poses.dtype)


def _quat_wxyz_multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _convert_trainer_gaussian_ply_to_worldmirror_basis(ply_path):
    path = Path(ply_path)
    if not path.exists():
        return str(path)

    with open(path, "rb") as handle:
        header = b""
        vertex_count = None
        props = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header in {path}")
            header += line
            text = line.decode("ascii", "replace").strip()
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property"):
                parts = text.split()
                props.append((parts[1], parts[2]))
            elif text == "end_header":
                data_offset = handle.tell()
                break

    if vertex_count is None:
        raise ValueError(f"PLY has no vertex count: {path}")
    prop_names = [name for _, name in props]
    required_xyz = {"x", "y", "z"}
    if not required_xyz.issubset(prop_names):
        return str(path)

    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "uint": "<u4",
    }
    dtype = np.dtype([(name, type_map.get(kind, "<f4")) for kind, name in props])
    vertices = np.fromfile(path, dtype=dtype, count=vertex_count, offset=data_offset).copy()

    old_x = vertices["x"].copy()
    old_y = vertices["y"].copy()
    old_z = vertices["z"].copy()
    vertices["x"] = old_x
    vertices["y"] = -old_z
    vertices["z"] = old_y

    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if set(rot_names).issubset(prop_names):
        quats = np.stack([vertices[name] for name in rot_names], axis=-1).astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        quats[valid] = quats[valid] / norms[valid]
        basis_quat = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)
        quats[valid] = _quat_wxyz_multiply(basis_quat, quats[valid])
        for idx, name in enumerate(rot_names):
            vertices[name] = quats[:, idx]

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    os.replace(tmp_path, path)
    return str(path)


def _load_camera_tensors_from_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "cameras" in data:
        cameras = data["cameras"]
        order = data.get("camera_order") or sorted(cameras.keys())
        poses = []
        intrs = []
        for camera_id in order:
            entry = cameras[str(camera_id)]
            if "camera_pose" in entry:
                c2w = np.asarray(entry["camera_pose"], dtype=np.float32)
            else:
                c2w = np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32))
            poses.append(c2w)
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))
    else:
        poses = []
        intrs = []
        for camera_id in sorted(data.keys()):
            entry = data[camera_id]
            if not isinstance(entry, dict) or "extrinsic" not in entry or "intrinsic" not in entry:
                continue
            poses.append(np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32)))
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))
    if not poses:
        return torch.empty((0, 4, 4)), torch.empty((0, 3, 3))
    return torch.from_numpy(np.stack(poses)).float(), torch.from_numpy(np.stack(intrs)).float()


def _find_latest_ply(result_dir):
    result_dir = Path(result_dir)
    preferred = result_dir / "ply"
    search_roots = [preferred] if preferred.exists() else []
    search_roots.append(result_dir)

    point_clouds = []
    fallback = []
    seen = set()
    for root in search_roots:
        for path in root.rglob("*.ply"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            match = re.fullmatch(r"point_cloud_(\d+)\.ply", path.name)
            if match:
                point_clouds.append((int(match.group(1)), path.stat().st_mtime, path))
            else:
                fallback.append(path)

    if point_clouds:
        point_clouds.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return str(point_clouds[0][2])
    if not fallback:
        return ""
    fallback.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(fallback[0])


def _normal_tensor_is_usable(normals):
    if normals is None or normals.numel() == 0:
        return False
    sample = normals.float()
    if not torch.isfinite(sample).all():
        return False
    raw_min = float(sample.min().item())
    raw_max = float(sample.max().item())
    raw_std = float(sample.std().item())
    if raw_max <= 0.02 or raw_std <= 0.005:
        return False
    decoded = sample * 2.0 - 1.0
    lengths = decoded.norm(dim=-1)
    valid_ratio = float(((lengths > 0.25) & (lengths < 1.75)).float().mean().item())
    return raw_max > raw_min and valid_ratio > 0.05


def _has_valid_normal_files(data_dir, max_files=3):
    normals_dir = Path(data_dir) / "normals"
    if not normals_dir.exists():
        return False
    normal_files = sorted(normals_dir.glob("*.png"))[:max_files]
    if not normal_files:
        return False
    for path in normal_files:
        try:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            continue
        if _normal_tensor_is_usable(torch.from_numpy(arr)):
            return True
    return False


def _load_metric_depth16_preview(path):
    with Image.open(path) as depth_pil:
        arr = np.asarray(depth_pil)
    if arr.ndim != 2 or arr.dtype.itemsize < 2:
        return None
    depth = np.frombuffer(arr.astype(np.uint16, copy=False), dtype=np.float16).astype(np.float32)
    return depth.reshape(arr.shape)


def _has_valid_depth_files(data_dir, max_files=3):
    depths_dir = Path(data_dir) / "depths"
    if not depths_dir.exists():
        return False
    depth_files = sorted(depths_dir.glob("*.png"))[:max_files]
    if not depth_files:
        return False
    for path in depth_files:
        try:
            depth = _load_metric_depth16_preview(path)
        except Exception:
            continue
        if depth is None:
            continue
        finite = np.isfinite(depth)
        if float(finite.mean()) < 0.999:
            continue
        valid = depth[finite & (depth > 1e-4)]
        if valid.size == 0:
            continue
        vmax = float(valid.max())
        if vmax > 1e-3 and vmax < 1e6:
            return True
    return False


def _ensure_scene_type_meta(data_dir, scene_type="unknown"):
    data_dir = Path(data_dir)
    candidates = [data_dir.parent / "meta_info.json", data_dir / "meta_info.json"]
    target = None
    meta = {}
    for path in candidates:
        if path.exists():
            target = path
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                meta = {}
            break
    if target is None:
        target = data_dir / "meta_info.json"
    if not meta.get("scene_type"):
        meta["scene_type"] = scene_type
        _ensure_dir(target.parent)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
    return str(target)


def _parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).replace(",", " ").split() if str(v).strip()]


def _worldstereo_keyframe_indices(num_frames, device=None):
    keyframe_count = max(1, (int(num_frames) - 1) // 4 + 1)
    indices = torch.linspace(0, int(num_frames) - 1, keyframe_count, device=device).round().long()
    return torch.unique_consecutive(indices.clamp(0, int(num_frames) - 1))


def _select_evenly_spaced_indices(indices, count):
    values = [int(index) for index in indices]
    if int(count) <= 0 or int(count) >= len(values):
        return values
    if int(count) == 1:
        return [values[0]]
    pick = np.rint(np.linspace(0, len(values) - 1, int(count))).astype(np.int64)
    pick = np.unique(np.clip(pick, 0, len(values) - 1))
    return [values[int(index)] for index in pick]


def _coerce_worldstereo_ref_index(ref_index, max_ref_index):
    if ref_index is None:
        value = 0
    elif isinstance(ref_index, torch.Tensor):
        value = int(ref_index.flatten()[0].detach().cpu().item()) if ref_index.numel() > 0 else 0
    elif isinstance(ref_index, (list, tuple)):
        value = int(ref_index[0]) if ref_index else 0
    else:
        value = int(ref_index)
    max_ref_index = max(0, int(max_ref_index))
    if max_ref_index < 19:
        value = int(round(float(value) * (float(max_ref_index) / 19.0)))
    return max(0, min(value, max_ref_index))


def _slice_render_conditioning_to_keyframes(pipeline_kwargs):
    render_video = pipeline_kwargs.get("render_video")
    num_frames = int(pipeline_kwargs.get("num_frames") or 0)
    if not isinstance(render_video, torch.Tensor) or num_frames <= 0:
        return
    keyframe_indices = _worldstereo_keyframe_indices(num_frames, device=render_video.device)
    if render_video.shape[2] == keyframe_indices.numel():
        return
    old_frames = int(render_video.shape[2])
    pipeline_kwargs["render_video"] = render_video.index_select(2, keyframe_indices).contiguous()
    for key in ("render_mask", "camera_embedding"):
        value = pipeline_kwargs.get(key)
        if isinstance(value, torch.Tensor) and value.shape[2] == old_frames:
            pipeline_kwargs[key] = value.index_select(2, keyframe_indices.to(value.device)).contiguous()
    camera_qt = pipeline_kwargs.get("camera_qt")
    if isinstance(camera_qt, torch.Tensor) and camera_qt.shape[1] == old_frames:
        pipeline_kwargs["camera_qt"] = camera_qt.index_select(1, keyframe_indices.to(camera_qt.device)).contiguous()
    max_ref_index = max(0, keyframe_indices.numel() - 2)
    pipeline_kwargs["ref_index"] = _coerce_worldstereo_ref_index(pipeline_kwargs.get("ref_index"), max_ref_index)
    print(f"[HYWorld2] Render VAE conditioning sliced to keyframes: {old_frames} -> {pipeline_kwargs['render_video'].shape[2]}")


def _sample_camera_tensors_to_frame_count(w2cs, Ks, frame_count):
    frame_count = int(frame_count)
    if frame_count <= 0 or w2cs.shape[0] == frame_count:
        return w2cs, Ks
    indices = np.linspace(0, w2cs.shape[0] - 1, frame_count, dtype=int)
    indices = torch.as_tensor(indices, dtype=torch.long, device=w2cs.device)
    return w2cs.index_select(0, indices), Ks.index_select(0, indices)


def _load_video_frames(path):
    _ensure_worldgen_path()
    from hyworld2.worldgen.src.general_utils import load_video

    return load_video(str(path))


def _export_video(frames, path, fps=16):
    from diffusers.utils import export_to_video

    path = Path(path)
    _ensure_dir(path.parent)
    export_to_video(frames, str(path), fps=fps)


def _encode_prompt_cache(pipeline, prompt, negative_prompt, do_classifier_free_guidance, device):
    execution_device = getattr(pipeline, "_execution_device", None)
    if callable(execution_device):
        execution_device = execution_device()
    if execution_device is None:
        execution_device = device
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt if prompt else "",
            negative_prompt=negative_prompt if negative_prompt else None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=torch.device(execution_device),
        )
    prompt_embeds = prompt_embeds.detach().to("cpu")
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.detach().to("cpu")
    return prompt_embeds, negative_prompt_embeds


def _build_prompt_cache(worldstereo_model, workspace, render_list, model_type, device):
    prompt_cache = {}
    encoded_by_text = {}
    pipeline = worldstereo_model["pipeline"]
    cfg = getattr(pipeline, "cfg", None) or getattr(worldstereo_model.get("worldstereo"), "cfg", None)
    negative_prompt = ""
    if cfg is not None:
        negative_prompt = getattr(cfg, "negative_prompt", "") or ""
    do_cfg = model_type != "worldstereo-memory-dmd"
    render_root = Path(workspace["scene_dir"]) / "render_results"

    # T5 is deliberately excluded from Accelerate's sequential CPU-offload
    # hooks by the WorldStereo loader.  Load it once for the whole prompt batch;
    # moving/freeing pipeline hooks around every caption turns a short text
    # encoding stage into repeated layer-by-layer CPU/GPU transfers.
    text_encoder = getattr(pipeline, "text_encoder", None)
    if text_encoder is not None and hasattr(text_encoder, "to"):
        text_encoder.to(device)
    for render_path in render_list:
        parts = Path(render_path).parts
        view_id, traj_id = parts[-3], parts[-2]
        caption_path = render_root / view_id / traj_id / "traj_caption.json"
        if not caption_path.exists():
            raise FileNotFoundError(
                f"Missing {caption_path}. Run HYWorld2 QwenVL in trajectory_caption mode before World Expansion; fallback prompts are disabled."
            )
        with open(caption_path, "r", encoding="utf-8") as handle:
            prompt = json.load(handle).get("prompt", "")
        # Several trajectories commonly share the same caption (always true for
        # manual_caption).  Encoding by trajectory made sequential CPU offload
        # shuttle the entire T5 stack once per trajectory for identical output.
        cache_key = (str(prompt), str(negative_prompt), bool(do_cfg))
        if cache_key not in encoded_by_text:
            encoded_by_text[cache_key] = _encode_prompt_cache(
                pipeline,
                prompt,
                negative_prompt,
                do_classifier_free_guidance=do_cfg,
                device=device,
            )
        prompt_cache[(view_id, traj_id)] = encoded_by_text[cache_key]

    # Video generation receives prompt_embeds directly and does not need T5.
    # Keep the encoder object registered on the pipeline, but ensure its weights
    # no longer occupy VRAM after the cache has been built.
    if text_encoder is not None and hasattr(text_encoder, "to"):
        with contextlib.suppress(Exception):
            text_encoder.to("cpu")
    if hasattr(pipeline, "maybe_free_model_hooks"):
        with contextlib.suppress(Exception):
            pipeline.maybe_free_model_hooks()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _hy_log(
        "World Expansion",
        f"Prompt cache encoded {len(encoded_by_text)} unique prompt(s) for {len(prompt_cache)} trajectory(s); text encoder offloaded",
    )
    return prompt_cache


def _worldstereo_cfg(worldstereo_model):
    pipeline = worldstereo_model["pipeline"]
    cfg = getattr(worldstereo_model.get("worldstereo"), "cfg", None)
    if cfg is None:
        cfg = getattr(pipeline, "cfg", None)
    if cfg is None:
        raise RuntimeError("WORLDSTEREO_MODEL has no cfg; cannot run HYWorld2 memory mode.")
    return cfg


def _safe_json_dumps(value):
    def default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return list(obj.shape)
        return str(obj)

    return json.dumps(value, indent=2, default=default)


def _hy_log(node, message):
    print(f"[HYWorld2 {node}] {message}")


def _hy_cache_debug(node, stage, payload):
    text = _safe_json_dumps(payload)
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    compact = payload
    try:
        compact = json.loads(text)
    except Exception:
        pass
    print(f"[HYWorld2 Cache] {node} {stage}: hash={digest} payload={_safe_json_dumps(compact)}")


def _load_workspace_panorama(scene):
    image_path = scene / "panorama_sr.png"
    if not image_path.exists():
        image_path = scene / "panorama.png"
    if not image_path.exists():
        raise FileNotFoundError(f"HYWorld2 Trajectories requires panorama.png in workspace: {scene}")
    return Image.open(image_path).convert("RGB")


def _parse_scene_type(text):
    lowered = str(text or "").lower()
    if "outdoor" in lowered and "indoor" not in lowered:
        return "outdoor"
    if "indoor" in lowered:
        return "indoor"
    if "outdoor" in lowered:
        return "outdoor"
    return "indoor"


def _parse_qwenvl_objects(text):
    from hyworld2.worldgen.src.json_utils import loads_repaired

    raw = str(text or "").strip()
    try:
        parsed = loads_repaired(raw)
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = parsed.get("objects") or parsed.get("items") or []
        else:
            items = []
    except Exception:
        cleaned = raw.replace("[", "").replace("]", "").replace('"', "").replace("'", "").replace("```json", "").replace("```", "")
        items = []
        for line in cleaned.replace("\n", ",").split(","):
            item = line.strip(" -\t\r")
            if item:
                items.append(item)
    result = []
    seen = set()
    for item in items:
        item = str(item).strip().replace("-", "_")
        item = " ".join(item.split())
        if not item or len(item.split()) > 8:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _qwenvl_object_key(item):
    return " ".join(str(item or "").replace("-", "_").lower().strip().split())


def _ensure_trajectory_planner_context(
    workspace,
    scene_type,
    apply_nav_traj,
    apply_detail_traj,
    detail_object_limit,
    force_vlm,
    qwen_model_id,
    qwen_quantization,
    qwen_attention_mode,
    qwen_device,
    qwen_max_new_tokens,
    qwen_max_image_edge,
    qwen_keep_model_loaded,
    qwen_cpu_offload,
):
    from hyworld2.worldgen.src.vlm_utils import get_qwen_caption_format
    from hyworld2.worldgen.src.navi_utils import get_detail_navigation_instruction, get_navigation_instruction

    qwen_device = "auto"
    qwen_cpu_offload = False
    scene = Path(workspace["scene_dir"])
    panorama = _load_workspace_panorama(scene)
    pano_tensor = _pil_list_to_image_tensor([panorama])
    qwen = HYWorld2QwenVL()
    written = {}
    requested_scene_type = str(scene_type or "auto").lower()
    if requested_scene_type not in ("auto", "indoor", "outdoor"):
        requested_scene_type = "auto"
    print(f"[HYWorld2 Trajectories] Planner context: scene={scene}")
    print(f"[HYWorld2 Trajectories] Planner context: QwenVL model={qwen_model_id}, quantization={qwen_quantization}, device={qwen_device}, cpu_offload={bool(qwen_cpu_offload)}")

    meta_path = scene / "meta_info.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            meta.update(loaded)
    if requested_scene_type in ("indoor", "outdoor"):
        if str(meta.get("scene_type", "")).lower() != requested_scene_type:
            meta["scene_type"] = requested_scene_type
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
            written["meta_info"] = str(meta_path)
        print(f"[HYWorld2 Trajectories] Planner context: using manual scene_type={requested_scene_type}")
    elif str(meta.get("scene_type", "unknown")).lower() not in ("indoor", "outdoor"):
        print(f"[HYWorld2 Trajectories] Planner context: classifying scene_type from 480px preview -> {meta_path}")
        scene_type_tensor = _pil_list_to_image_tensor([_qwenvl_preview_image(panorama, max_image_edge=480)])
        text = qwen._generate(
            qwen_model_id,
            get_qwen_caption_format("env_cls"),
            images=scene_type_tensor,
            device=qwen_device,
            max_new_tokens=min(int(qwen_max_new_tokens), 64),
            max_image_edge=480,
            quantization=qwen_quantization,
            attention_mode=qwen_attention_mode,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.0,
            keep_model_loaded=qwen_keep_model_loaded,
            cpu_offload=qwen_cpu_offload,
            seed=1024,
        )
        meta["scene_type"] = _parse_scene_type(text)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        written["meta_info"] = str(meta_path)
        print(f"[HYWorld2 Trajectories] Planner context: scene_type={meta['scene_type']}")
    else:
        print(f"[HYWorld2 Trajectories] Planner context: reusing scene_type={meta.get('scene_type')} from {meta_path}")
    workspace["scene_type"] = str(meta.get("scene_type", workspace.get("scene_type", "unknown"))).lower()

    objects_path = scene / "objects.json"
    if (apply_nav_traj or apply_detail_traj) and not objects_path.exists():
        print(f"[HYWorld2 Trajectories] Planner context: extracting navigation objects -> {objects_path}")
        text = qwen._generate(
            qwen_model_id,
            get_navigation_instruction(bool(force_vlm)),
            images=pano_tensor,
            device=qwen_device,
            max_new_tokens=int(qwen_max_new_tokens),
            max_image_edge=int(qwen_max_image_edge),
            quantization=qwen_quantization,
            attention_mode=qwen_attention_mode,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.1,
            keep_model_loaded=qwen_keep_model_loaded,
            cpu_offload=qwen_cpu_offload,
            seed=1024,
        )
        objects = _parse_qwenvl_objects(text)
        with open(objects_path, "w", encoding="utf-8") as handle:
            json.dump(objects, handle, indent=2)
        written["objects"] = str(objects_path)
        print(f"[HYWorld2 Trajectories] Planner context: wrote {len(objects)} navigation object(s)")
    elif apply_nav_traj or apply_detail_traj:
        print(f"[HYWorld2 Trajectories] Planner context: reusing navigation objects from {objects_path}")
    detail_objects_path = scene / "detail_objects.json"
    if apply_detail_traj and not detail_objects_path.exists():
        existing_objects = []
        if objects_path.exists():
            try:
                loaded_objects = _hyworld2_read_json_file(objects_path, default=[]) or []
                if isinstance(loaded_objects, list):
                    existing_objects = [str(item) for item in loaded_objects]
            except Exception:
                existing_objects = []
        print(f"[HYWorld2 Trajectories] Planner context: extracting extreme detail objects -> {detail_objects_path}")
        text = qwen._generate(
            qwen_model_id,
            get_detail_navigation_instruction(existing_objects, max_items=int(detail_object_limit), force_vlm=bool(force_vlm)),
            images=pano_tensor,
            device=qwen_device,
            max_new_tokens=int(qwen_max_new_tokens),
            max_image_edge=int(qwen_max_image_edge),
            quantization=qwen_quantization,
            attention_mode=qwen_attention_mode,
            temperature=0.2,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.1,
            keep_model_loaded=qwen_keep_model_loaded,
            cpu_offload=qwen_cpu_offload,
            seed=2048,
        )
        excluded = {_qwenvl_object_key(item) for item in existing_objects}
        detail_candidates = _parse_qwenvl_objects(text)
        detail_objects = [
            item for item in detail_candidates
            if _qwenvl_object_key(item) and _qwenvl_object_key(item) not in excluded
        ][: max(1, int(detail_object_limit))]
        with open(detail_objects_path, "w", encoding="utf-8") as handle:
            json.dump(detail_objects, handle, indent=2)
        written["detail_objects"] = str(detail_objects_path)
        print(f"[HYWorld2 Trajectories] Planner context: wrote {len(detail_objects)} extreme detail object(s)")
    elif apply_detail_traj:
        print(f"[HYWorld2 Trajectories] Planner context: reusing extreme detail objects from {detail_objects_path}")
    if not qwen_keep_model_loaded:
        HYWorld2QwenVL._clear_cache()
    return written


def _trajectory_scene_median_depth(scene):
    path = Path(scene) / "render_results" / "full_depth_prediction.pt"
    if not path.exists():
        return 1.0
    try:
        full_depth = torch.load(path, weights_only=False, map_location="cpu")
        distance = full_depth.get("distance") if isinstance(full_depth, dict) else None
        if distance is None:
            return 1.0
        values = distance.detach().float()
        values = values[torch.isfinite(values) & (values > 0)]
        if values.numel() == 0:
            return 1.0
        return float(torch.median(values).item())
    except Exception as exc:
        print(f"[HYWorld2 Trajectories] Could not read median depth: {exc}")
        return 1.0


def _anchor_camera_candidates(scene):
    render_root = Path(scene) / "render_results"
    paths = []
    for pattern in ("view*/traj*/camera.json", "target*/traj*/camera.json",
                    "wonder*/traj*/camera.json", "reconstruct*/traj*/camera.json"):
        paths.extend(render_root.glob(pattern))
    result = []
    for path in sorted(paths):
        if path.parts[-3].startswith("wonder_scan_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            w2cs = np.asarray(data.get("extrinsic", []), dtype=np.float64)
            if w2cs.ndim != 3 or w2cs.shape[1:] != (4, 4) or len(w2cs) == 0:
                continue
            c2ws = np.linalg.inv(w2cs)
            position = c2ws[-1, :3, 3].astype(np.float64)
            result.append({"path": path, "data": data, "c2w": c2ws[-1], "position": position})
        except Exception as exc:
            print(f"[HYWorld2 Trajectories] Skipping anchor candidate {path}: {exc}")
    return result


def _make_anchor_scan_c2ws(anchor_c2w, nframe, yaw_degrees):
    position = anchor_c2w[:3, 3].astype(np.float64)
    base_forward = anchor_c2w[:3, 2].astype(np.float64)
    base_forward[2] = 0.0
    if np.linalg.norm(base_forward) < 1e-6:
        base_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    base_forward = base_forward / np.linalg.norm(base_forward)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    frames = max(2, int(nframe))
    # A full scan includes the closing pose. With the native 21-frame format this
    # gives an 18-degree interval and exact cardinal poses at frames 0/5/10/15.
    full_circle = np.isclose(abs(float(yaw_degrees)), 360.0, atol=1e-6)
    angles = np.linspace(0.0, np.deg2rad(float(yaw_degrees)), frames, endpoint=full_circle)
    c2ws = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        forward = np.array([
            base_forward[0] * c - base_forward[1] * s,
            base_forward[0] * s + base_forward[1] * c,
            0.0,
        ], dtype=np.float64)
        forward = forward / max(np.linalg.norm(forward), 1e-8)
        cam_up = -up
        right = np.cross(cam_up, forward)
        right = right / max(np.linalg.norm(right), 1e-8)
        cam_up = np.cross(forward, right)
        cam_up = cam_up / max(np.linalg.norm(cam_up), 1e-8)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = cam_up
        c2w[:3, 2] = forward
        c2w[:3, 3] = position
        c2ws.append(c2w)
    return np.asarray(c2ws, dtype=np.float64)


def _safe_anchor_positions(scene, topk, min_distance, min_separation):
    """Choose scan anchors inside robust scene bounds and away from surfaces."""
    from scipy.spatial import cKDTree

    pcd_path = Path(scene) / "render_results" / "global_pcd.ply"
    if not pcd_path.exists():
        return []
    import trimesh

    cloud = trimesh.load(pcd_path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 32:
        return []
    # Ignore isolated depth outliers; they otherwise make the nominal scene box enormous.
    low, high = np.quantile(points, [0.01, 0.99], axis=0)
    extent = high - low
    if np.any(extent[:2] <= 1e-6):
        return []
    center = (low + high) * 0.5
    scale = max(float(np.median(extent[:2])), 1e-6)
    margin_xy = np.maximum(extent[:2] * 0.08, scale * 0.03)
    inner_low, inner_high = low[:2] + margin_xy, high[:2] - margin_xy
    if np.any(inner_high <= inner_low):
        return []

    candidates = _anchor_camera_candidates(scene)
    existing_z = [item["position"][2] for item in candidates if np.isfinite(item["position"][2])]
    camera_z = float(np.median(existing_z)) if existing_z else float(center[2])
    camera_z = float(np.clip(camera_z, low[2] + extent[2] * 0.05, high[2] - extent[2] * 0.05))

    # Existing valid poses are useful priors; a regular grid makes the method independent
    # of bad wonder/reconstruct endpoints and lets it discover large free regions.
    xy = [item["position"][:2] for item in candidates]
    grid_side = int(np.clip(np.ceil(np.sqrt(max(25, int(topk) * 16))), 5, 15))
    xs = np.linspace(inner_low[0], inner_high[0], grid_side)
    ys = np.linspace(inner_low[1], inner_high[1], grid_side)
    xy.extend(np.array(np.meshgrid(xs, ys)).reshape(2, -1).T)
    xy.append(center[:2])
    positions = np.asarray([[p[0], p[1], camera_z] for p in xy], dtype=np.float64)
    inside = np.all((positions[:, :2] >= inner_low) & (positions[:, :2] <= inner_high), axis=1)
    positions = positions[inside]

    # Subsample only for the distance index; quantile bounds above still use all points.
    index_points = points[::max(1, len(points) // 500000)]
    tree = cKDTree(index_points)
    clearance = tree.query(positions, k=1, workers=-1)[0]
    min_clearance = max(float(min_distance) * _trajectory_scene_median_depth(scene), scale * 0.015)
    valid = clearance >= min_clearance
    positions, clearance = positions[valid], clearance[valid]
    if not len(positions):
        return []

    # Prefer open, central locations. The centrality term avoids boundary-hugging anchors
    # even when a side of an outdoor point cloud is sparse.
    centrality = np.linalg.norm((positions[:, :2] - center[:2]) / np.maximum(extent[:2], 1e-6), axis=1)
    score = clearance - scale * 0.2 * centrality
    order = np.argsort(score)[::-1]
    separation = max(float(min_separation) * _trajectory_scene_median_depth(scene), scale * 0.08)
    selected = []
    for idx in order:
        position = positions[idx]
        if any(np.linalg.norm(position[:2] - other[:2]) < separation for other in selected):
            continue
        selected.append(position)
        if len(selected) >= int(topk):
            break
    return selected


def _write_anchor_scans(scene, topk, min_distance, min_separation, yaw_degrees, nframe):
    import cv2
    from hyworld2.worldgen.src.panorama_utils import split_panorama_image

    scene = Path(scene)
    positions = _safe_anchor_positions(scene, topk, min_distance, min_separation)
    if not positions:
        print("[HYWorld2 Trajectories] Anchor scan: no geometrically safe in-bounds positions found")
        return []
    median_depth = max(_trajectory_scene_median_depth(scene), 1e-6)
    print(
        "[HYWorld2 Trajectories] Anchor scan: "
        f"safe_positions={len(positions)}, topk={int(topk)}, median_depth={median_depth:.4f}, "
        f"min_distance={float(min_distance)}x, min_separation={float(min_separation)}x"
    )
    full_img = _load_workspace_panorama(scene)
    source_candidates = _anchor_camera_candidates(scene)
    template = source_candidates[0] if source_candidates else None
    if template is None:
        print("[HYWorld2 Trajectories] Anchor scan: camera intrinsics unavailable")
        return []
    written = []
    for index, position in enumerate(positions):
        data = template["data"]
        image_w = int(data["width"])
        image_h = int(data["height"])
        K = np.asarray(data["intrinsic"][0], dtype=np.float64)
        anchor_c2w = template["c2w"].copy()
        anchor_c2w[:3, 3] = position
        c2ws = _make_anchor_scan_c2ws(anchor_c2w, nframe, yaw_degrees)
        w2cs = np.linalg.inv(c2ws)
        dets = np.linalg.det(w2cs[:, :3, :3])
        up_z = c2ws[:, 2, 1]
        if np.any(dets < 0.9) or np.any(dets > 1.1) or np.any(up_z > -0.5):
            raise RuntimeError(
                "HYWorld2 anchor scan generated invalid camera orientation "
                f"for position {position.tolist()}: det_range=({float(dets.min()):.4f}, {float(dets.max()):.4f}), "
                f"up_z_range=({float(up_z.min()):.4f}, {float(up_z.max()):.4f})."
            )
        K_pano = K.copy()
        K_pano[0, :] /= image_w
        K_pano[1, :] /= image_h
        start = split_panorama_image(np.array(full_img), w2cs[0:1], np.array([K_pano]), h=image_h, w=image_w, interp=cv2.INTER_AREA)[0]
        view_dir = scene / "render_results" / f"wonder_scan_{index}"
        traj_dir = view_dir / "traj0"
        _ensure_dir(traj_dir)
        Image.fromarray(start).save(view_dir / "start_frame.png")
        camera_info = {
            "id": index,
            "type": "anchor_scan",
            "source_camera": str(template["path"]),
            "width": image_w,
            "height": image_h,
            "fov_x": float(data.get("fov_x", np.rad2deg(2.0 * np.arctan(image_w / (2.0 * K[0, 0]))))),
            "fov_y": float(data.get("fov_y", np.rad2deg(2.0 * np.arctan(image_h / (2.0 * K[1, 1]))))),
            "intrinsic": [K.tolist()] * len(w2cs),
            "extrinsic": w2cs.tolist(),
            "anchor_position": position.tolist(),
            "anchor_validation": "robust_pcd_bounds_and_surface_clearance",
            "yaw_degrees": float(yaw_degrees),
        }
        with open(traj_dir / "camera.json", "w", encoding="utf-8") as handle:
            json.dump(camera_info, handle, indent=2)
        written.append(str(traj_dir / "camera.json"))
        print(f"[HYWorld2 Trajectories] Anchor scan: wrote {traj_dir / 'camera.json'}")
    return written


class HYWorld2Workspace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace_name": ("STRING", {"default": "comfy_worldgen"}),
            },
            "optional": {
                "root_dir": ("STRING", {"default": ""}),
                "scene_dir": ("STRING", {"default": ""}),
                "panorama": ("IMAGE",),
                "scene_type": (["unknown", "indoor", "outdoor"], {"default": "unknown"}),
                "result_name": ("STRING", {"default": "worldstereo-memory-dmd"}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_WORKSPACE", "STRING")
    RETURN_NAMES = ("workspace", "info")
    FUNCTION = "build"
    CATEGORY = "VNCCS/HYWorld2"

    @classmethod
    def IS_CHANGED(cls, workspace_name, **kwargs):
        scene_dir = str(kwargs.get("scene_dir", "") or "").strip()
        if scene_dir:
            scene = Path(scene_dir)
        else:
            root_dir = str(kwargs.get("root_dir", "") or "").strip()
            root = Path(root_dir) if root_dir else _output_root() / "hyworld2_worldgen"
            scene = root / _sanitize_name(workspace_name, "comfy_worldgen")
        # Keep Comfy's native cache tied to node inputs, not to files this node
        # rewrites while running. Watching mtime of workspace_state/meta files
        # invalidates upstream cache after downstream failures.
        debug_payload = {
            "scene": str(scene),
            "kwargs": {key: kwargs[key] for key in sorted(kwargs) if key != "panorama"},
            "panorama": _hyworld2_image_tensor_fingerprint(kwargs.get("panorama")) if kwargs.get("panorama") is not None else None,
        }
        _hy_cache_debug("Workspace", "IS_CHANGED", debug_payload)
        state = [str(scene), _safe_json_dumps({key: kwargs[key] for key in sorted(kwargs) if key != "panorama"})]
        if kwargs.get("panorama") is not None:
            state.append(_safe_json_dumps({"input_panorama": _hyworld2_image_tensor_fingerprint(kwargs.get("panorama"))}))
        return "|".join(state)

    def build(self, workspace_name, root_dir="", scene_dir="", panorama=None, scene_type="unknown", result_name="worldstereo-memory-dmd"):
        _hy_log("Workspace", f"Stage 1/3: resolving workspace (name={workspace_name})")
        if str(scene_dir).strip():
            scene = Path(scene_dir)
        else:
            root = Path(root_dir) if str(root_dir).strip() else _output_root() / "hyworld2_worldgen"
            scene = root / _sanitize_name(workspace_name, "comfy_worldgen")
        _hy_log("Workspace", f"Workspace directory: {scene}")
        _ensure_dir(scene)
        workspace_state_path = _hyworld2_workspace_state_path(scene)
        workspace_state = _hyworld2_read_json_file(workspace_state_path, default={}) or {}
        cache_action = "reuse_without_panorama_input"
        if panorama is not None:
            _hy_log("Workspace", "Stage 2/3: checking input panorama cache")
            frames = _image_tensor_to_pil_list(panorama)
            if frames:
                incoming = frames[0].convert("RGB")
                incoming_fp = _hyworld2_pil_pixel_fingerprint(incoming)
                existing_fp = _hyworld2_image_file_pixel_fingerprint(scene / "panorama.png")
                cached_fp = workspace_state.get("panorama")
                if existing_fp == incoming_fp:
                    cache_action = "panorama_unchanged"
                    _hy_log("Workspace", "Input panorama matches workspace panorama; keeping existing derived files")
                else:
                    cache_action = "panorama_changed" if existing_fp else "panorama_initialized"
                    _hy_log("Workspace", f"Input panorama cache miss ({cache_action}); clearing derived workspace files")
                    _hyworld2_clear_workspace_derivatives(scene)
                    _ensure_dir(scene)
                    incoming.save(scene / "panorama.png")
                    _hy_log("Workspace", f"Saved panorama: {scene / 'panorama.png'}")
                if cached_fp != incoming_fp:
                    workspace_state["panorama"] = incoming_fp
            else:
                _hy_log("Workspace", "Stage 2/3: panorama input is empty; reusing workspace files")
        else:
            _hy_log("Workspace", "Stage 2/3: no panorama input connected; reusing workspace files")
        _ensure_dir(scene / "render_results")
        meta_path = scene / "meta_info.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                meta.update(loaded)
        if scene_type != "unknown" or "scene_type" not in meta:
            meta["scene_type"] = scene_type
        workspace_state["workspace_name"] = workspace_name
        workspace_state["result_name"] = result_name
        workspace_state["scene_type"] = meta.get("scene_type", "unknown")
        workspace_state["cache_action"] = cache_action
        _hy_log("Workspace", f"Stage 3/3: writing metadata scene_type={meta.get('scene_type', 'unknown')}, cache_action={cache_action}")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        _hyworld2_write_json_file(workspace_state_path, workspace_state)
        workspace = {
            "scene_dir": str(scene),
            "render_results_dir": str(scene / "render_results"),
            "workspace_name": workspace_name,
            "result_name": result_name,
            "scene_type": meta.get("scene_type", "unknown"),
            "cache_action": cache_action,
        }
        _hy_log("Workspace", "Workspace ready")
        return (workspace, _safe_json_dumps(workspace))


class HYWorld2QwenVL:
    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["scene_objects", "trajectory_caption", "prompt_refine"], {"default": "trajectory_caption"}),
                "model_id": (_qwenvl_model_names(), {"default": HYWORLD2_QWENVL_DEFAULT}),
                "quantization": (HYWORLD2_QWENVL_QUANTIZATION, {"default": HYWORLD2_QWENVL_DEFAULT_QUANTIZATION}),
                "attention_mode": (HYWORLD2_QWENVL_ATTENTION, {"default": "auto"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "images": ("IMAGE",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16}),
                "max_image_edge": ("INT", {"default": HYWORLD2_QWENVL_MAX_IMAGE_EDGE, "min": 128, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.05}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "write_results": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_LLM_CONTEXT", "STRING")
    RETURN_NAMES = ("llm_context", "text")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    @classmethod
    def _clear_cache(cls, keep_signature=None):
        for signature, bundle in list(cls._model_cache.items()):
            if keep_signature is not None and signature == keep_signature:
                continue
            try:
                model = bundle.get("model")
                if model is not None:
                    model.cpu()
            except Exception:
                pass
            cls._model_cache.pop(signature, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_bundle(self, model_id, quantization=HYWORLD2_QWENVL_DEFAULT_QUANTIZATION, attention_mode="auto", device="auto", keep_model_loaded=True, cpu_offload=False):
        try:
            from transformers import AutoProcessor, AutoTokenizer
            try:
                from transformers import AutoModelForImageTextToText as AutoModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModel
        except Exception as exc:
            raise ImportError("QwenVL requires transformers with vision-language model support. Install project requirements.") from exc

        device = "auto"
        cpu_offload = False
        requested_device = str(device or "auto").strip()
        selected_device = _qwenvl_normalize_device(device)
        allow_cpu_offload = bool(cpu_offload) and requested_device == "auto" and torch.cuda.is_available()
        quant_cfg, dtype, is_fp8 = _qwenvl_quantization_config(model_id, quantization, cpu_offload=allow_cpu_offload)
        force_sdpa = is_fp8 or quant_cfg is not None
        attn_impl = _qwenvl_resolve_attention(attention_mode, force_sdpa=force_sdpa)
        signature = (str(model_id), str(quantization), attn_impl, selected_device, allow_cpu_offload)
        if keep_model_loaded and signature in self._model_cache:
            return self._model_cache[signature], signature

        self._clear_cache()
        model_path = _qwenvl_ensure_model(model_id)
        load_kwargs = {
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "trust_remote_code": True,
        }
        if is_fp8:
            load_kwargs["device_map"] = None
            load_kwargs["torch_dtype"] = "auto"
        else:
            if allow_cpu_offload:
                load_kwargs["device_map"] = "auto"
                max_memory = _qwenvl_auto_max_memory()
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
            else:
                load_kwargs["device_map"] = selected_device if selected_device not in ("cpu", "mps") else None
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if quant_cfg is not None:
                load_kwargs["quantization_config"] = quant_cfg
        print(f"[HYWorld2 QwenVL] Loading {model_id} ({quantization}, attn={attn_impl}, device={selected_device}, cpu_offload={allow_cpu_offload})")
        print(f"[HYWorld2 QwenVL] Local model path: {model_path}")
        if "max_memory" in load_kwargs:
            print(f"[HYWorld2 QwenVL] Accelerate max_memory: {load_kwargs['max_memory']}")
        model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
        if selected_device in ("cpu", "mps") or is_fp8:
            model = model.to(selected_device)
        if hasattr(model, "hf_device_map"):
            print(f"[HYWorld2 QwenVL] Device map: {model.hf_device_map}")
        print("[HYWorld2 QwenVL] Loading processor/tokenizer")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("[HYWorld2 QwenVL] Model ready")
        bundle = {"model": model, "processor": processor, "tokenizer": tokenizer}
        if keep_model_loaded:
            self._model_cache[signature] = bundle
        return bundle, signature

    def _generate(
        self,
        model_id,
        prompt,
        images=None,
        device="auto",
        max_new_tokens=256,
        max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        quantization=HYWORLD2_QWENVL_DEFAULT_QUANTIZATION,
        attention_mode="auto",
        temperature=0.6,
        top_p=0.9,
        num_beams=1,
        repetition_penalty=1.2,
        keep_model_loaded=True,
        cpu_offload=False,
        seed=1,
    ):
        torch.manual_seed(int(seed))
        bundle, signature = self._load_bundle(
            model_id,
            quantization=quantization,
            attention_mode=attention_mode,
            device=device,
            keep_model_loaded=keep_model_loaded,
            cpu_offload=cpu_offload,
        )
        model = bundle["model"]
        processor = bundle["processor"]
        tokenizer = bundle["tokenizer"]
        pil_images = [_qwenvl_preview_image(image, max_image_edge=max_image_edge) for image in (_image_tensor_to_pil_list(images)[:8] if images is not None else [])]
        content = []
        for image in pil_images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=pil_images or None, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        stop_tokens = [tokenizer.eos_token_id]
        if getattr(tokenizer, "eot_id", None) is not None:
            stop_tokens.append(tokenizer.eot_id)
        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "num_beams": int(num_beams),
            "eos_token_id": stop_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if int(num_beams) == 1:
            generate_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
        else:
            generate_kwargs["do_sample"] = False
        with torch.no_grad():
            generated = model.generate(**inputs, **generate_kwargs)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if not keep_model_loaded:
            # Non-cached bundles are not visible to _clear_cache(). Explicitly
            # break their references here; otherwise Transformers/Accelerate/
            # quantization cycles can keep the final caption model resident until
            # a much later GC, overlapping WorldStereo despite keep=False.
            with contextlib.suppress(Exception):
                model.cpu()
            bundle.clear()
            del generated, inputs, model, processor, tokenizer, bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._clear_cache(keep_signature=None)
        else:
            self._clear_cache(keep_signature=signature)
        return result

    def run(
        self,
        workspace,
        mode,
        model_id,
        quantization,
        attention_mode,
        prompt,
        images=None,
        trajectory_set=None,
        device="auto",
        max_new_tokens=256,
        max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        temperature=0.6,
        top_p=0.9,
        num_beams=1,
        repetition_penalty=1.2,
        cpu_offload=False,
        keep_model_loaded=True,
        seed=1,
        write_results=True,
    ):
        device = "auto"
        cpu_offload = False
        scene = Path(workspace["scene_dir"])
        _hy_log("QwenVL", f"Stage 1/3: preparing prompt (mode={mode})")
        if not prompt.strip():
            if mode == "scene_objects":
                prompt = "Analyze this panoramic scene. Return concise JSON with scene_type, objects, navigable_areas, and visual_style."
            elif mode == "trajectory_caption":
                prompt = "Describe the visible trajectory render as a concise image generation prompt. Return only the prompt text."
            else:
                raise ValueError("prompt_refine requires a non-empty prompt; fallback prompts are disabled.")
        image_count = len(_image_tensor_to_pil_list(images)) if images is not None else 0
        traj_count = len((trajectory_set or {}).get("render_list", [])) if trajectory_set else 0
        _hy_log("QwenVL", f"Stage 2/3: generating text with model={model_id}, quantization={quantization}, device={device}, cpu_offload={bool(cpu_offload)}, images={image_count}, trajectories={traj_count}")
        text = self._generate(
            model_id,
            prompt,
            images=images,
            device=device,
            max_new_tokens=max_new_tokens,
            max_image_edge=max_image_edge,
            quantization=quantization,
            attention_mode=attention_mode,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            cpu_offload=bool(cpu_offload),
            keep_model_loaded=keep_model_loaded,
            seed=seed,
        )
        context = {"mode": mode, "text": text, "model_id": model_id, "quantization": quantization, "attention_mode": attention_mode}
        if write_results:
            _hy_log("QwenVL", "Stage 3/3: writing QwenVL outputs")
            if mode == "scene_objects":
                out_path = scene / "hyworld2_qwenvl_scene.json"
                try:
                    from hyworld2.worldgen.src.json_utils import loads_repaired

                    parsed = loads_repaired(text)
                except Exception:
                    parsed = {"raw": text}
                with open(out_path, "w", encoding="utf-8") as handle:
                    json.dump(parsed, handle, indent=2)
                context["scene_objects_path"] = str(out_path)
                _hy_log("QwenVL", f"Wrote scene context: {out_path}")
            elif mode == "trajectory_caption" and trajectory_set:
                render_list = trajectory_set.get("render_list", [])
                for render_path in render_list:
                    path = Path(render_path)
                    caption_path = path.parent / "traj_caption.json"
                    with open(caption_path, "w", encoding="utf-8") as handle:
                        json.dump({"prompt": text, "source": "HYWorld2 QwenVL"}, handle, indent=2)
                context["captions_written"] = len(render_list)
                _hy_log("QwenVL", f"Wrote {len(render_list)} trajectory caption file(s)")
        else:
            _hy_log("QwenVL", "Stage 3/3: write_results disabled")
        _hy_log("QwenVL", "QwenVL node complete")
        return (context, text)


class HYWorld2Trajectories:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
            },
            "optional": {
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**31 - 1, "control_after_generate": "fixed"}),
                "scene_type": (["auto", "indoor", "outdoor"], {"default": "auto"}),
                "additional_nav_traj": ("BOOLEAN", {"default": False}),
                "extreme_detail_traj": ("BOOLEAN", {"default": False}),
                "detail_object_limit": ("INT", {"default": 6, "min": 1, "max": 16}),
                "qwen_model_id": (_qwenvl_model_names(), {"default": HYWORLD2_QWENVL_DEFAULT}),
                "qwen_quantization": (HYWORLD2_QWENVL_QUANTIZATION, {"default": HYWORLD2_QWENVL_DEFAULT_QUANTIZATION}),
                "qwen_max_image_edge": ("INT", {"default": HYWORLD2_QWENVL_MAX_IMAGE_EDGE, "min": 128, "max": 4096, "step": 64}),
                "apply_anchor_scan": ("BOOLEAN", {"default": False}),
                "anchor_scan_topk": ("INT", {"default": 2, "min": 0, "max": 32}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_TRAJECTORY_SET", "STRING")
    RETURN_NAMES = ("trajectory_set", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    @classmethod
    def IS_CHANGED(cls, workspace, **kwargs):
        if not isinstance(workspace, dict) or not workspace.get("scene_dir"):
            debug_payload = {
                "workspace": "missing_or_unresolved",
                "workspace_type": type(workspace).__name__,
                "kwargs": {key: kwargs[key] for key in sorted(kwargs)},
            }
            _hy_cache_debug("Trajectories", "IS_CHANGED", debug_payload)
            return _safe_json_dumps(debug_payload)
        scene = Path(workspace["scene_dir"])
        # This node produces render_results and trajectory state files. They must
        # not participate in Comfy's native cache key, otherwise a downstream
        # error makes the next queue start from trajectory generation again.
        debug_payload = {
            "scene": str(scene),
            "result_name": str(workspace.get("result_name", "")),
            "workspace_scene_type": str(workspace.get("scene_type", "")),
            "workspace_panorama": workspace.get("panorama") or {},
            "kwargs": {key: kwargs[key] for key in sorted(kwargs)},
        }
        _hy_cache_debug("Trajectories", "IS_CHANGED", debug_payload)
        state = [
            str(scene),
            str(workspace.get("result_name", "")),
            str(workspace.get("scene_type", "")),
            _safe_json_dumps(workspace.get("panorama") or {}),
        ]
        if kwargs:
            state.append(_safe_json_dumps({key: kwargs[key] for key in sorted(kwargs)}))
        return "|".join(state)

    def _sort(self, workspace, generated=False, captions_written=None, anchor_scans_written=None, logs=None):
        render_root = Path(workspace["scene_dir"]) / "render_results"
        try:
            from hyworld2.worldgen.src.data_utils import sort_trajs
            render_list = sort_trajs(str(render_root))
        except Exception as exc:
            print(f"[HYWorld2 Trajectories] sort_trajs unavailable ({type(exc).__name__}: {exc}); using fallback.")
            render_list = []
        if not render_list:
            def fallback_key(path):
                path = Path(path)
                view_id = path.parts[-3] if len(path.parts) >= 3 else ""
                traj_id = path.parts[-2] if len(path.parts) >= 2 else ""
                if view_id.startswith("view"):
                    group = 0
                    traj_order = {"traj2": 0, "traj0": 1, "traj1": 2}.get(traj_id, 99)
                elif view_id.startswith("target"):
                    group = 1
                    traj_order = 0
                elif view_id.startswith("reconstruct"):
                    group = 4 if traj_id == "traj1" else 2
                    traj_order = 0
                elif view_id.startswith("wonder"):
                    group = 3
                    traj_order = 0
                else:
                    group = 9
                    traj_order = 0
                return group, view_id, traj_order, traj_id

            render_list = [str(path) for path in sorted(render_root.glob("**/render.mp4"), key=fallback_key)]
            if render_list:
                print(
                    "[HYWorld2 Trajectories] sort_trajs returned empty; "
                    f"using Windows-safe fallback with {len(render_list)} render(s)."
                )
        data = {
            "workspace": workspace,
            "render_list": render_list,
            "count": len(render_list),
            "generated": bool(generated),
            "captions_written": captions_written or [],
            "anchor_scans_written": anchor_scans_written or [],
            "logs": logs or [],
        }
        return (data, _safe_json_dumps(data))

    def _settings_signature(self, **settings):
        normalized = {}
        for key, value in settings.items():
            if isinstance(value, Path):
                normalized[key] = str(value)
            elif isinstance(value, (bool, int, float, str)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return hashlib.sha256(_safe_json_dumps(normalized).encode("utf-8")).hexdigest(), normalized

    def run(
        self,
        workspace,
        seed=1,
        scene_type="auto",
        additional_nav_traj=False,
        extreme_detail_traj=False,
        detail_object_limit=6,
        fov_x=120.0,
        fov_y=90.0,
        split_view_num=3,
        splitted_resolution=480,
        nframe=21,
        distance_threshold=0.1,
        obs_iteration_limit=3,
        rotation_deg=120.0,
        rotation_up=45.0,
        up_right=60.0,
        obs_decay=2 / 3,
        contract=8.0,
        apply_nav_traj=False,
        wonder_topk=3,
        recon_topk=5,
        move_dist=8.0,
        radius_threshold=4.0,
        min_angle_threshold=40.0,
        traj_sim_threshold=0.7,
        traj_sim_threshold_recon=0.7,
        apply_up_route=False,
        apply_recon_iteration=False,
        eloop_dist=0.25,
        force_vlm=False,
        cellSize=0.1,
        cellHeight=0.1,
        agentHeight=0.2,
        agentRadius=0.1,
        agentMaxClimb=0.1,
        maxSlope=30.0,
        roof_height_threshold=0.1,
        sam3_path=HYWORLD2_SAM3_REPO_ID,
        local_files_only=False,
        render_processes=0,
        caption_mode="qwenvl_missing",
        qwen_model_id=HYWORLD2_QWENVL_DEFAULT,
        qwen_quantization=HYWORLD2_QWENVL_DEFAULT_QUANTIZATION,
        qwen_attention_mode="auto",
        qwen_device="auto",
        qwen_cpu_offload=False,
        qwen_max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        qwen_max_new_tokens=256,
        qwen_keep_model_loaded=True,
        qwen_frame_count=4,
        apply_anchor_scan=False,
        anchor_scan_topk=2,
        anchor_scan_min_distance=1.0,
        anchor_scan_min_separation=0.75,
        anchor_scan_yaw_degrees=360.0,
        points_per_pixel=20,
        global_pcd_voxel_size=0.0,
        no_llm_mode=False,
        image_width=0,
        image_height=0,
    ):
        qwen_device = "auto"
        qwen_cpu_offload = False
        apply_detail_traj = bool(extreme_detail_traj)
        detail_object_limit = max(1, min(16, int(detail_object_limit)))
        apply_object_nav_traj = bool(additional_nav_traj or apply_nav_traj)
        apply_nav_traj = bool(apply_object_nav_traj or apply_detail_traj)
        scene = Path(workspace["scene_dir"])
        render_root = scene / "render_results"
        logs = []
        settings_signature, settings_state = self._settings_signature(
            seed=int(seed),
            scene_type=str(scene_type),
            additional_nav_traj=bool(apply_object_nav_traj),
            extreme_detail_traj=bool(apply_detail_traj),
            detail_object_limit=int(detail_object_limit),
            fov_x=float(fov_x),
            fov_y=float(fov_y),
            split_view_num=int(split_view_num),
            splitted_resolution=int(splitted_resolution),
            nframe=int(nframe),
            distance_threshold=float(distance_threshold),
            obs_iteration_limit=int(obs_iteration_limit),
            rotation_deg=float(rotation_deg),
            rotation_up=float(rotation_up),
            up_right=float(up_right),
            obs_decay=float(obs_decay),
            contract=float(contract),
            wonder_topk=int(wonder_topk),
            recon_topk=int(recon_topk),
            move_dist=float(move_dist),
            radius_threshold=float(radius_threshold),
            min_angle_threshold=float(min_angle_threshold),
            traj_sim_threshold=float(traj_sim_threshold),
            traj_sim_threshold_recon=float(traj_sim_threshold_recon),
            apply_up_route=bool(apply_up_route),
            apply_recon_iteration=bool(apply_recon_iteration),
            eloop_dist=float(eloop_dist),
            force_vlm=bool(force_vlm),
            cellSize=float(cellSize),
            cellHeight=float(cellHeight),
            agentHeight=float(agentHeight),
            agentRadius=float(agentRadius),
            agentMaxClimb=float(agentMaxClimb),
            maxSlope=float(maxSlope),
            roof_height_threshold=float(roof_height_threshold),
            sam3_path=str(sam3_path or HYWORLD2_SAM3_REPO_ID),
            local_files_only=bool(local_files_only),
            qwen_model_id=str(qwen_model_id),
            qwen_quantization=str(qwen_quantization),
            qwen_attention_mode=str(qwen_attention_mode),
            qwen_max_image_edge=int(qwen_max_image_edge),
            qwen_max_new_tokens=int(qwen_max_new_tokens),
            apply_anchor_scan=bool(apply_anchor_scan),
            anchor_scan_topk=int(anchor_scan_topk),
            anchor_scan_min_distance=float(anchor_scan_min_distance),
            anchor_scan_min_separation=float(anchor_scan_min_separation),
            anchor_scan_yaw_degrees=float(anchor_scan_yaw_degrees),
            points_per_pixel=int(points_per_pixel),
            global_pcd_voxel_size=float(global_pcd_voxel_size),
            no_llm_mode=bool(no_llm_mode),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        cache_ok, cache_reason, cache_state = _hyworld2_trajectory_cache_status(
            scene,
            settings_signature,
            require_nav=bool(apply_nav_traj),
            require_detail=bool(apply_detail_traj),
            require_anchor=bool(apply_anchor_scan),
            anchor_topk=int(anchor_scan_topk),
            workspace_cache_action=workspace.get("cache_action"),
        )
        if cache_ok:
            print(f"[HYWorld2 Trajectories] Auto cache hit: {cache_reason}. Reusing {render_root}")
            logs.append({"stage": "auto_cache", "action": "reuse_existing", "reason": cache_reason})
            if cache_state.get("_repair_state") or cache_state.get("panorama") != _hyworld2_image_file_pixel_fingerprint(scene / "panorama.png"):
                render_list = self._sort(workspace, generated=False, logs=logs)[0].get("render_list", [])
                _hyworld2_write_json_file(
                    _hyworld2_trajectory_state_path(scene),
                    {
                        "panorama": _hyworld2_image_file_pixel_fingerprint(scene / "panorama.png"),
                        "settings_signature": settings_signature,
                        "settings": settings_state,
                        "render_count": len(render_list),
                        "captions_written": 0,
                        "anchor_scans_written": 0,
                        "repaired_from_existing_artifacts": True,
                    },
                )
            return self._sort(workspace, generated=False, logs=logs)

        skip_exist = True
        render_root_existed = render_root.exists()
        print(f"[HYWorld2 Trajectories] Auto cache miss: {cache_reason}")
        if cache_reason == "trajectory settings changed":
            # Camera geometry settings (resolution/FOV included) must rebuild every
            # camera.json and start frame; reusing them would mismatch K and pixels.
            skip_exist = False
            logs.append({"stage": "auto_cache", "action": "camera_regeneration", "reason": cache_reason})
        if cache_reason == "panorama cache mismatch" and render_root_existed and workspace.get("cache_action") != "panorama_unchanged":
            print("[HYWorld2 Trajectories] Source panorama changed; clearing stale render_results for full regeneration.")
            shutil.rmtree(render_root)
            render_root_existed = False
            skip_exist = False
            logs.append({"stage": "auto_cache", "action": "full_regeneration", "reason": cache_reason})
        else:
            logs.append({"stage": "auto_cache", "action": "generate", "reason": cache_reason})

        _ensure_dir(render_root)
        missing_geometry = _hyworld2_missing_memory_prerequisites(scene)
        if skip_exist and missing_geometry and render_root_existed:
            print("[HYWorld2 Trajectories] Existing render_results is incomplete; forcing geometry/trajectory rebuild.")
            for path in missing_geometry:
                print(f"[HYWorld2 Trajectories] Missing prerequisite: {path}")
            skip_exist = False
        print("[HYWorld2 Trajectories] Stage 0/5: official trajectory pipeline")
        print(f"[HYWorld2 Trajectories] Workspace: {scene}")
        print(f"[HYWorld2 Trajectories] SAM3 repo/path: {sam3_path or HYWORLD2_SAM3_REPO_ID}")

        planner_written = {}
        if bool(no_llm_mode):
            resolved_scene_type = str(scene_type).lower()
            if resolved_scene_type not in ("indoor", "outdoor"):
                resolved_scene_type = str(workspace.get("scene_type", "indoor")).lower()
            if resolved_scene_type not in ("indoor", "outdoor"):
                resolved_scene_type = "indoor"
            meta_path = scene / "meta_info.json"
            meta = _hyworld2_read_json_file(meta_path, default={}) or {}
            meta["scene_type"] = resolved_scene_type
            _hyworld2_write_json_file(meta_path, meta)
            workspace["scene_type"] = resolved_scene_type
            print(f"[HYWorld2 Trajectories Test] LLM-free mode; scene_type={resolved_scene_type}")
        else:
            print("[HYWorld2 Trajectories] Releasing Comfy models before local QwenVL planner")
            _release_model_memory("HYWorld2 Trajectories")
            print("[HYWorld2 Trajectories] Stage 1/5: preparing local QwenVL planner context")
            planner_written = _ensure_trajectory_planner_context(
                workspace, scene_type=scene_type, apply_nav_traj=bool(apply_nav_traj),
                apply_detail_traj=bool(apply_detail_traj), detail_object_limit=int(detail_object_limit),
                force_vlm=bool(force_vlm), qwen_model_id=qwen_model_id,
                qwen_quantization=qwen_quantization, qwen_attention_mode=qwen_attention_mode,
                qwen_device=qwen_device, qwen_max_new_tokens=int(qwen_max_new_tokens),
                qwen_max_image_edge=int(qwen_max_image_edge),
                qwen_keep_model_loaded=bool(qwen_keep_model_loaded), qwen_cpu_offload=bool(qwen_cpu_offload),
            )
            HYWorld2QwenVL._clear_cache()
            print("[HYWorld2 Trajectories] Stage 1/5 complete: planner context ready")
            print("[HYWorld2 Trajectories] Releasing Comfy/Qwen models before geometry generation")
            _release_model_memory("HYWorld2 Trajectories")

        print("[HYWorld2 Trajectories] Stage 2/5: generating official camera trajectories")
        from hyworld2.worldgen import traj_generate, traj_render

        generate_config = Namespace(
            target_path=str(scene),
            fov_x=float(fov_x),
            fov_y=float(fov_y),
            seed=int(seed),
            split_view_num=int(split_view_num),
            splitted_resolution=int(splitted_resolution),
            image_width=max(0, int(image_width)),
            image_height=max(0, int(image_height)),
            nframe=int(nframe),
            distance_threshold=float(distance_threshold),
            obs_iteration_limit=int(obs_iteration_limit),
            rotation_deg=float(rotation_deg),
            rotation_up=float(rotation_up),
            up_right=float(up_right),
            obs_decay=float(obs_decay),
            contract=float(contract),
            skip_exist=bool(skip_exist),
            apply_nav_traj=bool(apply_nav_traj),
            apply_object_nav_traj=bool(apply_object_nav_traj),
            apply_detail_traj=bool(apply_detail_traj),
            detail_object_limit=int(detail_object_limit),
            wonder_topk=int(wonder_topk),
            recon_topk=int(recon_topk),
            move_dist=float(move_dist),
            radius_threshold=float(radius_threshold),
            min_angle_threshold=float(min_angle_threshold),
            traj_sim_threshold=float(traj_sim_threshold),
            traj_sim_threshold_recon=float(traj_sim_threshold_recon),
            apply_up_route=bool(apply_up_route),
            apply_recon_iteration=bool(apply_recon_iteration),
            eloop_dist=float(eloop_dist),
            force_vlm=bool(force_vlm),
            cellSize=float(cellSize),
            cellHeight=float(cellHeight),
            agentHeight=float(agentHeight),
            agentRadius=float(agentRadius),
            agentMaxClimb=float(agentMaxClimb),
            maxSlope=float(maxSlope),
            roof_height_threshold=float(roof_height_threshold),
            node_rank=0,
            node_size=1,
            sam3_path=sam3_path or HYWORLD2_SAM3_REPO_ID,
            local_files_only=bool(local_files_only),
        )
        traj_generate.run_traj_generate(generate_config)
        logs.append({"stage": "traj_generate", "mode": "native_api"})
        print("[HYWorld2 Trajectories] Stage 2/5 complete: camera trajectories generated")

        anchor_scans_written = []
        print("[HYWorld2 Trajectories] Stage 3/5: optional anchor scan")
        if bool(apply_anchor_scan) and int(anchor_scan_topk) > 0:
            anchor_scans_written = _write_anchor_scans(
                scene,
                topk=int(anchor_scan_topk),
                min_distance=float(anchor_scan_min_distance),
                min_separation=float(anchor_scan_min_separation),
                yaw_degrees=float(anchor_scan_yaw_degrees),
                nframe=int(nframe),
            )
        else:
            print("[HYWorld2 Trajectories] Anchor scan disabled")
        print(f"[HYWorld2 Trajectories] Stage 3/5 complete: {len(anchor_scans_written)} scan camera file(s)")

        if int(render_processes) not in (0, 1):
            print("[HYWorld2 Trajectories] render_processes is ignored in native mode; using one in-process renderer.")
        print("[HYWorld2 Trajectories] Stage 4/5: rendering trajectories natively with 1 process")
        render_config = Namespace(target_path=str(scene), seed=int(seed), node_rank=0, node_size=1,
                                  disable_vlm_caption=True, points_per_pixel=int(points_per_pixel),
                                  global_pcd_voxel_size=float(global_pcd_voxel_size))
        traj_render.run_traj_render(render_config, rank=0, world_size=1, local_rank=0)
        logs.append({"stage": "traj_render", "mode": "native_api", "world_size": 1})
        print("[HYWorld2 Trajectories] Stage 4/5 complete: render.mp4/render_mask.mp4 generated")

        sorted_output = self._sort(
            workspace, generated=True, captions_written=[],
            anchor_scans_written=anchor_scans_written, logs=logs,
        )
        data = sorted_output[0]
        render_list = data["render_list"]
        print(f"[HYWorld2 Trajectories] Stage 5/5: sorted {len(render_list)} trajectory render(s); captions deferred to World Expansion")
        _hyworld2_write_json_file(
            _hyworld2_trajectory_state_path(scene),
            {
                "panorama": _hyworld2_image_file_pixel_fingerprint(scene / "panorama.png"),
                "settings_signature": settings_signature,
                "settings": settings_state,
                "render_count": len(render_list),
                "captions_written": 0,
                "anchor_scans_written": len(anchor_scans_written),
            },
        )
        data["planner_context_written"] = planner_written
        return (data, _safe_json_dumps(data))


class HYWorld2TrajectoriesTest(HYWorld2Trajectories):
    """Experimental trajectory node with no captioning or LLM dependency."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"workspace": ("HYWORLD2_WORKSPACE",)},
            "optional": {
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**31 - 1, "control_after_generate": "fixed"}),
                "scene_type": (["indoor", "outdoor"], {"default": "indoor"}),
                "image_width": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 8,
                                           "tooltip": "Saved trajectory frame width; camera intrinsics are scaled to this width."}),
                "image_height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 8,
                                            "tooltip": "Saved trajectory frame height; camera intrinsics are scaled to this height."}),
                "fov_x": ("FLOAT", {"default": 90.0, "min": 10.0, "max": 170.0, "step": 1.0,
                                        "tooltip": "Horizontal field of view in degrees. Vertical FOV is derived from the final aspect ratio."}),
                "points_per_pixel": ("INT", {"default": 10, "min": 1, "max": 64, "step": 1,
                                                   "tooltip": "Point layers per pixel. 8-12 is faster and uses less VRAM; 20 matches the original."}),
                "global_pcd_voxel_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001,
                                                        "tooltip": "Render-only voxel downsample size in scene units; 0 disables it."}),
                "apply_anchor_scan": ("BOOLEAN", {"default": True}),
                "anchor_scan_topk": ("INT", {"default": 2, "min": 0, "max": 32}),
                "anchor_scan_min_clearance": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.05,
                                                        "tooltip": "Minimum distance from camera to scene surfaces, as a multiple of median depth."}),
                "anchor_scan_min_separation": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 10.0, "step": 0.05}),
                "anchor_scan_yaw_degrees": ("FLOAT", {"default": 360.0, "min": 1.0, "max": 360.0, "step": 1.0}),
                # Keep restored widgets appended so existing serialized test
                # nodes retain the value order of every current setting.
                "additional_nav_traj": ("BOOLEAN", {"default": False}),
                "extreme_detail_traj": ("BOOLEAN", {"default": False}),
                "detail_object_limit": ("INT", {"default": 6, "min": 1, "max": 16}),
            },
        }

    def _sort(self, workspace, generated=False, captions_written=None, anchor_scans_written=None, logs=None):
        """Keep every rendered regular trajectory as well as optional anchor scans."""
        result = super()._sort(
            workspace, generated=generated, captions_written=captions_written,
            anchor_scans_written=anchor_scans_written, logs=logs,
        )
        data = result[0]
        render_root = Path(workspace["scene_dir"]) / "render_results"

        def order(path):
            path = Path(path)
            view_id, traj_id = path.parts[-3], path.parts[-2]
            if view_id.startswith("view"):
                return (0, view_id, int(traj_id[4:]) if traj_id.startswith("traj") and traj_id[4:].isdigit() else 999)
            if view_id.startswith("target"):
                return (1, view_id, traj_id)
            if view_id.startswith("reconstruct") and traj_id != "traj1":
                return (2, view_id, traj_id)
            if view_id.startswith("wonder"):
                return (3, view_id, traj_id)
            if view_id.startswith("reconstruct"):
                return (4, view_id, traj_id)
            return (9, view_id, traj_id)

        all_renders = [str(path) for path in sorted(render_root.glob("**/render.mp4"), key=order)]
        data["render_list"] = all_renders
        data["count"] = len(all_renders)
        data["regular_render_count"] = sum(
            1 for path in all_renders if Path(path).parts[-3].startswith("view")
        )
        data["anchor_render_count"] = sum(
            1 for path in all_renders if Path(path).parts[-3].startswith("wonder_scan_")
        )
        print(
            "[HYWorld2 Trajectories Test] Output trajectories: "
            f"regular={data['regular_render_count']}, anchors={data['anchor_render_count']}, "
            f"total={data['count']}"
        )
        return (data, _safe_json_dumps(data))

    def run(self, workspace, seed=1, scene_type="indoor", image_width=640, image_height=480,
            fov_x=90.0, points_per_pixel=10,
            global_pcd_voxel_size=0.0, apply_anchor_scan=True, anchor_scan_topk=2,
            anchor_scan_min_clearance=0.15, anchor_scan_min_separation=0.75,
            anchor_scan_yaw_degrees=360.0, additional_nav_traj=False,
            extreme_detail_traj=False, detail_object_limit=6, **kwargs):
        from hyworld2.worldgen.src.general_utils import adjust_image_size
        final_h, final_w = adjust_image_size(int(image_height), int(image_width))
        # Preserve square pixels: fy == fx in pixel units. This makes vertical FOV
        # follow the chosen resolution instead of retaining the old 90-degree constant.
        fov_y = np.rad2deg(2.0 * np.arctan(np.tan(np.deg2rad(float(fov_x)) * 0.5) * final_h / final_w))
        return super().run(
            workspace, seed=seed, scene_type=scene_type,
            additional_nav_traj=bool(additional_nav_traj),
            extreme_detail_traj=bool(extreme_detail_traj),
            detail_object_limit=int(detail_object_limit),
            apply_nav_traj=False, force_vlm=False, nframe=21,
            image_width=image_width, image_height=image_height, fov_x=fov_x, fov_y=fov_y,
            apply_anchor_scan=apply_anchor_scan, anchor_scan_topk=anchor_scan_topk,
            anchor_scan_min_distance=anchor_scan_min_clearance,
            anchor_scan_min_separation=anchor_scan_min_separation,
            anchor_scan_yaw_degrees=anchor_scan_yaw_degrees,
            points_per_pixel=points_per_pixel, global_pcd_voxel_size=global_pcd_voxel_size,
            no_llm_mode=True,
        )


class HYWorld2MemoryBank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
            },
            "optional": {
                "image_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "nframe": ("INT", {"default": 0, "min": 0, "max": 257}),
                "max_reference": ("INT", {"default": 8, "min": 1, "max": 64}),
                "align_nframe": ("INT", {"default": 8, "min": 1, "max": 64}),
                "downsampled_pts": ("INT", {"default": 2_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "kb_anomaly_percentile": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 100.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING", "IMAGE")
    RETURN_NAMES = ("memory_bank", "info", "memory_images")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    @classmethod
    def IS_CHANGED(cls, workspace, trajectory_set, **kwargs):
        if not isinstance(workspace, dict) or not workspace.get("scene_dir"):
            debug_payload = {
                "workspace": "missing_or_unresolved",
                "workspace_type": type(workspace).__name__,
                "trajectory_set_type": type(trajectory_set).__name__,
                "kwargs": {key: kwargs[key] for key in sorted(kwargs)},
            }
            _hy_cache_debug("MemoryBank", "IS_CHANGED", debug_payload)
            return _safe_json_dumps(debug_payload)
        scene = Path(workspace["scene_dir"])
        render_list = []
        if isinstance(trajectory_set, dict):
            render_list = [str(path) for path in trajectory_set.get("render_list", [])]
        render_payload = {
            "trajectory_count": len(render_list),
            "render_list": render_list,
        }
        kwargs_payload = {key: kwargs[key] for key in sorted(kwargs)}
        debug_payload = {
            "scene": str(scene),
            "result_name": str(workspace.get("result_name", "")),
            "workspace_scene_type": str(workspace.get("scene_type", "")),
            "workspace_panorama": workspace.get("panorama") or {},
            "kwargs": kwargs_payload,
            "trajectory_count": len(render_list),
            "render_first": render_list[0] if render_list else "",
            "render_last": render_list[-1] if render_list else "",
            "render_list_hash": hashlib.sha256(_safe_json_dumps(render_list).encode("utf-8", errors="ignore")).hexdigest()[:16],
            "trajectory_generated": bool(trajectory_set.get("generated", False)) if isinstance(trajectory_set, dict) else None,
        }
        _hy_cache_debug("MemoryBank", "IS_CHANGED", debug_payload)
        # Keep this key stable across HYWorld2 Trajectories' generated-vs-cache-hit
        # output metadata. A downstream trainer failure must not invalidate
        # MemoryBank, PrepareWorldMirrorBatch, WorldMirror, or MemoryAlignment.
        state = [
            str(scene),
            str(workspace.get("result_name", "")),
            _safe_json_dumps(kwargs_payload),
            _safe_json_dumps(render_payload),
        ]
        return "|".join(state)

    def run(self, workspace, trajectory_set, image_width=0, image_height=0, nframe=0, max_reference=8, align_nframe=8, downsampled_pts=2_000_000, kb_anomaly_percentile=90.0):
        _hy_log("Memory Bank", "Stage 1/3: initializing memory bank")
        _ensure_worldgen_path()
        from hyworld2.worldgen.src.retrieval_wm import PanoramaMemoryBank

        scene = Path(workspace["scene_dir"])
        traj_workspace = trajectory_set.get("workspace", {}) if isinstance(trajectory_set, dict) else {}
        traj_scene = Path(traj_workspace.get("scene_dir", scene))
        if traj_scene.resolve() != scene.resolve():
            raise ValueError(
                "HYWorld2 Memory Bank got workspace and trajectory_set from different scene directories:\n"
                f"- workspace: {scene}\n"
                f"- trajectory_set: {traj_scene}"
            )
        if int(trajectory_set.get("count", 0)) <= 0:
            raise ValueError(
                "HYWorld2 Memory Bank requires a non-empty HYWorld2 Trajectories output. "
                "Connect HYWorld2 Trajectories.trajectory_set and let it build or reuse the trajectory workspace first."
            )
        missing = _hyworld2_missing_memory_prerequisites(scene)
        if missing:
            raise FileNotFoundError(
                "HYWorld2 Memory Bank requires completed HYWorld2 Trajectories base geometry before initialization. "
                "Connect HYWorld2 Trajectories.trajectory_set to this node so Comfy executes trajectories before Memory Bank. "
                "Missing:\n"
                + "\n".join(f"- {path}" for path in missing)
            )
        if image_width <= 0 or image_height <= 0:
            _hy_log("Memory Bank", "Stage 2/3: resolving image size from trajectory start frame or panorama")
            from imagesize import get as image_size

            start_frames = sorted((scene / "render_results").glob("*/start_frame.png"))
            if start_frames:
                image_width, image_height = image_size(str(start_frames[0]))
                _hy_log("Memory Bank", f"Using start frame size {image_width}x{image_height}: {start_frames[0]}")
            else:
                pano = scene / "panorama.png"
                image_width, image_height = image_size(str(pano))
                _hy_log("Memory Bank", f"Using panorama size {image_width}x{image_height}: {pano}")
        else:
            _hy_log("Memory Bank", f"Stage 2/3: using explicit image size {image_width}x{image_height}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _hy_log("Memory Bank", f"Stage 3/3: constructing PanoramaMemoryBank on {device} (pts_num={int(downsampled_pts)})")
        bank = PanoramaMemoryBank(
            root_path=str(scene),
            image_width=int(image_width),
            image_height=int(image_height),
            device=device,
            nframe=int(nframe) if int(nframe) > 0 else 21,
            max_reference=int(max_reference),
            align_nframe=int(align_nframe),
            rank=0,
            world_size=1,
            results_name=workspace.get("result_name", "worldstereo-memory-dmd"),
            valid_threshold=0.15,
            pts_num=int(downsampled_pts),
            kb_anomaly_percentile=float(kb_anomaly_percentile),
        )
        state = {"workspace": workspace, "bank": bank, "device": str(device), "image_width": int(image_width), "image_height": int(image_height)}
        memory_images = _pil_list_to_image_tensor(getattr(bank, "ref_frames", []))
        if memory_images.numel() == 0:
            memory_images = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        info = {
            "scene_dir": str(scene),
            "device": str(device),
            "memory_size": int(bank.mem_size),
            "results_path": bank.results_path,
            "memory_image_count": int(memory_images.shape[0]),
            "memory_frame_names_preview": list(getattr(bank, "fnames", []))[:16],
        }
        _hy_log("Memory Bank", f"Memory bank ready: memory_size={int(bank.mem_size)}, results_path={bank.results_path}")
        return (state, _safe_json_dumps(info), memory_images)


class HYWorld2WorldExpansion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "memory_bank": ("HYWORLD2_MEMORY_BANK",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
                "model": ("WORLDSTEREO_MODEL",),
            },
            "optional": {
                "qwen_model_id": (_qwenvl_model_names(), {"default": HYWORLD2_QWENVL_DEFAULT}),
                "qwen_quantization": (HYWORLD2_QWENVL_QUANTIZATION, {"default": HYWORLD2_QWENVL_DEFAULT_QUANTIZATION}),
                "qwen_attention_mode": (HYWORLD2_QWENVL_ATTENTION, {"default": "auto"}),
                "qwen_max_image_edge": ("INT", {"default": HYWORLD2_QWENVL_MAX_IMAGE_EDGE, "min": 128, "max": 4096, "step": 64}),
                "qwen_max_new_tokens": ("INT", {"default": 192, "min": 16, "max": 2048, "step": 16}),
                "qwen_keep_model_loaded": ("BOOLEAN", {"default": True}),
                "qwen_frame_count": ("INT", {"default": 4, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**31 - 1, "control_after_generate": "fixed"}),
                "max_trajectories": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "manual_caption": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "If non-empty, use this prompt for every trajectory and skip Qwen/VL caption analysis entirely.",
                }),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING")
    RETURN_NAMES = ("memory_bank", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def _ensure_captions(
        self,
        workspace,
        render_list,
        caption_mode,
        qwen_model_id,
        qwen_device,
        qwen_max_new_tokens,
        qwen_quantization=HYWORLD2_QWENVL_DEFAULT_QUANTIZATION,
        qwen_attention_mode="auto",
        qwen_keep_model_loaded=True,
        qwen_cpu_offload=False,
        qwen_max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        qwen_frame_count=4,
    ):
        if caption_mode == "existing_files_only":
            _hy_log("World Expansion", "Caption stage: existing_files_only, not generating captions")
            return []
        qwen_device = "auto"
        qwen_cpu_offload = False
        qwen = HYWorld2QwenVL()
        written = []
        _hy_log("World Expansion", f"Caption stage: mode={caption_mode}, trajectories={len(render_list)}, model={qwen_model_id}, cpu_offload={bool(qwen_cpu_offload)}")
        for render_path in render_list:
            traj_dir = Path(render_path).parent
            caption_path = traj_dir / "traj_caption.json"
            if caption_path.exists() and caption_mode != "qwenvl_overwrite":
                _hy_log("World Expansion", f"Caption stage: reusing {caption_path}")
                continue
            _hy_log("World Expansion", f"Caption stage: generating caption for {render_path}")
            frames = _load_video_frames(render_path)
            sample = []
            if frames:
                sample = [frames[0]]
                if len(frames) > 2:
                    sample.append(frames[len(frames) // 2])
                if len(frames) > 1:
                    sample.append(frames[-1])
                if int(qwen_frame_count) > len(sample):
                    idx = np.linspace(0, len(frames) - 1, min(int(qwen_frame_count), len(frames)), dtype=int)
                    sample = [frames[i] for i in idx]
            start_frame = traj_dir.parent / "start_frame.png"
            if start_frame.exists():
                sample.insert(0, Image.open(start_frame).convert("RGB"))
            if not sample:
                raise FileNotFoundError(f"Cannot caption trajectory; no render frames found for {render_path}")
            prompt = (
                "Create a concise photorealistic video generation prompt for this HYWorld2 camera "
                "trajectory. Describe stable scene layout, materials, lighting, and newly visible areas. "
                "Return only the prompt text, no JSON and no commentary."
            )
            text = qwen._generate(
                qwen_model_id,
                prompt,
                images=_pil_list_to_image_tensor(sample[: max(1, int(qwen_frame_count))]),
                device=qwen_device,
                max_new_tokens=qwen_max_new_tokens,
                max_image_edge=int(qwen_max_image_edge),
                quantization=qwen_quantization,
                attention_mode=qwen_attention_mode,
                temperature=0.6,
                top_p=0.9,
                num_beams=1,
                repetition_penalty=1.2,
                keep_model_loaded=qwen_keep_model_loaded,
                cpu_offload=qwen_cpu_offload,
                seed=1,
            )
            if not text.strip():
                raise RuntimeError(f"QwenVL returned an empty caption for {render_path}")
            with open(caption_path, "w", encoding="utf-8") as handle:
                json.dump({"prompt": text.strip(), "source": "HYWorld2 World Expansion QwenVL"}, handle, indent=2)
            written.append(str(caption_path))
            _hy_log("World Expansion", f"Caption stage: wrote {caption_path}")
        return written

    def run(
        self,
        workspace,
        memory_bank,
        trajectory_set,
        model,
        qwen_model_id=HYWORLD2_QWENVL_DEFAULT,
        qwen_quantization=HYWORLD2_QWENVL_DEFAULT_QUANTIZATION,
        qwen_attention_mode="auto",
        qwen_device="auto",
        qwen_cpu_offload=False,
        qwen_max_image_edge=HYWORLD2_QWENVL_MAX_IMAGE_EDGE,
        qwen_max_new_tokens=192,
        qwen_keep_model_loaded=True,
        qwen_frame_count=4,
        seed=1,
        max_trajectories=0,
        manual_caption="",
    ):
        qwen_device = "auto"
        qwen_cpu_offload = False
        caption_mode = "qwenvl_missing"
        _hy_log("World Expansion", "Stage 1/6: preparing WorldStereo memory expansion")
        _ensure_worldgen_path()
        from hyworld2.worldgen.src.data_utils import load_mutli_traj_dataset

        bank = memory_bank["bank"]
        device = torch.device(memory_bank.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        pipeline = model["pipeline"]
        cfg = _worldstereo_cfg(model)
        model_type = model.get("model_type") or workspace.get("result_name", "worldstereo-memory-dmd")
        if int(getattr(bank, "nframe", 0)) != int(getattr(cfg, "nframe", getattr(bank, "nframe", 21))):
            bank.nframe = int(getattr(cfg, "nframe", bank.nframe))
        render_list = list(trajectory_set.get("render_list", []))
        if int(max_trajectories) > 0:
            render_list = render_list[: int(max_trajectories)]
        state_path = _hyworld2_world_expansion_state_path(workspace["scene_dir"])
        expansion_state = _hyworld2_read_json_file(state_path, default={}) or {}
        manual_caption = str(manual_caption or "").strip()
        caption_source = "manual" if manual_caption else "qwenvl"
        caption_signature = hashlib.sha256(
            (f"{caption_source}:" + manual_caption).encode("utf-8", errors="ignore")
        ).hexdigest()
        legacy_seed = 1
        state_seed = expansion_state.get("seed", legacy_seed)
        seed_matches = int(state_seed) == int(seed)
        if not expansion_state:
            _hy_log("World Expansion", f"No expansion seed state found; treating existing result videos as legacy seed={legacy_seed}")
            seed_matches = int(seed) == legacy_seed
        elif seed_matches:
            _hy_log("World Expansion", f"Expansion cache seed matches: seed={int(seed)}")
        else:
            _hy_log("World Expansion", f"Expansion cache seed changed: cached={state_seed}, requested={int(seed)}; result videos will be regenerated")
        cached_caption_signature = expansion_state.get("caption_signature")
        if manual_caption:
            caption_matches = cached_caption_signature == caption_signature
        else:
            # Preserve compatibility with old auto-caption states which predate
            # caption_signature; a manual prompt must always match explicitly.
            caption_matches = cached_caption_signature in (None, caption_signature)
        if not caption_matches:
            _hy_log(
                "World Expansion",
                f"Caption source/text changed ({caption_source}); result videos will be regenerated",
            )
        render_items = []
        pending_render_list = []
        existing_result_count = 0
        for render_path in render_list:
            render_parts = Path(render_path).parts
            view_id, traj_id = render_parts[-3], render_parts[-2]
            traj_dir = Path(workspace["scene_dir"]) / "render_results" / view_id / traj_id
            result_path = traj_dir / f"{model_type}_result.mp4"
            has_result = result_path.is_file() and result_path.stat().st_size > 0
            can_reuse_result = bool(has_result and seed_matches and caption_matches)
            render_items.append(
                {
                    "render_path": render_path,
                    "view_id": view_id,
                    "traj_id": traj_id,
                    "traj_dir": traj_dir,
                    "result_path": result_path,
                    "has_result": bool(has_result),
                    "can_reuse_result": bool(can_reuse_result),
                }
            )
            if can_reuse_result:
                existing_result_count += 1
            else:
                pending_render_list.append(render_path)
        _hy_log("World Expansion", f"Stage 2/6: trajectory count={len(render_list)}, device={device}, model_type={model_type}")
        _hy_log(
            "World Expansion",
            f"Seed-valid WorldStereo result videos: {existing_result_count}/{len(render_list)}; "
            f"pending generation: {len(pending_render_list)}",
        )
        _hy_log("World Expansion", "Stage 3/6: ensuring trajectory captions")
        captions_written = []
        if pending_render_list:
            if manual_caption:
                _hy_log(
                    "World Expansion",
                    f"Manual caption provided; skipping Qwen/VL analysis for {len(pending_render_list)} trajectory(s)",
                )
                for render_path in pending_render_list:
                    caption_path = Path(render_path).parent / "traj_caption.json"
                    with open(caption_path, "w", encoding="utf-8") as handle:
                        json.dump(
                            {"prompt": manual_caption, "source": "HYWorld2 World Expansion manual caption"},
                            handle,
                            indent=2,
                            ensure_ascii=False,
                        )
                    captions_written.append(str(caption_path))
            else:
                captions_written = self._ensure_captions(
                    workspace,
                    pending_render_list,
                    caption_mode,
                    qwen_model_id,
                    qwen_device,
                    int(qwen_max_new_tokens),
                    qwen_quantization,
                    qwen_attention_mode,
                    bool(qwen_keep_model_loaded),
                    bool(qwen_cpu_offload),
                    int(qwen_max_image_edge),
                    int(qwen_frame_count),
                )
        else:
            _hy_log("World Expansion", "Stage 3/6: all result videos exist; caption generation skipped")
        _hy_log("World Expansion", f"Stage 3/6 complete: captions_written={len(captions_written)}")
        # keep_model_loaded is useful while captioning many trajectories, but the
        # VLM must never overlap the much larger WorldStereo generation stage.
        # On a 16 GiB card that overlap leaves zero free VRAM and Accelerate OOMs
        # while moving even a small offloaded WorldStereo block to CUDA.
        _hy_log("World Expansion", "Releasing Qwen/VL before WorldStereo prompt encoding and generation")
        HYWorld2QwenVL._clear_cache()
        _hy_log("World Expansion", "Stage 4/6: encoding prompt cache")
        prompt_cache = _build_prompt_cache(model, workspace, pending_render_list, model_type, device) if pending_render_list else {}
        _hy_log("World Expansion", f"Stage 4/6 complete: cached {len(prompt_cache)} prompt embedding set(s)")
        generator = torch.Generator(device=device).manual_seed(int(seed))
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        completed = []
        _hy_log("World Expansion", "Stage 5/6: generating trajectory videos and updating memory")
        for item in render_items:
            render_path = item["render_path"]
            view_id = item["view_id"]
            traj_id = item["traj_id"]
            traj_dir = item["traj_dir"]
            result_path = item["result_path"]
            _hy_log("World Expansion", f"Trajectory {len(completed)+1}/{len(render_list)}: {view_id}/{traj_id}")
            camera_data = json.load(open(traj_dir / "camera.json", "r", encoding="utf-8"))
            tar_w2cs = torch.from_numpy(np.asarray(camera_data["extrinsic"], dtype=np.float32)).to(device)
            tar_Ks = torch.from_numpy(np.asarray(camera_data["intrinsic"], dtype=np.float32)).to(device)
            if item["can_reuse_result"]:
                _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: reusing existing result {result_path}")
                frames = _load_video_frames(result_path)
                update_w2cs, update_Ks = _sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(frames))
                bank.update_memory(frames, update_w2cs, update_Ks, view_id=view_id, traj_id=traj_id)
                completed.append(str(result_path))
                continue
            _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: retrieving references from memory bank")
            retrieved_frames, ref_index, ref_index_dict, ref_w2cs, _ = bank.retrieval(tar_w2cs, tar_Ks, view_id=view_id, traj_id=traj_id)
            memory_dir = traj_dir / "memory_inputs"
            _ensure_dir(memory_dir)
            _export_video(retrieved_frames / 255.0, memory_dir / f"{model_type}.mp4", fps=16)
            with open(memory_dir / f"{model_type}_ref_index.json", "w", encoding="utf-8") as handle:
                json.dump(ref_index_dict, handle, indent=2)
            with open(memory_dir / f"{model_type}_ref_w2cs.json", "w", encoding="utf-8") as handle:
                json.dump(ref_w2cs.detach().cpu().numpy().tolist(), handle, indent=2)
            meta_data = load_mutli_traj_dataset(
                cfg=cfg,
                input_path=str(Path(workspace["scene_dir"]) / "render_results"),
                output_path=str(Path(workspace["scene_dir"]) / "render_results"),
                view_id=view_id,
                traj_id=traj_id,
                device=device,
                ref_index=ref_index,
                model_type=model_type,
                task_type="panorama",
            )
            pipeline_kwargs = {k: v for k, v in meta_data.items() if v is not None}
            pipeline_kwargs.update(generator=generator, output_type="pt", latent_cond_mode=getattr(cfg, "latent_cond_mode", "first_frame_only"))
            cached_prompt_embeds, cached_negative_prompt_embeds = prompt_cache[(view_id, traj_id)]
            pipeline_kwargs.pop("prompt", None)
            pipeline_kwargs.update(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=cached_prompt_embeds.to(device),
                negative_prompt_embeds=cached_negative_prompt_embeds.to(device) if cached_negative_prompt_embeds is not None else None,
            )
            if model_type == "worldstereo-memory-dmd":
                pipeline_kwargs["mode"] = "test"
                _slice_render_conditioning_to_keyframes(pipeline_kwargs)
            else:
                pipeline_kwargs["guidance_scale"] = 5.0
            with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: running WorldStereo generation")
                output = pipeline(**pipeline_kwargs).frames[0].float()
            frames_np = output.permute(0, 2, 3, 1).detach().cpu().clamp(0, 1).numpy()
            _export_video(frames_np, result_path, fps=16)
            gen_frames = _load_video_frames(result_path)
            update_w2cs, update_Ks = _sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(gen_frames))
            bank.update_memory(gen_frames, update_w2cs, update_Ks, view_id=view_id, traj_id=traj_id)
            completed.append(str(result_path))
            _hy_log("World Expansion", f"Trajectory {view_id}/{traj_id}: wrote {result_path} and updated memory")
            del output, pipeline_kwargs, meta_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        memory_bank["bank"] = bank
        del pipeline
        _hy_log("World Expansion", "Stage 6/6: releasing model memory")
        _release_model_memory("HYWorld2 World Expansion")
        _hyworld2_write_json_file(
            state_path,
            {
                "seed": int(seed),
                "model_type": str(model_type),
                "render_count": len(render_list),
                "completed_count": len(completed),
                "result_paths": completed,
                "caption_source": caption_source,
                "caption_signature": caption_signature,
            },
        )
        _hy_log("World Expansion", f"World expansion complete: completed={len(completed)}")
        return (
            memory_bank,
            _safe_json_dumps(
                {
                    "completed": completed,
                    "count": len(completed),
                    "captions_written": captions_written,
                    "caption_source": caption_source,
                    "seed": int(seed),
                    "seed_cache_action": "reuse_existing" if not pending_render_list else "generated_pending",
                    "state_path": str(state_path),
                }
            ),
        )


def _comfy_node_output(value, index=0):
    if hasattr(value, "result"):
        result = value.result
        return result[index]
    return value[index]


def _comfy_node_class(name):
    import importlib

    comfy_nodes = importlib.import_module("nodes")
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    node_cls = mappings.get(name)
    if node_cls is None and hasattr(comfy_nodes, name):
        node_cls = getattr(comfy_nodes, name)
    if node_cls is None:
        fallback_modules = {
            "ReferenceLatent": "comfy_extras.nodes_edit_model",
            "EmptyFlux2LatentImage": "comfy_extras.nodes_flux",
            "Flux2Scheduler": "comfy_extras.nodes_flux",
            "KSamplerSelect": "comfy_extras.nodes_custom_sampler",
            "CFGGuider": "comfy_extras.nodes_custom_sampler",
            "RandomNoise": "comfy_extras.nodes_custom_sampler",
            "SamplerCustomAdvanced": "comfy_extras.nodes_custom_sampler",
            "ImageScaleToTotalPixels": "comfy_extras.nodes_post_processing",
        }
        module_name = fallback_modules.get(name)
        if module_name:
            module = importlib.import_module(module_name)
            node_cls = getattr(module, name, None)
    if node_cls is None:
        raise RuntimeError(f"Comfy node class '{name}' is not available. Update ComfyUI or enable the required official node extension.")
    return node_cls


def _run_comfy_node(name, method_name, *args, **kwargs):
    node = _comfy_node_class(name)()
    method = getattr(node, method_name, None)
    if method is None:
        method = getattr(node, "execute", None)
    if method is None:
        raise RuntimeError(f"Comfy node '{name}' has no callable '{method_name}' or 'execute' method.")
    return _comfy_node_output(method(*args, **kwargs), 0)


def _image_tensor_size(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {type(image).__name__}: {getattr(image, 'shape', None)}")
    return int(image.shape[2]), int(image.shape[1])


def _pil_to_single_image_tensor(image):
    return _pil_list_to_image_tensor([image.convert("RGB")])


def _image_tensor_first_to_pil(image):
    frames = _image_tensor_to_pil_list(image)
    if not frames:
        raise RuntimeError("Klein pipeline returned an empty IMAGE output.")
    return frames[0].convert("RGB")


def _resize_pil(image, size):
    image = image.convert("RGB")
    if image.size == tuple(size):
        return image
    return image.resize(tuple(size), Image.Resampling.LANCZOS)


def _scale_intrinsics_to_size(Ks, old_size, new_size):
    old_w, old_h = max(1, int(old_size[0])), max(1, int(old_size[1]))
    new_w, new_h = max(1, int(new_size[0])), max(1, int(new_size[1]))
    scaled = Ks.clone().float()
    scaled[..., 0, :] *= float(new_w) / float(old_w)
    scaled[..., 1, :] *= float(new_h) / float(old_h)
    return scaled


def _make_memory_bank_points(width, height, device):
    x = torch.arange(int(width), dtype=torch.float32, device=device)
    y = torch.arange(int(height), dtype=torch.float32, device=device)
    points = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)
    points = points.permute(1, 0, 2).reshape(-1, 2)
    pad = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    return torch.cat([points, pad], dim=-1)


def _normalize_memory_bank_image_size(bank, memory_bank, target_size):
    target_w, target_h = int(target_size[0]), int(target_size[1])
    old_w = int(getattr(bank, "image_width", memory_bank.get("image_width", target_w)) or target_w)
    old_h = int(getattr(bank, "image_height", memory_bank.get("image_height", target_h)) or target_h)
    expected_points = target_w * target_h
    current_points = getattr(bank, "points", None)
    points_ok = isinstance(current_points, torch.Tensor) and int(current_points.shape[0]) == int(expected_points)
    if old_w == target_w and old_h == target_h and points_ok:
        return (old_w, old_h)

    _hy_log(
        "Klein Expansion",
        f"Normalizing memory bank size: {old_w}x{old_h} -> {target_w}x{target_h}; "
        f"points={int(current_points.shape[0]) if isinstance(current_points, torch.Tensor) else 0}->{expected_points}",
    )
    if old_w != target_w or old_h != target_h:
        bank.ref_frames = [_resize_pil(frame, (target_w, target_h)) for frame in getattr(bank, "ref_frames", [])]
    if (old_w != target_w or old_h != target_h) and isinstance(getattr(bank, "ref_Ks", None), torch.Tensor) and bank.ref_Ks.numel() > 0:
        bank.ref_Ks = _scale_intrinsics_to_size(bank.ref_Ks, (old_w, old_h), (target_w, target_h)).to(bank.ref_Ks.device)
    bank.image_width = target_w
    bank.image_height = target_h
    device = getattr(bank, "device", None)
    if device is None:
        ref_Ks = getattr(bank, "ref_Ks", None)
        device = ref_Ks.device if isinstance(ref_Ks, torch.Tensor) else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bank.points = _make_memory_bank_points(target_w, target_h, device)
    memory_bank["image_width"] = target_w
    memory_bank["image_height"] = target_h
    return (old_w, old_h)


class HYWorld2KleinWorldExpansion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "memory_bank": ("HYWORLD2_MEMORY_BANK",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "positive_prompt": ("STRING", {"default": "Reconstruct image based on image2", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "result_name": ("STRING", {"default": "klein9b-memory-lora"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 4096}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["euler"], {"default": "euler"}),
                "input_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 1, "min": 1, "max": 256}),
                "frames_per_trajectory": ("INT", {"default": 3, "min": 1, "max": 64}),
                "use_context_image": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**31 - 1, "control_after_generate": "fixed"}),
                "max_trajectories": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING")
    RETURN_NAMES = ("memory_bank", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def _scale_reference(self, image, input_megapixels, resolution_steps):
        return _run_comfy_node("ImageScaleToTotalPixels", "execute", image, "lanczos", float(input_megapixels), int(resolution_steps))

    def _run_klein_frame(
        self,
        model,
        clip,
        vae,
        render_frame,
        context_frame,
        positive_base,
        negative_base,
        sampler,
        seed,
        steps,
        cfg,
        input_megapixels,
        resolution_steps,
    ):
        image1 = self._scale_reference(_pil_to_single_image_tensor(render_frame), input_megapixels, resolution_steps)
        image2 = self._scale_reference(_pil_to_single_image_tensor(context_frame), input_megapixels, resolution_steps)
        width, height = _image_tensor_size(image1)

        latent1 = _run_comfy_node("VAEEncode", "encode", vae, image1)
        latent2 = _run_comfy_node("VAEEncode", "encode", vae, image2)
        positive = _run_comfy_node("ReferenceLatent", "execute", positive_base, latent1)
        positive = _run_comfy_node("ReferenceLatent", "execute", positive, latent2)
        negative = _run_comfy_node("ReferenceLatent", "execute", negative_base, latent1)
        negative = _run_comfy_node("ReferenceLatent", "execute", negative, latent2)
        guider = _run_comfy_node("CFGGuider", "execute", model, positive, negative, float(cfg))
        sigmas = _run_comfy_node("Flux2Scheduler", "execute", int(steps), width, height)
        latent = _run_comfy_node("EmptyFlux2LatentImage", "execute", width, height, 1)
        noise = _run_comfy_node("RandomNoise", "execute", int(seed))
        samples = _run_comfy_node("SamplerCustomAdvanced", "execute", noise, guider, sampler, sigmas, latent)
        image = _run_comfy_node("VAEDecode", "decode", vae, samples)
        return _image_tensor_first_to_pil(image)

    def run(
        self,
        workspace,
        memory_bank,
        trajectory_set,
        model,
        clip,
        vae,
        positive_prompt="Reconstruct image based on image2",
        negative_prompt="",
        result_name="klein9b-memory-lora",
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        input_megapixels=1.0,
        resolution_steps=1,
        frames_per_trajectory=3,
        use_context_image=True,
        seed=1,
        max_trajectories=0,
    ):
        _hy_log("Klein Expansion", "Stage 1/5: preparing Klein memory expansion")
        bank = memory_bank["bank"]
        device = torch.device(memory_bank.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        model_type = str(result_name or "klein9b-memory-lora").strip() or "klein9b-memory-lora"
        render_list = list(trajectory_set.get("render_list", []))
        if int(max_trajectories) > 0:
            render_list = render_list[: int(max_trajectories)]

        state_path = _hyworld2_klein_expansion_state_path(workspace["scene_dir"])
        expansion_state = _hyworld2_read_json_file(state_path, default={}) or {}
        settings_signature = {
            "seed": int(seed),
            "model_type": model_type,
            "positive_prompt": str(positive_prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": str(sampler_name),
            "input_megapixels": float(input_megapixels),
            "resolution_steps": int(resolution_steps),
            "frames_per_trajectory": int(frames_per_trajectory),
            "use_context_image": bool(use_context_image),
            "keyframe_selection": "anchor_scan_circular_v2",
        }
        settings_match = expansion_state.get("settings_signature") == settings_signature
        sampler = _run_comfy_node("KSamplerSelect", "execute", str(sampler_name))
        positive_base = _run_comfy_node("CLIPTextEncode", "encode", clip, str(positive_prompt or ""))
        negative_base = _run_comfy_node("CLIPTextEncode", "encode", clip, str(negative_prompt or ""))

        render_items = []
        pending_count = 0
        for render_path in render_list:
            render_parts = Path(render_path).parts
            view_id, traj_id = render_parts[-3], render_parts[-2]
            traj_dir = Path(workspace["scene_dir"]) / "render_results" / view_id / traj_id
            result_path = traj_dir / f"{model_type}_result.mp4"
            has_result = result_path.is_file() and result_path.stat().st_size > 0
            can_reuse_result = bool(has_result and settings_match)
            if not can_reuse_result:
                pending_count += 1
            render_items.append(
                {
                    "render_path": Path(render_path),
                    "view_id": view_id,
                    "traj_id": traj_id,
                    "traj_dir": traj_dir,
                    "result_path": result_path,
                    "can_reuse_result": can_reuse_result,
                }
            )

        _hy_log("Klein Expansion", f"Stage 2/5: trajectory count={len(render_items)}, pending={pending_count}, model_type={model_type}")
        completed = []
        generated_frame_count = 0
        target_size = None
        original_bank_size = (int(getattr(bank, "image_width", memory_bank.get("image_width", 0)) or 0), int(getattr(bank, "image_height", memory_bank.get("image_height", 0)) or 0))

        _hy_log("Klein Expansion", "Stage 3/5: generating/reusing trajectory videos")
        for item_index, item in enumerate(render_items):
            view_id = item["view_id"]
            traj_id = item["traj_id"]
            traj_dir = item["traj_dir"]
            result_path = item["result_path"]
            camera_data = json.load(open(traj_dir / "camera.json", "r", encoding="utf-8"))
            tar_w2cs = torch.from_numpy(np.asarray(camera_data["extrinsic"], dtype=np.float32)).to(device)
            tar_Ks = torch.from_numpy(np.asarray(camera_data["intrinsic"], dtype=np.float32)).to(device)

            _hy_log("Klein Expansion", f"Trajectory {item_index + 1}/{len(render_items)}: {view_id}/{traj_id}")
            if item["can_reuse_result"]:
                frames = _load_video_frames(result_path)
                if not frames:
                    raise RuntimeError(f"Reusable Klein result has no frames: {result_path}")
                if target_size is None:
                    target_size = frames[0].size
                    original_bank_size = _normalize_memory_bank_image_size(bank, memory_bank, target_size)
                frames = [_resize_pil(frame, target_size) for frame in frames]
                update_w2cs, update_Ks = _sample_camera_tensors_to_frame_count(tar_w2cs, tar_Ks, len(frames))
                update_Ks = _scale_intrinsics_to_size(update_Ks, original_bank_size, target_size)
                bank.update_memory(frames, update_w2cs, update_Ks, view_id=view_id, traj_id=traj_id)
                completed.append(str(result_path))
                continue

            render_frames = _load_video_frames(item["render_path"])
            if str(camera_data.get("type", "")).lower() == "anchor_scan":
                # Anchor scans are deliberately sparse directional samples (for
                # example 0/90/180/270). WorldStereo's every-fourth-frame rule
                # would collapse a four-frame scan to a single image.
                canonical_keyframe_indices = list(range(len(render_frames)))
            else:
                canonical_keyframe_indices = _worldstereo_keyframe_indices(len(render_frames)).detach().cpu().numpy().astype(np.int64).tolist()
            if (str(camera_data.get("type", "")).lower() == "anchor_scan"
                    and np.isclose(abs(float(camera_data.get("yaw_degrees", 0.0))), 360.0, atol=1e-6)
                    and 0 < int(frames_per_trajectory) < len(canonical_keyframe_indices)):
                # The final frame closes the circle and duplicates 0 degrees. Sample
                # the unique interval [0, N-1), yielding 0/5/10/15 for N=21,count=4.
                unique_count = max(1, len(canonical_keyframe_indices) - 1)
                picks = np.rint(
                    np.arange(int(frames_per_trajectory), dtype=np.float64)
                    * unique_count / int(frames_per_trajectory)
                ).astype(np.int64)
                keyframe_indices = [canonical_keyframe_indices[int(index)] for index in picks]
            else:
                keyframe_indices = _select_evenly_spaced_indices(canonical_keyframe_indices, int(frames_per_trajectory))
            if not keyframe_indices:
                raise RuntimeError(f"No keyframes selected for trajectory: {item['render_path']}")
            _hy_log(
                "Klein Expansion",
                f"Trajectory {view_id}/{traj_id}: selected {len(keyframe_indices)}/{len(canonical_keyframe_indices)} keyframe(s), "
                f"render_indices={keyframe_indices}",
            )

            generated_frames = []
            previous_frame = None
            for local_index, render_index in enumerate(keyframe_indices):
                render_frame = render_frames[int(render_index)].convert("RGB")
                if bool(use_context_image) and previous_frame is not None:
                    context_frame = previous_frame
                else:
                    context_frame = Image.new("RGB", render_frame.size, (0, 0, 0))
                frame_seed = int(seed) + generated_frame_count
                _hy_log(
                    "Klein Expansion",
                    f"Trajectory {view_id}/{traj_id}: Klein frame {local_index + 1}/{len(keyframe_indices)}, "
                    f"seed={frame_seed}, context={'previous' if bool(use_context_image) and previous_frame is not None else 'black'}",
                )
                generated = self._run_klein_frame(
                    model,
                    clip,
                    vae,
                    render_frame,
                    context_frame,
                    positive_base,
                    negative_base,
                    sampler,
                    frame_seed,
                    int(steps),
                    float(cfg),
                    float(input_megapixels),
                    int(resolution_steps),
                )
                if target_size is None:
                    target_size = generated.size
                    original_bank_size = _normalize_memory_bank_image_size(bank, memory_bank, target_size)
                generated = _resize_pil(generated, target_size)
                generated_frames.append(generated)
                previous_frame = generated
                generated_frame_count += 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _export_video([np.asarray(frame, dtype=np.float32) / 255.0 for frame in generated_frames], result_path, fps=16)
            selected_w2cs = tar_w2cs[torch.as_tensor(keyframe_indices, dtype=torch.long, device=tar_w2cs.device)]
            selected_Ks = tar_Ks[torch.as_tensor(keyframe_indices, dtype=torch.long, device=tar_Ks.device)]
            selected_Ks = _scale_intrinsics_to_size(selected_Ks, original_bank_size, target_size)
            bank.update_memory(generated_frames, selected_w2cs, selected_Ks, view_id=view_id, traj_id=traj_id)
            completed.append(str(result_path))
            _hy_log("Klein Expansion", f"Trajectory {view_id}/{traj_id}: wrote {result_path} and updated memory")

        memory_bank["bank"] = bank
        _hy_log("Klein Expansion", "Stage 4/5: writing expansion state")
        _hyworld2_write_json_file(
            state_path,
            {
                "seed": int(seed),
                "model_type": model_type,
                "render_count": len(render_list),
                "completed_count": len(completed),
                "result_paths": completed,
                "target_size": list(target_size) if target_size else None,
                "input_megapixels": float(input_megapixels),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": str(sampler_name),
                "frames_per_trajectory": int(frames_per_trajectory),
                "use_context_image": bool(use_context_image),
                "settings_signature": settings_signature,
            },
        )
        _hy_log("Klein Expansion", f"Stage 5/5 complete: completed={len(completed)}, target_size={target_size}")
        return (
            memory_bank,
            _safe_json_dumps(
                {
                    "completed": completed,
                    "count": len(completed),
                    "seed": int(seed),
                    "seed_cache_action": "reuse_existing" if pending_count == 0 else "generated_pending",
                    "state_path": str(state_path),
                    "target_size": list(target_size) if target_size else None,
                    "frames_per_trajectory": int(frames_per_trajectory),
                    "use_context_image": bool(use_context_image),
                }
            ),
        )


def _dataset_fit_image(image, size, mode="contain", fill=(0, 0, 0)):
    """Resize an RGB image to an exact dataset size without accidental distortion."""
    image = image.convert("RGB")
    target_w, target_h = int(size[0]), int(size[1])
    if image.size == (target_w, target_h):
        return image
    if mode == "stretch":
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    source_w, source_h = image.size
    scale = (max if mode == "cover" else min)(target_w / source_w, target_h / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    if mode == "cover":
        left = max(0, (resized_w - target_w) // 2)
        top = max(0, (resized_h - target_h) // 2)
        return resized.crop((left, top, left + target_w, top + target_h))
    canvas = Image.new("RGB", (target_w, target_h), fill)
    canvas.paste(resized, ((target_w - resized_w) // 2, (target_h - resized_h) // 2))
    return canvas


class HYWorld2SaveExpansionDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "trajectory_set": ("HYWORLD2_TRAJECTORY_SET",),
            },
            "optional": {
                "expansion_dependency": ("HYWORLD2_MEMORY_BANK",),
                "output_directory": ("STRING", {"default": "hyworld2_klein_dataset"}),
                "world_expansion_result_name": ("STRING", {"default": "worldstereo-memory-dmd"}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "max_frames_per_trajectory": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "panorama_fit": (["contain", "cover", "stretch"], {"default": "contain"}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_directory", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"
    OUTPUT_NODE = True

    def run(
        self,
        workspace,
        trajectory_set,
        expansion_dependency=None,
        output_directory="hyworld2_klein_dataset",
        world_expansion_result_name="worldstereo-memory-dmd",
        frame_stride=1,
        max_frames_per_trajectory=0,
        panorama_fit="contain",
        overwrite_existing=False,
    ):
        del expansion_dependency  # Optional execution-order link from World Expansion.
        scene = Path(workspace["scene_dir"])
        output = Path(str(output_directory or "hyworld2_klein_dataset")).expanduser()
        if not output.is_absolute():
            output = _output_root() / output
        output = output.resolve()
        folders = {name: output / name for name in ("img", "control1", "control2", "control3")}
        for folder in folders.values():
            _ensure_dir(folder)

        panorama_path = scene / "panorama_sr.png"
        if not panorama_path.exists():
            panorama_path = scene / "panorama.png"
        if not panorama_path.exists():
            raise FileNotFoundError(f"Dataset export requires panorama.png or panorama_sr.png: {scene}")
        panorama = Image.open(panorama_path).convert("RGB")

        result_name = str(world_expansion_result_name or "worldstereo-memory-dmd").strip()
        render_list = list(trajectory_set.get("render_list", []))
        records = []
        skipped_trajectories = []
        written = 0
        stride = max(1, int(frame_stride))
        max_frames = max(0, int(max_frames_per_trajectory))

        for render_value in render_list:
            render_path = Path(render_value)
            if len(render_path.parts) < 3:
                continue
            view_id, traj_id = render_path.parts[-3], render_path.parts[-2]
            traj_dir = scene / "render_results" / view_id / traj_id
            result_path = traj_dir / f"{result_name}_result.mp4"
            if not result_path.exists():
                skipped_trajectories.append({"trajectory": f"{view_id}/{traj_id}", "reason": f"missing {result_path.name}"})
                continue
            target_frames = _load_video_frames(result_path)
            hole_frames = _load_video_frames(render_path)
            if not target_frames or not hole_frames:
                skipped_trajectories.append({"trajectory": f"{view_id}/{traj_id}", "reason": "empty target or render video"})
                continue

            target_indices = list(range(0, len(target_frames), stride))
            if max_frames > 0:
                target_indices = target_indices[:max_frames]
            for sequence_index, target_index in enumerate(target_indices):
                # Align videos by normalized time, which remains correct if codecs or
                # generation stages emitted different frame counts.
                if len(target_frames) <= 1 or len(hole_frames) <= 1:
                    hole_index = 0
                else:
                    hole_index = int(round(target_index * (len(hole_frames) - 1) / (len(target_frames) - 1)))
                target = target_frames[target_index].convert("RGB")
                size = target.size
                control1 = _dataset_fit_image(panorama, size, mode=str(panorama_fit))
                control2 = _dataset_fit_image(hole_frames[hole_index], size, mode="stretch")
                control3 = (
                    Image.new("RGB", size, (0, 0, 0))
                    if sequence_index == 0
                    else _dataset_fit_image(target_frames[max(0, target_index - 1)], size, mode="stretch")
                )

                safe_view = re.sub(r"[^A-Za-z0-9_.-]+", "_", view_id)
                safe_traj = re.sub(r"[^A-Za-z0-9_.-]+", "_", traj_id)
                filename = f"{safe_view}__{safe_traj}__{sequence_index:06d}.png"
                paths = {name: folder / filename for name, folder in folders.items()}
                if not bool(overwrite_existing) and any(path.exists() for path in paths.values()):
                    raise FileExistsError(
                        f"Dataset sample already exists: {filename}. Enable overwrite_existing or choose another output_directory."
                    )
                target.save(paths["img"], format="PNG")
                control1.save(paths["control1"], format="PNG")
                control2.save(paths["control2"], format="PNG")
                control3.save(paths["control3"], format="PNG")
                records.append({
                    "file": filename, "view_id": view_id, "traj_id": traj_id,
                    "sequence_index": sequence_index, "target_frame": target_index,
                    "control2_frame": hole_index, "width": size[0], "height": size[1],
                    "target_video": str(result_path), "control2_video": str(render_path),
                    "panorama": str(panorama_path),
                })
                written += 1

        manifest_path = output / "metadata.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        info = {
            "dataset_directory": str(output), "samples": written,
            "trajectories_requested": len(render_list), "skipped_trajectories": skipped_trajectories,
            "folders": {key: str(value) for key, value in folders.items()},
            "manifest": str(manifest_path), "panorama_fit": str(panorama_fit),
        }
        _hy_log("Dataset Export", f"Saved {written} aligned samples to {output}")
        if written == 0:
            raise RuntimeError(f"Dataset export produced no samples: {_safe_json_dumps(info)}")
        return (str(output), _safe_json_dumps(info))


class HYWorld2PrepareWorldMirrorBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"memory_bank": ("HYWORLD2_MEMORY_BANK",)}}

    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR", "HYWORLD2_WORLDMIRROR_BATCH", "STRING")
    RETURN_NAMES = ("images", "camera_poses", "camera_intrinsics", "worldmirror_batch", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, memory_bank):
        _hy_log("Prepare WorldMirror Batch", "Stage 1/4: preparing WorldMirror export batch")
        bank = memory_bank["bank"]
        world_mirror_dir = Path(bank.root_path) / "render_results" / bank.results_path / "world_mirror_data"
        render_root = (Path(bank.root_path) / "render_results" / bank.results_path).resolve()
        world_mirror_resolved = world_mirror_dir.resolve()
        if world_mirror_dir.exists():
            if render_root not in world_mirror_resolved.parents:
                raise RuntimeError(f"Refusing to clear unexpected WorldMirror directory: {world_mirror_dir}")
            shutil.rmtree(world_mirror_dir)
        images_dir = _ensure_dir(world_mirror_dir / "images")
        _hy_log("Prepare WorldMirror Batch", f"WorldMirror directory: {world_mirror_dir}")
        cameras = {"num_cameras": 0, "extrinsics": [], "intrinsics": []}
        name_map = {}
        images = []
        poses = []
        intrs = []
        entries = []
        for gi, fname in enumerate(bank.fnames):
            view_id, traj_id, frame_id = fname.split("/")
            camera_id = f"pano-{frame_id}" if view_id.startswith("render_results") else f"{view_id}-{traj_id}-{frame_id}"
            entries.append((camera_id, fname, gi))

        # WorldMirror writes depth_NNNN by sorted image/camera id. Keep the tensor
        # batch, cameras.json, files, and name_map in that exact same order.
        _hy_log("Prepare WorldMirror Batch", f"Stage 2/4: exporting {len(entries)} reference frame(s)")
        entries.sort(key=lambda item: item[0])
        for index, (camera_id, fname, gi) in enumerate(entries):
            view_id, traj_id, frame_id = fname.split("/")
            image = bank.ref_frames[gi].convert("RGB")
            image.save(images_dir / f"{camera_id}.png")
            pose = torch.linalg.inv(bank.ref_w2cs[gi].detach().cpu().float())
            cameras["extrinsics"].append({"camera_id": camera_id, "matrix": pose.numpy().tolist()})
            cameras["intrinsics"].append({"camera_id": camera_id, "matrix": bank.ref_Ks[gi].detach().cpu().numpy().tolist()})
            images.append(image)
            poses.append(pose)
            intrs.append(bank.ref_Ks[gi].detach().cpu())
            name_map[fname] = str(index).zfill(4)
        cameras["num_cameras"] = len(images)
        _hy_log("Prepare WorldMirror Batch", "Stage 3/4: writing cameras.json and name_map.json")
        with open(world_mirror_dir / "cameras.json", "w", encoding="utf-8") as handle:
            json.dump(cameras, handle, indent=2)
        with open(world_mirror_dir / "name_map.json", "w", encoding="utf-8") as handle:
            json.dump(name_map, handle, indent=2)
        bank.world_mirror_dir = str(world_mirror_dir)
        bank.name_map = name_map
        image_tensor = _pil_list_to_image_tensor(images)
        camera_poses_raw = torch.stack(poses).float()
        camera_poses = _normalize_c2w_poses_to_first(camera_poses_raw)
        camera_intrinsics = torch.stack(intrs).float()
        batch = {
            "memory_bank": memory_bank,
            "world_mirror_dir": str(world_mirror_dir),
            "name_map": name_map,
            "images": image_tensor,
            "camera_poses": camera_poses,
            "camera_poses_raw_c2w": camera_poses_raw,
            "camera_intrinsics": camera_intrinsics,
        }
        _hy_log("Prepare WorldMirror Batch", f"Stage 4/4 complete: images={len(images)}, world_mirror_dir={world_mirror_dir}")
        return (
            image_tensor,
            camera_poses,
            camera_intrinsics,
            batch,
            _safe_json_dumps(
                {
                    "frames": len(images),
                    "world_mirror_dir": world_mirror_dir,
                    "cameras_json_pose_basis": "official_hyworld2_c2w",
                    "camera_pose_tensor_basis": "official_first_relative_c2w",
                }
            ),
        )


class HYWorld2MemoryAlignment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "worldmirror_batch": ("HYWORLD2_WORLDMIRROR_BATCH",),
                "mode": (["consume_worldmirror_depths", "align_and_export", "bypass"], {"default": "align_and_export"}),
            },
            "optional": {
                "raw_splats": ("VNCCS_SPLAT",),
                "ply_data": ("PLY_DATA",),
                "downsampled_pts": ("INT", {"default": 2_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_MEMORY_BANK", "STRING", "STRING")
    RETURN_NAMES = ("memory_bank", "aligned_ply", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, worldmirror_batch, mode, raw_splats=None, ply_data=None, downsampled_pts=2_000_000, debug_mode=False):
        _hy_log("Memory Alignment", f"Stage 1/4: consuming WorldMirror depths (mode={mode})")
        memory_bank = worldmirror_batch["memory_bank"]
        bank = memory_bank["bank"]
        world_mirror_dir = Path(worldmirror_batch["world_mirror_dir"])
        depth_dir = _ensure_dir(world_mirror_dir / "results" / "depth")
        depths, depth_source = _raw_worldmirror_depths_to_numpy(raw_splats)
        if not depths and mode != "bypass":
            raise ValueError(
                "HYWorld2 Memory Alignment requires connected raw_splats with metric depth: raw_splats.gs_depth or raw_splats.depth."
            )
        _hy_log("Memory Alignment", f"Writing {len(depths)} depth map(s) to {depth_dir} (source={depth_source})")
        for index, depth in enumerate(depths):
            np.save(depth_dir / f"depth_{index:04d}.npy", depth)
        if mode == "align_and_export":
            _hy_log("Memory Alignment", "Stage 2/4: running memory bank alignment")
            _ensure_single_process_dist(bank)
            bank.alignment(debug_mode=bool(debug_mode))
            _hy_log("Memory Alignment", "Stage 3/4: exporting aligned/global point clouds")
            export_dir = Path(bank.root_path) / "render_results" / bank.results_path
            _ensure_dir(export_dir)
            bank.export_pcd(str(export_dir), N_points=int(downsampled_pts))
            aligned = str(export_dir / "aligned_pcd.ply")
            bypass_source_points = 0
        elif mode == "bypass":
            _hy_log("Memory Alignment", "Stage 2/4: bypassing alignment and exporting source point clouds")
            aligned, bypass_source_points = _export_bypass_memory_bank_pcds(bank, ply_data, raw_splats, downsampled_pts)
        else:
            _hy_log("Memory Alignment", "Stage 2/4: consume depths only; alignment/export skipped")
            aligned = ""
            bypass_source_points = 0
        memory_bank["bank"] = bank
        _hy_log("Memory Alignment", f"Stage 4/4 complete: aligned_ply={aligned or '<none>'}")
        return (
            memory_bank,
            aligned,
            _safe_json_dumps(
                {
                    "mode": mode,
                    "depths_written": len(depths),
                    "depth_source": depth_source,
                    "aligned_ply": aligned,
                    "bypass_source_points": bypass_source_points,
                    "alignment_ran": mode == "align_and_export",
                }
            ),
        )


class HYWorld2GSData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workspace": ("HYWORLD2_WORKSPACE",),
                "mode": (["build", "validate", "repair_metadata"], {"default": "build"}),
            },
            "optional": {
                "memory_bank": ("HYWORLD2_MEMORY_BANK",),
                "result_name": ("STRING", {"default": ""}),
                "out_name": ("STRING", {"default": "gs_data"}),
                "save_normal": ("BOOLEAN", {"default": True}),
                "split_sky": ("BOOLEAN", {"default": True}),
                "split_align": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYWORLD2_GS_DATA", "STRING")
    RETURN_NAMES = ("gs_data", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"

    def run(self, workspace, mode, memory_bank=None, result_name="", out_name="gs_data", save_normal=True, split_sky=True, split_align=False):
        _hy_log("GS Data", f"Stage 1/3: preparing GS dataset (mode={mode})")
        scene = Path(workspace["scene_dir"])
        gs_dir = scene / _sanitize_name(out_name, "gs_data")
        _hy_log("GS Data", f"Scene: {scene}")
        _hy_log("GS Data", f"GS data directory: {gs_dir}")
        if mode == "validate":
            _hy_log("GS Data", "Stage 2/3: validating required files")
            required = [gs_dir / "cameras.json", gs_dir / "points.ply", gs_dir / "images"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"HYWorld2 GS data missing required files: {missing}")
            _hy_log("GS Data", "Stage 3/3 complete: dataset is valid")
            return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps({"valid": True, "gs_data_dir": gs_dir}))
        if mode == "repair_metadata":
            _hy_log("GS Data", "Stage 2/3: repairing metadata")
            meta = gs_dir / "meta_info.json"
            if not meta.exists():
                with open(meta, "w", encoding="utf-8") as handle:
                    json.dump({"scene_type": workspace.get("scene_type", "unknown")}, handle, indent=2)
                _hy_log("GS Data", f"Wrote {meta}")
            _hy_log("GS Data", "Stage 3/3 complete: metadata ready")
            return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps({"repaired": True, "gs_data_dir": gs_dir}))
        _ensure_worldgen_path()
        import hyworld2.worldgen.gen_gs_data as gen_gs_data

        if not hasattr(gen_gs_data, "run_gen_gs_data"):
            raise RuntimeError("gen_gs_data.py must expose run_gen_gs_data for native node execution.")
        _hy_log("GS Data", "Stage 2/3: running gen_gs_data")
        _hy_log("GS Data", f"Options: result_name={result_name or workspace.get('result_name', 'worldstereo-memory-dmd')}, save_normal={bool(save_normal)}, split_sky={bool(split_sky)}, split_align={bool(split_align)}")
        result = gen_gs_data.run_gen_gs_data(
            root_path=str(scene),
            out_name=out_name,
            result_name=result_name or workspace.get("result_name", "worldstereo-memory-dmd"),
            save_normal=bool(save_normal),
            split_sky=bool(split_sky),
            split_align=bool(split_align),
            world_size=1,
        )
        gs_dir = Path(result["output_path"])
        _hy_log("GS Data", f"Stage 3/3 complete: output_path={gs_dir}")
        return ({"workspace": workspace, "gs_data_dir": str(gs_dir)}, _safe_json_dumps(result))


class HYWorld2Train3DGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_data": ("HYWORLD2_GS_DATA",),
            },
            "optional": {
                "train_sampling_preset": (["standard", "half_pano_per_epoch", "random_pano_50_per_epoch"], {"default": "standard"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "patch_size": (["Full", "712", "512", "256"], {"default": "Full"}),
                "max_steps": ("INT", {"default": 8000, "min": 1, "max": 100000, "step": 100}),
                "save_steps": ("STRING", {"default": "8000"}),
                "eval_steps": ("STRING", {"default": "8000"}),
                "ply_steps": ("STRING", {"default": "8000"}),
                "downsample_pts_num": ("INT", {"default": 1_000_000, "min": 1, "max": 50_000_000, "step": 100000}),
                "save_ply": ("BOOLEAN", {"default": True}),
                "disable_video": ("BOOLEAN", {"default": True}),
                "disable_viewer": ("BOOLEAN", {"default": True}),
                "depth_loss": ("BOOLEAN", {"default": True}),
                "normal_loss": ("BOOLEAN", {"default": True}),
                "sky_depth_from_pcd": ("BOOLEAN", {"default": True}),
                "use_scale_regularization": ("BOOLEAN", {"default": True}),
                "use_mask_gaussian": ("BOOLEAN", {"default": True}),
                "mask_export_stochastic": ("BOOLEAN", {"default": True}),
                "mask_export_anchor_protection": ("BOOLEAN", {"default": False}),
                "use_anchor_protection": ("BOOLEAN", {"default": True}),
                "do_prune": ("BOOLEAN", {"default": False}),
                "prune_opacity_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "antialiased": ("BOOLEAN", {"default": True}),
                "normalize_world_space": ("BOOLEAN", {"default": True}),
                "export_mesh": ("BOOLEAN", {"default": True}),
                "strategy_refine_start_iter": ("INT", {"default": 150, "min": 0, "max": 100000, "step": 10}),
                "strategy_refine_stop_iter": ("INT", {"default": 3500, "min": 0, "max": 100000, "step": 10}),
                "strategy_refine_every": ("INT", {"default": 100, "min": 1, "max": 100000, "step": 10}),
                "strategy_refine_scale2d_stop_iter": ("INT", {"default": 3500, "min": 0, "max": 100000, "step": 10}),
                "strategy_reset_every": ("INT", {"default": 99990, "min": 1, "max": 1000000, "step": 10}),
                "strategy_grow_grad2d": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "strategy_prune_scale3d": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.01}),
                "convert_ply_to_worldmirror_preview_basis": ("BOOLEAN", {"default": False}),
                # Keep new widgets appended so existing serialized workflows
                # retain the value order of all legacy widgets above.
                "progressive_patch_schedule": ("STRING", {"default": "0:256,1500:384,3500:full"}),
                "lpips_every": ("INT", {"default": 4, "min": 0, "max": 1000, "step": 1}),
                "depth_every": ("INT", {"default": 2, "min": 0, "max": 1000, "step": 1}),
                "normal_every": ("INT", {"default": 4, "min": 0, "max": 1000, "step": 1}),
                "scale_scheduled_losses": ("BOOLEAN", {"default": True}),
                "optimizer_mode": (["visible_adam", "fused_adam", "adam", "sparse_adam"], {"default": "visible_adam"}),
                "progressive_gaussian_budget": ("STRING", {"default": "0:750000,1000:1000000,2000:1400000,3000:2000000"}),
                "spatial_sort_step": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 100}),
                "progress_log_every": ("INT", {"default": 25, "min": 1, "max": 1000, "step": 1}),
                "profile_training": ("BOOLEAN", {"default": True}),
                "profile_every": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 10}),
                "mesh_max_gaussians": ("INT", {"default": 4000000, "min": 100000, "max": 50000000, "step": 100000}),
            },
        }

    RETURN_TYPES = ("STRING", "TENSOR", "TENSOR", "STRING", "STRING")
    RETURN_NAMES = ("ply_path", "camera_poses", "camera_intrinsics", "train_dir", "info")
    FUNCTION = "run"
    CATEGORY = "VNCCS/HYWorld2"
    OUTPUT_NODE = True

    def run(
        self,
        gs_data,
        train_sampling_preset="standard",
        batch_size=1,
        patch_size="Full",
        progressive_patch_schedule="0:256,1500:384,3500:full",
        max_steps=8000,
        save_steps="8000",
        eval_steps="8000",
        ply_steps="8000",
        downsample_pts_num=1_000_000,
        save_ply=True,
        disable_video=True,
        disable_viewer=True,
        depth_loss=True,
        normal_loss=True,
        lpips_every=4,
        depth_every=2,
        normal_every=4,
        scale_scheduled_losses=True,
        sky_depth_from_pcd=True,
        use_scale_regularization=True,
        use_mask_gaussian=True,
        mask_export_stochastic=True,
        mask_export_anchor_protection=False,
        use_anchor_protection=True,
        do_prune=False,
        prune_opacity_threshold=0.01,
        antialiased=True,
        optimizer_mode="visible_adam",
        progressive_gaussian_budget="0:750000,1000:1000000,2000:1400000,3000:2000000",
        spatial_sort_step=0,
        progress_log_every=25,
        profile_training=True,
        profile_every=100,
        mesh_max_gaussians=4000000,
        normalize_world_space=True,
        export_mesh=True,
        strategy_refine_start_iter=150,
        strategy_refine_stop_iter=3500,
        strategy_refine_every=100,
        strategy_refine_scale2d_stop_iter=3500,
        strategy_reset_every=99990,
        strategy_grow_grad2d=0.0001,
        strategy_prune_scale3d=0.1,
        convert_ply_to_worldmirror_preview_basis=False,
    ):
        _hy_log("Train 3DGS", "Stage 1/5: preparing trainer config")
        _ensure_worldgen_path()
        import hyworld2.worldgen.world_gs_trainer as trainer

        data_dir = Path(gs_data["gs_data_dir"])
        out_dir = data_dir.parent / "gs_results"
        _hy_log("Train 3DGS", f"Input data_dir: {data_dir}")
        _hy_log("Train 3DGS", f"Output train_dir: {out_dir}")
        _reset_dir(out_dir, "HYWorld2 train_dir")
        _ensure_scene_type_meta(data_dir)
        progressive_gaussian_budget = str(progressive_gaussian_budget or "").strip()
        effective_refine_stop_iter = int(strategy_refine_stop_iter)
        if progressive_gaussian_budget:
            try:
                last_budget_step = max(
                    int(entry.strip().split(":", 1)[0])
                    for entry in progressive_gaussian_budget.split(",")
                    if entry.strip()
                )
                # Old serialized workflows retain the former default (750).
                # Keep ADC alive long enough to reach the appended progressive
                # budget stages without requiring users to recreate the node.
                effective_refine_stop_iter = max(
                    effective_refine_stop_iter,
                    last_budget_step + max(500, int(strategy_refine_every)),
                )
            except (ValueError, IndexError):
                # The trainer's strict parser will provide the actionable error.
                pass
        strategy = trainer.BudgetedDefaultStrategy(
            verbose=True,
            refine_start_iter=int(strategy_refine_start_iter),
            refine_stop_iter=effective_refine_stop_iter,
            refine_every=int(strategy_refine_every),
            refine_scale2d_stop_iter=int(strategy_refine_scale2d_stop_iter),
            reset_every=int(strategy_reset_every),
            grow_grad2d=float(strategy_grow_grad2d),
            prune_scale3d=float(strategy_prune_scale3d),
        )
        cfg = trainer.Config(strategy=strategy)
        cfg.data_dir = str(data_dir)
        cfg.result_dir = str(out_dir)
        cfg.batch_size = int(batch_size)
        cfg.patch_size = None if str(patch_size) == "Full" else int(patch_size)
        cfg.progressive_patch_schedule = str(progressive_patch_schedule or "").strip()
        if hasattr(cfg, "train_sampling_preset"):
            cfg.train_sampling_preset = str(train_sampling_preset)
        cfg.max_steps = int(max_steps)
        cfg.save_steps = _parse_int_list(save_steps)
        cfg.eval_steps = _parse_int_list(eval_steps)
        cfg.ply_steps = _parse_int_list(ply_steps)
        cfg.downsample_pts_num = int(downsample_pts_num)
        cfg.save_ply = bool(save_ply)
        cfg.disable_video = bool(disable_video)
        cfg.disable_viewer = bool(disable_viewer)
        if hasattr(cfg, "dataloader_num_workers"):
            cfg.dataloader_num_workers = 0
        depth_files_valid = _has_valid_depth_files(data_dir)
        normal_files_valid = _has_valid_normal_files(data_dir)
        cfg.depth_loss = bool(depth_loss and depth_files_valid)
        cfg.normal_loss = bool(normal_loss and normal_files_valid)
        cfg.lpips_every = max(0, int(lpips_every))
        cfg.depth_every = max(0, int(depth_every))
        cfg.normal_every = max(0, int(normal_every))
        cfg.scale_scheduled_losses = bool(scale_scheduled_losses)
        cfg.sky_depth_from_pcd = bool(sky_depth_from_pcd and cfg.depth_loss and normal_files_valid)
        cfg.use_scale_regularization = bool(use_scale_regularization)
        cfg.use_mask_gaussian = bool(use_mask_gaussian)
        if hasattr(cfg, "mask_export_stochastic"):
            cfg.mask_export_stochastic = bool(mask_export_stochastic)
        if hasattr(cfg, "mask_export_anchor_protection"):
            cfg.mask_export_anchor_protection = bool(mask_export_anchor_protection)
        if hasattr(cfg, "use_anchor_protection"):
            cfg.use_anchor_protection = bool(use_anchor_protection)
        cfg.do_prune = bool(do_prune)
        cfg.prune_opacity_threshold = float(prune_opacity_threshold)
        cfg.antialiased = bool(antialiased)
        optimizer_mode = str(optimizer_mode)
        cfg.visible_adam = optimizer_mode == "visible_adam"
        cfg.fused_adam = optimizer_mode == "fused_adam"
        cfg.sparse_grad = optimizer_mode == "sparse_adam"
        cfg.packed = cfg.sparse_grad
        cfg.progressive_gaussian_budget = progressive_gaussian_budget
        cfg.spatial_sort_step = max(0, int(spatial_sort_step))
        cfg.progress_log_every = max(1, int(progress_log_every))
        cfg.profile_training = bool(profile_training)
        cfg.profile_every = max(1, int(profile_every))
        cfg.mesh_max_gaussians = max(100000, int(mesh_max_gaussians))
        cfg.no_normalize = not bool(normalize_world_space)
        if hasattr(cfg, "export_mesh"):
            cfg.export_mesh = bool(export_mesh)
        _hy_log(
            "Train 3DGS",
            "Config: "
            f"batch_size={cfg.batch_size}, patch_size={cfg.patch_size or 'Full'}, "
            f"train_sampling_preset={getattr(cfg, 'train_sampling_preset', 'standard')}, "
            f"max_steps={cfg.max_steps}, downsample_pts_num={cfg.downsample_pts_num}, save_ply={cfg.save_ply}, "
            f"depth_loss={cfg.depth_loss}, normal_loss={cfg.normal_loss}, sky_depth_from_pcd={cfg.sky_depth_from_pcd}, "
            f"loss_every={cfg.lpips_every}/{cfg.depth_every}/{cfg.normal_every}, optimizer_mode={optimizer_mode}, "
            f"progressive_patch={cfg.progressive_patch_schedule or 'off'}, progressive_budget={cfg.progressive_gaussian_budget or 'off'}, "
            f"use_scale_regularization={cfg.use_scale_regularization}, use_mask_gaussian={cfg.use_mask_gaussian}, "
            f"use_anchor_protection={getattr(cfg, 'use_anchor_protection', False)}, "
            f"antialiased={cfg.antialiased}, normalize_world_space={bool(normalize_world_space)}"
        )
        command_info = {
            "data_dir": str(data_dir),
            "result_dir": str(out_dir),
            "batch_size": int(cfg.batch_size),
            "patch_size": cfg.patch_size,
            "progressive_patch_schedule": cfg.progressive_patch_schedule,
            "train_sampling_preset": str(getattr(cfg, "train_sampling_preset", "standard")),
            "max_steps": int(max_steps),
            "save_steps": cfg.save_steps,
            "eval_steps": cfg.eval_steps,
            "ply_steps": cfg.ply_steps,
            "downsample_pts_num": int(cfg.downsample_pts_num),
            "save_ply": bool(cfg.save_ply),
            "disable_video": bool(cfg.disable_video),
            "disable_viewer": bool(cfg.disable_viewer),
            "dataloader_num_workers": int(getattr(cfg, "dataloader_num_workers", -1)),
            "depth_loss_requested": bool(depth_loss),
            "depth_loss_enabled": bool(cfg.depth_loss),
            "normal_loss_requested": bool(normal_loss),
            "normal_loss_enabled": bool(cfg.normal_loss),
            "lpips_every": int(cfg.lpips_every),
            "depth_every": int(cfg.depth_every),
            "normal_every": int(cfg.normal_every),
            "scale_scheduled_losses": bool(cfg.scale_scheduled_losses),
            "sky_depth_from_pcd_requested": bool(sky_depth_from_pcd),
            "sky_depth_from_pcd_enabled": bool(cfg.sky_depth_from_pcd),
            "use_scale_regularization": bool(cfg.use_scale_regularization),
            "use_mask_gaussian": bool(cfg.use_mask_gaussian),
            "mask_export_stochastic": bool(getattr(cfg, "mask_export_stochastic", False)),
            "mask_export_anchor_protection": bool(getattr(cfg, "mask_export_anchor_protection", False)),
            "use_anchor_protection": bool(getattr(cfg, "use_anchor_protection", False)),
            "do_prune": bool(cfg.do_prune),
            "prune_opacity_threshold": float(cfg.prune_opacity_threshold),
            "antialiased": bool(cfg.antialiased),
            "optimizer_mode": optimizer_mode,
            "visible_adam": bool(cfg.visible_adam),
            "fused_adam": bool(cfg.fused_adam),
            "packed": bool(cfg.packed),
            "sparse_grad": bool(cfg.sparse_grad),
            "progressive_gaussian_budget": cfg.progressive_gaussian_budget,
            "spatial_sort_step": int(cfg.spatial_sort_step),
            "progress_log_every": int(cfg.progress_log_every),
            "profile_training": bool(cfg.profile_training),
            "profile_every": int(cfg.profile_every),
            "mesh_max_gaussians": int(cfg.mesh_max_gaussians),
            "normalize_world_space": bool(normalize_world_space),
            "export_mesh": bool(getattr(cfg, "export_mesh", False)),
            "strategy": {
                "refine_start_iter": int(strategy.refine_start_iter),
                "refine_stop_iter": int(strategy.refine_stop_iter),
                "refine_every": int(strategy.refine_every),
                "refine_scale2d_stop_iter": int(strategy.refine_scale2d_stop_iter),
                "reset_every": int(strategy.reset_every),
                "grow_grad2d": float(strategy.grow_grad2d),
                "prune_scale3d": float(strategy.prune_scale3d),
                "prune_opa": float(strategy.prune_opa),
                "grow_scale3d": float(strategy.grow_scale3d),
                "grow_scale2d": float(strategy.grow_scale2d),
                "prune_scale2d": float(strategy.prune_scale2d),
            },
            "official_hyworld2_stage5_profile": True,
            "convert_ply_to_worldmirror_preview_basis": bool(convert_ply_to_worldmirror_preview_basis),
            # Background Preview uses these fields to decide whether a basis
            # conversion is still required.  Without them, an already-converted
            # point_cloud_*.ply was detected by filename as HYWorld2 and rotated
            # a second time.
            "ply_basis": "worldmirror" if bool(convert_ply_to_worldmirror_preview_basis) else "hyworld2_worldgen",
            "camera_pose_basis": "worldmirror_c2w" if bool(convert_ply_to_worldmirror_preview_basis) else "hyworld2_worldgen_c2w",
            "lpips_net": cfg.lpips_net,
            "in_process": True,
        }
        with open(out_dir / "train_command.json", "w", encoding="utf-8") as handle:
            json.dump(command_info, handle, indent=2)
        _hy_log("Train 3DGS", f"Stage 2/5: wrote train command metadata: {out_dir / 'train_command.json'}")
        if depth_loss and not cfg.depth_loss:
            print(f"[HYWorld2 Train 3DGS] depth_loss requested but valid metric float16-packed depths are missing under {data_dir / 'depths'}; disabling depth_loss.")
        if normal_loss and not cfg.normal_loss:
            print(f"[HYWorld2 Train 3DGS] normal_loss requested but normals are missing/constant under {data_dir / 'normals'}; disabling normal_loss.")
        if sky_depth_from_pcd and not cfg.sky_depth_from_pcd:
            print("[HYWorld2 Train 3DGS] sky_depth_from_pcd requested but depth/normal inputs are not usable; disabling sky_depth_from_pcd.")
        _hy_log("Train 3DGS", "Stage 3/5: running 3DGS trainer")
        with torch.inference_mode(False), torch.enable_grad():
            trainer.main(0, 0, 1, cfg)
        _hy_log("Train 3DGS", "Stage 4/5: locating and converting latest PLY")
        ply_path = _find_latest_ply(out_dir)
        if ply_path:
            _hy_log("Train 3DGS", f"Latest PLY: {ply_path}")
            if bool(convert_ply_to_worldmirror_preview_basis):
                _hy_log("Train 3DGS", "Converting PLY to WorldMirror preview basis (non-official compatibility path)")
                ply_path = _convert_trainer_gaussian_ply_to_worldmirror_basis(ply_path)
            _hy_log("Train 3DGS", f"PLY ready: {ply_path}")
        else:
            _hy_log("Train 3DGS", "No PLY file found after training")
        camera_json = out_dir / "ply" / "trainer_cameras.json"
        if not camera_json.exists():
            candidates = sorted((out_dir / "ply").glob("trainer_cameras_*.json")) if (out_dir / "ply").exists() else []
            camera_json = candidates[-1] if candidates else data_dir / "cameras.json"
        poses, intrs = _load_camera_tensors_from_json(camera_json) if camera_json.exists() else (torch.empty((0, 4, 4)), torch.empty((0, 3, 3)))
        if poses.numel() > 0 and bool(convert_ply_to_worldmirror_preview_basis):
            poses = torch.stack([_worldstereo_c2w_to_worldmirror_c2w(pose) for pose in poses]).float()
        _hy_log("Train 3DGS", f"Stage 5/5 complete: cameras={int(poses.shape[0]) if poses.ndim >= 1 else 0}, camera_json={camera_json}")
        output_basis = "worldmirror" if bool(convert_ply_to_worldmirror_preview_basis) else "hyworld2_worldgen"
        info = {
            "ply_path": ply_path,
            "train_dir": str(out_dir),
            "camera_json": str(camera_json) if camera_json.exists() else "",
            "camera_pose_basis": "worldmirror_c2w" if bool(convert_ply_to_worldmirror_preview_basis) else "hyworld2_worldgen_c2w",
            "ply_basis": output_basis,
            "official_hyworld2_stage5_profile": True,
            "convert_ply_to_worldmirror_preview_basis": bool(convert_ply_to_worldmirror_preview_basis),
        }
        return (ply_path, poses, intrs, str(out_dir), _safe_json_dumps(info))


NODE_CLASS_MAPPINGS = {
    "HYWorld2Workspace": HYWorld2Workspace,
    "HYWorld2QwenVL": HYWorld2QwenVL,
    "HYWorld2Trajectories": HYWorld2Trajectories,
    "HYWorld2TrajectoriesTest": HYWorld2TrajectoriesTest,
    "HYWorld2MemoryBank": HYWorld2MemoryBank,
    "HYWorld2WorldExpansion": HYWorld2WorldExpansion,
    "HYWorld2KleinWorldExpansion": HYWorld2KleinWorldExpansion,
    "HYWorld2SaveExpansionDataset": HYWorld2SaveExpansionDataset,
    "HYWorld2PrepareWorldMirrorBatch": HYWorld2PrepareWorldMirrorBatch,
    "HYWorld2MemoryAlignment": HYWorld2MemoryAlignment,
    "HYWorld2GSData": HYWorld2GSData,
    "HYWorld2Train3DGS": HYWorld2Train3DGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld2Workspace": "HYWorld2 Workspace",
    "HYWorld2QwenVL": "HYWorld2 QwenVL",
    "HYWorld2Trajectories": "HYWorld2 Trajectories",
    "HYWorld2TrajectoriesTest": "HYWorld2 Trajectories (LLM-free test)",
    "HYWorld2MemoryBank": "HYWorld2 Memory Bank",
    "HYWorld2WorldExpansion": "HYWorld2 World Expansion",
    "HYWorld2KleinWorldExpansion": "HYWorld2 Klein World Expansion (experimental)",
    "HYWorld2SaveExpansionDataset": "HYWorld2 Save Expansion Dataset",
    "HYWorld2PrepareWorldMirrorBatch": "HYWorld2 Prepare WorldMirror Batch",
    "HYWorld2MemoryAlignment": "HYWorld2 Memory Alignment",
    "HYWorld2GSData": "HYWorld2 GS Data",
    "HYWorld2Train3DGS": "HYWorld2 Train 3DGS",
}
