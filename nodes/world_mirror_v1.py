"""
VNCCS 3D Background Generation Nodes

Uses WorldMirror for 3D reconstruction from images.
Includes equirectangular panorama to perspective views conversion.
"""

import os
import sys
import math
import re
import json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import folder_paths


# nodes/ -> repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# worldmirror/ lives directly in repo root
WORLDMIRROR_DIR = os.path.join(PROJECT_ROOT, "worldmirror")
if WORLDMIRROR_DIR not in sys.path:
    sys.path.insert(0, WORLDMIRROR_DIR)

# Import FastPLYRenderer after setting up sys.path
# Import FastPLYRenderer after setting up sys.path
try:
    from src.utils.fast_ply_render import FastPLYRenderer
    # Import ported utils for advanced filtering
    from src.utils.visual_util import segment_sky, download_file_from_url
    from src.utils.geometry import depth_edge, normals_edge
except ImportError:
    # If using direct import (e.g. dev environment)
    from background_data.worldmirror.src.utils.fast_ply_render import FastPLYRenderer
    from background_data.worldmirror.src.utils.visual_util import segment_sky, download_file_from_url
    from background_data.worldmirror.src.utils.geometry import depth_edge, normals_edge

try:
    import onnxruntime
    SKYSEG_AVAILABLE = True
except ImportError:
    SKYSEG_AVAILABLE = False
    print("⚠️ [VNCCS] onnxruntime not found. Sky segmentation will be disabled.")


# ----------------------------------------------------------------------------
# GSPLAT DIAGNOSTICS
# ----------------------------------------------------------------------------
try:
    import gsplat
    print(f"✅ [VNCCS] gsplat library detected: Version {gsplat.__version__}")
    GSPLAT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [VNCCS] gsplat library NOT found: {e}")
    GSPLAT_AVAILABLE = False
except Exception as e:
    print(f"❌ [VNCCS] Error loading gsplat: {e}")
    GSPLAT_AVAILABLE = False

if torch.cuda.is_available():
    print(f"✅ [VNCCS] CUDA is available: {torch.version.cuda}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ [VNCCS] CUDA is NOT available. gsplat requires CUDA.")
# ----------------------------------------------------------------------------


# ============================================================================
# Utility Functions
# ============================================================================

def build_rotation_matrix(pitch_deg, yaw_deg, roll_deg=0.0):
    """
    Build a 3x3 rotation matrix from pitch, yaw, and roll (in degrees).
    Order of application: Pitch (X) -> Yaw (Y) -> Roll (Z)
    Matches equirect_to_perspective logic.
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)
    
    # Rx (Pitch)
    cp, sp = math.cos(p), math.sin(p)
    Rx = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ], dtype=np.float32)
    
    # Ry (Yaw)
    cy, sy = math.cos(y), math.sin(y)
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=np.float32)
    
    # Rz (Roll)
    R = Ry @ Rx # Order: v' = Ry @ Rx @ v
    if roll_deg != 0:
        cr, sr = math.cos(r), math.sin(r)
        Rz = np.array([
            [cr, -sr, 0],
            [sr, cr, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        R = Rz @ R
        
    return R


def build_camera_pose_matrix(pitch_deg, yaw_deg, roll_deg=0.0, pose_convention="opencv_c2w"):
    """Build a 4x4 camera pose matrix for exported camera priors."""
    if pose_convention == "flip_pitch":
        R = build_rotation_matrix(-pitch_deg, yaw_deg, roll_deg)
    elif pose_convention == "flip_yaw":
        R = build_rotation_matrix(pitch_deg, -yaw_deg, roll_deg)
    elif pose_convention == "flip_pitch_yaw":
        R = build_rotation_matrix(-pitch_deg, -yaw_deg, roll_deg)
    else:
        R = build_rotation_matrix(pitch_deg, yaw_deg, roll_deg)

    axis_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    if pose_convention == "opencv_w2c":
        R = R.T
    elif pose_convention == "opengl_c2w":
        R = R @ axis_flip
    elif pose_convention == "opengl_w2c":
        R = (R @ axis_flip).T

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    return pose


def _load_preview_camera_tensors_from_json(camera_json):
    camera_json = Path(camera_json)
    if not camera_json.exists():
        return torch.empty((0, 4, 4)), torch.empty((0, 3, 3))
    with open(camera_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    poses = []
    intrs = []
    if isinstance(data, dict) and "cameras" in data:
        cameras = data.get("cameras") or {}
        order = data.get("camera_order") or sorted(cameras.keys(), key=lambda value: str(value))
        for camera_id in order:
            entry = cameras.get(str(camera_id), cameras.get(camera_id))
            if not isinstance(entry, dict):
                continue
            if "camera_pose" in entry:
                pose = np.asarray(entry["camera_pose"], dtype=np.float32)
            elif "extrinsic" in entry:
                pose = np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32))
            else:
                continue
            if "intrinsic" not in entry:
                continue
            poses.append(pose)
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))
    elif isinstance(data, dict):
        for camera_id in sorted(data.keys(), key=lambda value: str(value)):
            entry = data[camera_id]
            if not isinstance(entry, dict) or "extrinsic" not in entry or "intrinsic" not in entry:
                continue
            poses.append(np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32)))
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))

    if not poses:
        return torch.empty((0, 4, 4)), torch.empty((0, 3, 3))
    return torch.from_numpy(np.stack(poses)).float(), torch.from_numpy(np.stack(intrs)).float()


def _find_preview_camera_json_for_ply(ply_path):
    ply_path = Path(ply_path)
    candidates = []
    if re.fullmatch(r"point_cloud_(\d+)\.ply", ply_path.name):
        step = re.fullmatch(r"point_cloud_(\d+)\.ply", ply_path.name).group(1)
        candidates.append(ply_path.with_name(f"trainer_cameras_{step}.json"))
    candidates.extend([
        ply_path.with_name("trainer_cameras.json"),
        ply_path.parent / "cameras.json",
        ply_path.parent.parent / "gs_data" / "cameras.json",
        ply_path.parent.parent / "cameras.json",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    json_candidates = sorted(ply_path.parent.glob("trainer_cameras_*.json"))
    return json_candidates[-1] if json_candidates else None


def _read_ply_header_bytes(path, max_bytes=262144):
    with open(path, "rb") as handle:
        data = handle.read(max_bytes)
    marker = b"end_header\n"
    index = data.find(marker)
    if index < 0:
        marker = b"end_header\r\n"
        index = data.find(marker)
    if index < 0:
        return ""
    return data[:index + len(marker)].decode("ascii", errors="replace")


def _ply_header_has_gaussian_fields(path):
    header = _read_ply_header_bytes(path)
    if not header:
        return False
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    names = set()
    for line in header.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "property":
            names.add(parts[2])
    return required.issubset(names)


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


def _preview_worldmirror_basis_cache_path(ply_path):
    path = Path(ply_path)
    stat = path.stat()
    key = f"basis_v3_opencv_y_down:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).hexdigest()[:16]
    cache_dir = Path(folder_paths.get_temp_directory()) / "hyworld2_preview_basis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{path.stem}.worldmirror_{digest}.ply"


def _convert_gaussian_ply_to_worldmirror_preview_basis(ply_path):
    path = Path(ply_path)
    if not path.exists() or not _ply_header_has_gaussian_fields(path):
        return str(path)

    out_path = _preview_worldmirror_basis_cache_path(path)
    if out_path.exists() and out_path.stat().st_mtime >= path.stat().st_mtime:
        return str(out_path)

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
                if len(parts) >= 3 and parts[1] != "list":
                    props.append((parts[1], parts[2]))
            elif text == "end_header":
                data_offset = handle.tell()
                break

    if vertex_count is None:
        raise ValueError(f"PLY has no vertex count: {path}")

    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "short": "<i2",
        "ushort": "<u2",
    }
    dtype = np.dtype([(name, type_map.get(kind, "<f4")) for kind, name in props])
    vertices = np.fromfile(path, dtype=dtype, count=vertex_count, offset=data_offset).copy()
    prop_names = set(vertices.dtype.names or [])
    if not {"x", "y", "z"}.issubset(prop_names):
        return str(path)

    old_x = vertices["x"].copy()
    old_y = vertices["y"].copy()
    old_z = vertices["z"].copy()
    vertices["x"] = old_x
    vertices["y"] = -old_z
    vertices["z"] = old_y

    if {"nx", "ny", "nz"}.issubset(prop_names):
        old_nx = vertices["nx"].copy()
        old_ny = vertices["ny"].copy()
        old_nz = vertices["nz"].copy()
        vertices["nx"] = old_nx
        vertices["ny"] = -old_nz
        vertices["nz"] = old_ny

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

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    os.replace(tmp_path, out_path)
    print(f"[VNCCS_BackgroundPreview] Converted HYWorld2 PLY to preview basis: {out_path}")
    return str(out_path)


def _preview_stat_line(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "empty"
    return (
        f"min={float(arr.min()):.6g}, "
        f"max={float(arr.max()):.6g}, "
        f"mean={float(arr.mean()):.6g}"
    )


def _read_preview_ply_sample(path, sample_count=200000):
    import struct

    path = Path(path)
    with open(path, "rb") as handle:
        header_bytes = bytearray()
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
        data_offset = len(header_bytes)
        header = header_bytes.decode("ascii", errors="replace")

        fmt = None
        vertex_count = 0
        in_vertex = False
        props = []
        list_props = []
        for raw_line in header.splitlines():
            parts = raw_line.strip().split()
            if not parts:
                continue
            if parts[:2] == ["format", "binary_little_endian"]:
                fmt = "binary_little_endian"
            elif parts[:2] == ["format", "ascii"]:
                fmt = "ascii"
            elif len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and parts[0] == "property":
                if len(parts) >= 5 and parts[1] == "list":
                    list_props.append(" ".join(parts[1:]))
                elif len(parts) >= 3:
                    props.append((parts[1], parts[2]))

        type_info = {
            "char": ("b", 1), "int8": ("b", 1),
            "uchar": ("B", 1), "uint8": ("B", 1),
            "short": ("h", 2), "int16": ("h", 2),
            "ushort": ("H", 2), "uint16": ("H", 2),
            "int": ("i", 4), "int32": ("i", 4),
            "uint": ("I", 4), "uint32": ("I", 4),
            "float": ("f", 4), "float32": ("f", 4),
            "double": ("d", 8), "float64": ("d", 8),
        }
        offsets = {}
        stride = 0
        for prop_type, name in props:
            if prop_type not in type_info:
                continue
            offsets[name] = (prop_type, stride)
            stride += type_info[prop_type][1]
        if fmt != "binary_little_endian" or stride <= 0 or vertex_count <= 0:
            return {
                "format": fmt or "unknown",
                "vertex_count": vertex_count,
                "properties": [name for _, name in props],
                "list_properties": list_props,
                "unsupported": True,
            }

        read_count = min(int(sample_count), int(vertex_count))
        payload = handle.read(read_count * stride)

    values = {name: [] for _, name in props}
    for index in range(read_count):
        base = index * stride
        for name, (prop_type, offset) in offsets.items():
            code, size = type_info[prop_type]
            if base + offset + size <= len(payload):
                values[name].append(struct.unpack_from("<" + code, payload, base + offset)[0])

    summary = {
        "format": fmt,
        "vertex_count": vertex_count,
        "sample_count": read_count,
        "stride": stride,
        "properties": [name for _, name in props],
        "list_properties": list_props,
        "values": values,
        "data_offset": data_offset,
    }
    return summary


def _log_preview_ply_diagnostics(path, label):
    try:
        stats = _read_preview_ply_sample(path)
        print(f"[VNCCS_BackgroundPreview][diag] {label} PLY: {path}")
        print(
            f"[VNCCS_BackgroundPreview][diag] {label} format={stats.get('format')}, "
            f"vertices={stats.get('vertex_count')}, sample={stats.get('sample_count')}, "
            f"stride={stats.get('stride')}, properties={stats.get('properties')}"
        )
        if stats.get("list_properties"):
            print(f"[VNCCS_BackgroundPreview][diag] {label} list_properties={stats['list_properties']}")
        values = stats.get("values") or {}
        for group_name, names in (
            ("xyz", ("x", "y", "z")),
            ("f_dc", ("f_dc_0", "f_dc_1", "f_dc_2")),
            ("opacity_raw", ("opacity",)),
            ("scale_raw", ("scale_0", "scale_1", "scale_2")),
            ("rot", ("rot_0", "rot_1", "rot_2", "rot_3")),
        ):
            present = [name for name in names if name in values and values[name]]
            if not present:
                continue
            if len(present) == 1:
                print(f"[VNCCS_BackgroundPreview][diag] {label} {group_name}: {_preview_stat_line(values[present[0]])}")
            else:
                joined = "; ".join(f"{name}: {_preview_stat_line(values[name])}" for name in present)
                print(f"[VNCCS_BackgroundPreview][diag] {label} {group_name}: {joined}")
        if all(name in values for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
            sh_c0 = 0.28209479177387814
            rgb = np.stack([0.5 + sh_c0 * np.asarray(values[f"f_dc_{idx}"], dtype=np.float64) for idx in range(3)], axis=1)
            print(
                f"[VNCCS_BackgroundPreview][diag] {label} rgb_from_sh: "
                f"r({_preview_stat_line(rgb[:, 0])}); "
                f"g({_preview_stat_line(rgb[:, 1])}); "
                f"b({_preview_stat_line(rgb[:, 2])})"
            )
        if "opacity" in values and values["opacity"]:
            alpha = 1.0 / (1.0 + np.exp(-np.asarray(values["opacity"], dtype=np.float64)))
            print(f"[VNCCS_BackgroundPreview][diag] {label} alpha_sigmoid: {_preview_stat_line(alpha)}")
    except Exception as exc:
        print(f"[VNCCS_BackgroundPreview][diag] {label} PLY diagnostics failed: {type(exc).__name__}: {exc}")


def _log_preview_splat_diagnostics(path, label, sample_count=200000):
    import struct

    try:
        path = Path(path)
        if not path.exists() or path.suffix.lower() != ".splat":
            return
        size = path.stat().st_size
        count = size // 32
        read_count = min(int(sample_count), int(count))
        xyz = [[], [], []]
        scales = [[], [], []]
        alpha = []
        with open(path, "rb") as handle:
            for _ in range(read_count):
                row = handle.read(32)
                if len(row) < 32:
                    break
                vals = struct.unpack_from("<6f", row, 0)
                for idx in range(3):
                    xyz[idx].append(vals[idx])
                    scales[idx].append(vals[idx + 3])
                alpha.append(row[27] / 255.0)
        print(f"[VNCCS_BackgroundPreview][diag] {label} SPLAT: {path}")
        print(f"[VNCCS_BackgroundPreview][diag] {label} splats={count}, sample={read_count}, bytes={size}")
        print(
            f"[VNCCS_BackgroundPreview][diag] {label} splat xyz: "
            f"x({_preview_stat_line(xyz[0])}); y({_preview_stat_line(xyz[1])}); z({_preview_stat_line(xyz[2])})"
        )
        print(
            f"[VNCCS_BackgroundPreview][diag] {label} splat scale: "
            f"x({_preview_stat_line(scales[0])}); y({_preview_stat_line(scales[1])}); z({_preview_stat_line(scales[2])})"
        )
        print(f"[VNCCS_BackgroundPreview][diag] {label} splat alpha: {_preview_stat_line(alpha)}")
    except Exception as exc:
        print(f"[VNCCS_BackgroundPreview][diag] {label} SPLAT diagnostics failed: {type(exc).__name__}: {exc}")


def _camera_debug_angles_from_pose(pose):
    pose = np.asarray(pose, dtype=np.float32)
    forward = pose[:3, 2]
    norm = max(float(np.linalg.norm(forward)), 1e-8)
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(forward[1]) / norm))))
    up = pose[:3, 1]
    roll = math.degrees(math.atan2(float(up[0]), max(float(abs(up[1])), 1e-8)))
    return yaw, pitch, roll


def _fmt_vec3_np(vec):
    return f"({float(vec[0]): .4f},{float(vec[1]): .4f},{float(vec[2]): .4f})"


def equirect_to_perspective(
    equirect_img, 
    fov_deg, 
    yaw_deg, 
    pitch_deg, 
    roll_deg=0.0,
    output_size=(512, 512), 
    dynamic_fov=False,
    sampling_mode="bilinear",
):
    """
    Extract a perspective view from an equirectangular panorama using PyTorch grid_sample.
    Supports Dynamic FOV reduction for off-horizon views.
    """
    # 1. Dynamic FOV Reduction (Strategies 1)
    if dynamic_fov and abs(pitch_deg) > 15:
        # Reduce FOV as we look up/down to strictly avoid "polar stretching"
        # At 90 degree pitch, FOV becomes limited to avoid infinity stretch
        scale = math.cos(math.radians(pitch_deg))
        # Clamp scale to reasonable minimum (e.g. 0.5) to don't vanish
        scale = max(0.5, scale * 0.8 + 0.2) 
        original_fov = fov_deg
        fov_deg = fov_deg * scale
        # print(f"🔭 Dynamic FOV: {original_fov}° -> {fov_deg:.1f}° (Pitch {pitch_deg}°)")

    # Convert image to tensor [1, C, H, W]
    if isinstance(equirect_img, Image.Image):
        img_np = np.array(equirect_img.convert("RGB"))
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    else:
        img_np = np.array(equirect_img)
        # Robustly detect if image is already 0-1 or 0-255
        is_normalized = img_np.dtype == np.float32 or img_np.dtype == np.float64
        if is_normalized and img_np.max() <= 1.01:
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0)
        else:
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    B, C, H, W = img_tensor.shape
    out_w, out_h = output_size
    
    fov_rad = math.radians(fov_deg)
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)
    
    # 1.1 Focal length and Aspect Ratio
    f = 1.0 / math.tan(fov_rad / 2)
    aspect = out_w / out_h
    
    # Use torch meshgrid
    device = torch.device('cpu') 
    
    # Horizontal grid scales with aspect ratio to avoid squashing
    xv = torch.linspace(-aspect, aspect, out_w, device=device)
    yv = torch.linspace(-1, 1, out_h, device=device)
    
    grid_y, grid_x = torch.meshgrid(yv, xv, indexing='ij')
    
    # 2. Ray Construction (OpenCV Convention: X-Right, Y-Down, Z-Forward)
    # Rectilinear: (x, y, f) where x in [-aspect, aspect] and y in [-1, 1]
    rays_x = grid_x
    rays_y = grid_y # Y-down (grid_y -1 is Top, 1 is Bottom)
    rays_z = torch.full_like(grid_x, f)
    
    # Normalize rays
    rays_norm = torch.sqrt(rays_x**2 + rays_y**2 + rays_z**2)
    rays_x = rays_x / rays_norm
    rays_y = rays_y / rays_norm
    rays_z = rays_z / rays_norm
    
    # Apply Rotations
    # Pitch (X axis)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    rays_y_p = rays_y * cos_p - rays_z * sin_p
    rays_z_p = rays_y * sin_p + rays_z * cos_p
    rays_x_p = rays_x 
    
    # Yaw (Y axis)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    rays_x_r = rays_x_p * cos_y + rays_z_p * sin_y
    rays_z_r = -rays_x_p * sin_y + rays_z_p * cos_y
    rays_y_r = rays_y_p 
    
    # Roll (Z axis)
    if roll_rad != 0:
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
        rx, ry = rays_x_r, rays_y_r
        rays_x_r = rx * cos_r - ry * sin_r
        rays_y_r = rx * sin_r + ry * cos_r
    
    # XYZ -> Spherical (theta, phi)
    # Theta: Atan2(X, Z) in OpenCV world
    theta = torch.atan2(rays_x_r, rays_z_r)
    # Phi: Asin(-Y / Norm) because standard equirect has -Up to +Down? 
    # Actually most panoramas have +Up (phi > 0) towards the top.
    # In OpenCV, -Y is Up.
    phi = torch.asin(torch.clamp(-rays_y_r, -1.0, 1.0))
    
    # Spherical -> UV Grid [-1, 1]
    u = theta / math.pi 
    v = -2.0 * phi / math.pi 
    
    grid = torch.stack((u, v), dim=-1).unsqueeze(0) # [1, H, W, 2]
    
    if sampling_mode not in ("bilinear", "bicubic"):
        sampling_mode = "bilinear"

    # Sampling
    out_tensor = torch.nn.functional.grid_sample(
        img_tensor, 
        grid, 
        mode=sampling_mode,
        padding_mode='border', 
        align_corners=True
    )
    
    out_img = Image.fromarray((out_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    
    return out_img, f


def create_filter_mask(
    pts3d_conf: np.ndarray,
    depth_preds: np.ndarray, 
    normal_preds: np.ndarray,
    sky_mask: np.ndarray,
    distortion_mask: np.ndarray = None,
    confidence_percentile: float = 10.0,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = True,
    apply_edge_mask: bool = True,
    apply_sky_mask: bool = False,
) -> np.ndarray:
    """
    Create comprehensive filter mask based on confidence, edges, sky segmentation, and distortion.
    """
    S, H, W = pts3d_conf.shape[:3]
    final_mask_list = []
    
    # Precompute global confidence threshold if needed
    conf_thresh = 0.0
    if apply_confidence_mask:
        conf_thresh = np.percentile(pts3d_conf, confidence_percentile)
        
    for i in range(S):
        # Start with all valid
        mask = np.ones((H, W), dtype=bool)
        
        # 0. Distortion Mask (Priority Filter)
        if distortion_mask is not None:
             # Mask < 0.2 considered invalid (edges/distortion)
             d_mask = distortion_mask[i].squeeze()
             if d_mask.shape != (H, W):
                 d_mask = cv2.resize(d_mask, (W, H))
             mask &= (d_mask > 0.5)

        # 1. Confidence Mask
        if apply_confidence_mask:
            mask &= (pts3d_conf[i] > conf_thresh)
            
        # 2. Sky Mask
        if apply_sky_mask and sky_mask is not None:
             # sky_mask is True for NON-sky (it's a keep mask)
             mask &= sky_mask[i]
             
        # 3. Edge Mask (Depth discontinuities)
        if apply_edge_mask:
            depth_i = depth_preds[i]
            # Ensure 2D for gradient calculation
            if depth_i.ndim == 3 and depth_i.shape[-1] == 1:
                depth_i = depth_i.squeeze(-1)
            
            # Simple gradient-based edge detection
            # We approximate edge detection if helper not available, or assume normals_edge was imported but maybe not working fully in this context?
            # Let's use the robust gradient magnitude on depth
            gy, gx = np.gradient(depth_i)
            grad_mag = np.sqrt(gx**2 + gy**2)
            edge_mask = grad_mag < edge_depth_threshold
            mask &= edge_mask
            
        final_mask_list.append(mask)

    return np.stack(final_mask_list, axis=0)



# ============================================================================
# ComfyUI Nodes
# ============================================================================

class VNCCS_LoadWorldMirrorModel:
    """Load WorldMirror model for 3D reconstruction."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cpu"}),
                "sampling_strategy": (["conservative", "uniform"], {"default": "uniform"}),
                "enable_conf_filter": ("BOOLEAN", {"default": False}), 
                "conf_threshold_percent": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0}),
            }
        }
    
    RETURN_TYPES = ("WORLDMIRROR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VNCCS/3D"
    
    def load_model(self, device="cuda", sampling_strategy="uniform", enable_conf_filter=False, conf_threshold_percent=30.0):
        from src.models.models.worldmirror import WorldMirror
        
        print(f"🔄 Loading WorldMirror model (Strategy: {sampling_strategy}, Conf Filter: {enable_conf_filter}, Thresh: {conf_threshold_percent}%)")
        
        gs_params = {
            "enable_conf_filter": enable_conf_filter,
            "conf_threshold_percent": conf_threshold_percent
        }
        
        model = WorldMirror.from_pretrained(
            "tencent/HunyuanWorld-Mirror", 
            sampling_strategy=sampling_strategy,
            gs_params=gs_params
        )
        model = model.to(device)
        model.eval()
        print("✅ WorldMirror model loaded")
        
        return ({"model": model, "device": device},)


class VNCCS_WorldMirror3D:
    """Run 3D reconstruction on input images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WORLDMIRROR_MODEL",),
                "images": ("IMAGE",),
                "use_gsplat": ("BOOLEAN", {"default": True, "tooltip": "Enable Gaussian Splatting renderer (High Quality). If disabled, falls back to Point Cloud."}),
            },
            "optional": {
                "target_size": ("INT", {"default": 518, "min": 252, "max": 1024, "step": 14}),
                "offload_scheme": (["none", "model_cpu_offload", "sequential_cpu_offload"], {"default": "none"}),
                "stabilization": (["none", "panorama_lock"], {"default": "none"}),
                "confidence_percentile": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "apply_sky_mask": ("BOOLEAN", {"default": False, "tooltip": "Remove sky regions (requires onnxruntime and skyseg.onnx)"}),
                "filter_edges": ("BOOLEAN", {"default": True, "tooltip": "Remove artifact points at object boundaries"}),
                "edge_normal_threshold": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 90.0, "step": 0.5}),
                "edge_depth_threshold": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Distortion Mask Threshold. 0.5 means discard anything beyond 50% weight. Increase to fix ghosting, decrease to fix holes."}),
                "use_direct_points": ("BOOLEAN", {"default": False, "tooltip": "Use Direct Point Cloud Prediction (PTS3D) instead of Depth Projection. Bypasses camera distortion issues but relies on model's internal geometry."}),
                "use_consensus": ("BOOLEAN", {"default": False, "tooltip": "Enable Consensus Merging (Voxel Depth Filter). High-quality mode that removes depth outliers and ghosting from overlapping views. Use for Panoramas."}),
                "consensus_tolerance": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01, "tooltip": "Depth deviation threshold for consensus merging. Lower = tighter/cleaner walls, Higher = more forgiving."}),
                "resolution_mode": (["Standard", "HD", "Ultra"], {"default": "Standard", "tooltip": "Standard: Single-pass 518px. HD: Multi-pass tiling for 1024px backgrounds. Ultra: Overlapping patches for maximum detail."}),
                "camera_intrinsics": ("TENSOR", {"tooltip": "Optional: Intrinsics matrices from 'Equirect 360 to Views' node. REQUIRED for correct geometry at high FOVs."}),
                "camera_poses": ("TENSOR", {"tooltip": "Optional: Extrinsic matrices (poses) from 'Equirect 360 to Views' node. REQUIRED for correct patch alignment in HD/Ultra modes."}),
            }
        }
    
    RETURN_TYPES = ("PLY_DATA", "IMAGE", "IMAGE", "TENSOR", "TENSOR", "VNCCS_SPLAT")
    RETURN_NAMES = ("ply_data", "depth_maps", "normal_maps", "camera_poses", "camera_intrinsics", "raw_splats")
    FUNCTION = "run_inference"
    CATEGORY = "VNCCS/3D"
    
    def run_inference(self, model, images, use_gsplat=True, target_size=518, offload_scheme="none", stabilization="none", 
                      confidence_percentile=10.0, apply_sky_mask=False, filter_edges=True,
                      edge_normal_threshold=5.0, edge_depth_threshold=0.03, mask_threshold=0.5, use_direct_points=False,
                      use_consensus=False, consensus_tolerance=0.15, resolution_mode="Standard", camera_intrinsics=None, camera_poses=None):
        from torchvision import transforms
        
        if apply_sky_mask and not SKYSEG_AVAILABLE:
            print("⚠️ Sky segmentation requested but onnxruntime is missing. Ignoring.")
            apply_sky_mask = False
        
        # Ensure target_size is divisible by 14
        target_size = (target_size // 14) * 14
        
        worldmirror = model["model"]
        device = model["device"]
        
        # Convert ComfyUI images to tensor
        B, H, W, C = images.shape
        
        tensor_list = []
        mask_list = [] # Store per-view distortion masks if present
        
        converter = transforms.ToTensor()
        patch_size = 14
        
        for i in range(B):
            img = images[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Check for Alpha Channel (Distortion Mask)
            curr_mask = None
            if pil_img.mode == "RGBA":
                # Split Alpha
                r, g, b, a = pil_img.split()
                pil_img = Image.merge("RGB", (r, g, b))
                curr_mask = a # Grayscale mask [0..255]
            
            orig_w, orig_h = pil_img.size
            new_w = target_size
            new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
            
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            tensor_img = converter(pil_img)
            
            # Use same transform for mask
            if curr_mask is not None:
                curr_mask = curr_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
                tensor_mask = torch.from_numpy(np.array(curr_mask)).float() / 255.0
            else:
                tensor_mask = None

            if new_h > target_size:
                crop_start = (new_h - target_size) // 2
                tensor_img = tensor_img[:, crop_start:crop_start + target_size, :]
                if tensor_mask is not None:
                    tensor_mask = tensor_mask[crop_start:crop_start + target_size, :]
                
                # Update Intrinsics for Crop center shift
                if camera_intrinsics is not None:
                    camera_intrinsics[i, 1, 2] -= crop_start

            # --- CRITICAL FIX: Intrinsic Scaling ---
            # If we resized from orig_w/h to new_w/h, we MUST scale the geometric priors
            # Otherwise the model thinks the FOV is different than what is shown (Maznya bug)
            if camera_intrinsics is not None:
                scale_x = new_w / orig_w
                scale_y = new_h / orig_h
                camera_intrinsics[i, 0, 0] *= scale_x # fx
                camera_intrinsics[i, 1, 1] *= scale_y # fy
                camera_intrinsics[i, 0, 2] *= scale_x # cx
                camera_intrinsics[i, 1, 2] *= scale_y # cy
            
            tensor_list.append(tensor_img)
            mask_list.append(tensor_mask)

        # --- HD Tiling Logic ---
        # If HD/Ultra is enabled, we slice EACH view into multiple overlapping 518px patches.
        # This effectively increases point density by 4x (HD) or 9x (Ultra).
        if resolution_mode in ["HD", "Ultra"]:
            print(f"👺 [WorldMirror] HD Tiling Enabled ({resolution_mode}). Slicing views...")
            new_tensor_list = []
            new_mask_list = []
            new_intrinsics_list = []
            new_poses_list = [] # Trace poses for each patch
            
            # Grid size: HD = 2x2, Ultra = 3x3
            grid_n = 2 if resolution_mode == "HD" else 3
            
            for i in range(len(tensor_list)):
                img_t = tensor_list[i] # [C, H, W]
                mask_t = mask_list[i] # [H, W] or None
                
                _, h, w = img_t.shape
                
                # We want grid_n x grid_n patches of size 518
                patch_w, patch_h = 518, 518
                
                # Calculate step to cover the image with grid_n patches
                # step = (w - patch_w) / (grid_n - 1)
                step_x = (w - patch_w) / (grid_n - 1) if grid_n > 1 else 0
                step_y = (h - patch_h) / (grid_n - 1) if grid_n > 1 else 0
                
                for row in range(grid_n):
                    for col in range(grid_n):
                        oy = int(row * step_y)
                        ox = int(col * step_x)
                        
                        # Extract Patch
                        p_img = img_t[:, oy:oy+patch_h, ox:ox+patch_w]
                        p_mask = mask_t[oy:oy+patch_h, ox:ox+patch_w] if mask_t is not None else None
                        
                        new_tensor_list.append(p_img)
                        new_mask_list.append(p_mask)
                        
                        # Correct Intrinsics if provided
                        if camera_intrinsics is not None:
                            # camera_intrinsics is already scaled to full-size patch parent
                            orig_K = camera_intrinsics[i].clone()
                            # Correction: cx' = cx - ox, cy' = cy - oy
                            orig_K[0, 2] -= ox
                            orig_K[1, 2] -= oy
                            new_intrinsics_list.append(orig_K)
                        
                        # Correct/Inject Poses if provided
                        if camera_poses is not None:
                            # Each patch inherits the same orientation as its parent view
                            # This staples them together in 3D space
                            orig_P = camera_poses[i].clone()
                            new_poses_list.append(orig_P)
            
            tensor_list = new_tensor_list
            mask_list = new_mask_list
            if camera_intrinsics is not None:
                camera_intrinsics = torch.stack(new_intrinsics_list)
            if camera_poses is not None:
                camera_poses = torch.stack(new_poses_list)
            
            # Update B and target_size for the rest of the function
            B = len(tensor_list)
            target_size = 518 # Patches are always 518
            print(f"   -> Result: {B} patches of {target_size}px.")

        
        # device management
        execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_device = next(worldmirror.parameters()).device
        
        imgs_tensor = torch.stack(tensor_list)
        # Use execution_device (GPU) for inputs, as accelerate expects inputs on the compute device
        imgs_tensor = imgs_tensor.unsqueeze(0).to(execution_device)

        print(f"🚀 Running WorldMirror inference on {B} images (offload: {offload_scheme})...")
        
        if offload_scheme != "none" and execution_device.type == "cuda":
            if offload_scheme == "sequential_cpu_offload":
                 # Manual Block Offloading (OOM fix for <16GB/24GB cards)
                 # We keep the main model on CPU, but move heads to GPU.
                 # The transformer handles its own internal block moving.
                 
                 # 0. STRIP ACCELERATE HOOKS if present (conflicting with manual offload)
                 def recursive_remove_hooks(module):
                     if hasattr(module, "_hf_hook"):
                         del module._hf_hook
                     if hasattr(module, "_old_forward"):
                         module.forward = module._old_forward
                         del module._old_forward
                     for child in module.children():
                         recursive_remove_hooks(child)

                 try:
                     recursive_remove_hooks(worldmirror)
                     # Also try accelerate's official method just in case
                     from accelerate.hooks import remove_hook_from_module
                     remove_hook_from_module(worldmirror, recurse=True)
                 except Exception as e:
                     print(f"⚠️ Failed to remove hooks: {e}")

                 # 1. Enable manual offload in transformer
                 if hasattr(worldmirror.visual_geometry_transformer, "manual_offload"):
                      worldmirror.visual_geometry_transformer.manual_offload = True
                 
                 # 2. Move heads to GPU manually (they are small-ish)
                 heads = [
                     getattr(worldmirror, "cam_head", None),
                     getattr(worldmirror, "pts_head", None),
                     getattr(worldmirror, "depth_head", None),
                     getattr(worldmirror, "norm_head", None),
                     getattr(worldmirror, "gs_head", None),
                 ]
                 for head in heads:
                     if head is not None:
                         # Check for corrupted/meta state from previous runs
                         param = next(head.parameters(), None)
                         if param is not None and param.device.type == 'meta':
                             raise RuntimeError(
                                 "CRITICAL: WorldMirror model is in a corrupted state (weights are on 'meta' device). "
                                 "This happens if a previous run crashed or used a different offload scheme. "
                                 "Please RESTART ComfyUI or invalidating the 'Load WorldMirror Model' node to reload fresh weights."
                             )
                         
                         # If head is on meta device, this might fail unless we materialize it.
                         head.to(execution_device)
            
            elif offload_scheme == "model_cpu_offload":
                try:
                    from accelerate import cpu_offload
                    cpu_offload(worldmirror, execution_device=execution_device)
                except ImportError:
                    print("⚠️ Accelerate not installed, ignoring model_cpu_offload")
                except Exception as e:
                    print(f"⚠️ Offload failed: {e}")
                    # Do NOT fallback to full GPU load if offload requested, as it likely causes OOM
        else:
            # Standard behavior: move everything to GPU
            param = next(worldmirror.parameters())
            # Check if likely meta device
            if param.device.type == 'meta':
                 print("⚠️ Model is on meta device, attempting to materialize empty weights...")
                 worldmirror.to_empty(device=execution_device)
                 # This is risky as weights are lost. But better than crash.
                 # Ideally user should reload model.
            elif param.device != execution_device:
                 worldmirror.to(execution_device)

        views = {"img": imgs_tensor}
        if camera_poses is not None:
             # camera_poses is [S, 4, 4]. Model expects [B, S, 4, 4]
             # Note: logic in worldmirror.py strips it to [B, S, 3, 4] internally
             views["camera_poses"] = camera_poses.unsqueeze(0).to(execution_device)
        
        # Prepare Distortion Mask (from Alpha Channel) - Moved UP before inference
        distortion_mask_np = None
        filter_mask_tensor = None
        if any(m is not None for m in mask_list):
            print("👺 Preparing distortion masks for filter...")
            clean_masks = []
            for m in mask_list:
                if m is not None:
                     clean_masks.append(m.numpy())
                else:
                     clean_masks.append(np.ones((target_size, target_size), dtype=np.float32))
            distortion_mask_np = np.stack(clean_masks, axis=0) # [S, H, W]
            
            # Create logic mask for Splat Filtering (Thresholded)
            filter_mask_tensor = torch.from_numpy(distortion_mask_np > mask_threshold).to(execution_device)
            filter_mask_tensor = filter_mask_tensor.unsqueeze(0)
            
            # Store RAW masks (0..1) for Refinement Weighted Loss
            raw_distortion_masks = torch.from_numpy(distortion_mask_np).unsqueeze(0).to(execution_device)
        else:
            raw_distortion_masks = None


        cond_flags = [0, 0, 0]
        if camera_intrinsics is not None:
             print("👺 [WorldMirror] Enabling Camera Intrinsics Conditioning...")
             # camera_intrinsics is [S, 3, 3]. Model expects [B, S, 3, 3]
             views["camera_intrs"] = camera_intrinsics.unsqueeze(0).to(execution_device)
             cond_flags[2] = 1 # Enable Rays/Intrinsics conditioning
        
        if camera_poses is not None:
             print("👺 [WorldMirror] Enabling Camera Pose Conditioning...")
             # Since we injected views["camera_poses"] above, we just enable the flag
             cond_flags[0] = 1 # Enable Pose conditioning
        
        try:
            # Override GS State based on toggle
            original_gs_state = getattr(worldmirror, "enable_gs", True)
            
            # Force disable if gsplat lib is missing
            if use_gsplat and not GSPLAT_AVAILABLE:
                print("⚠️ gsplat requested but library not found. Falling back to Point Cloud.")
                worldmirror.enable_gs = False
            else:
                worldmirror.enable_gs = use_gsplat

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    predictions = worldmirror(
                        views=views, 
                        cond_flags=cond_flags, 
                        stabilization=stabilization, 
                        confidence_percentile=confidence_percentile,
                        filter_mask=filter_mask_tensor,
                        use_direct_points=use_direct_points,
                        use_consensus=use_consensus,
                        consensus_tolerance=consensus_tolerance
                    )
            
            # Post-inference: Attach distortion masks for the Refiner
            if raw_distortion_masks is not None:
                predictions["distortion_masks"] = raw_distortion_masks
        finally:
            # Restore original state to avoid side effects
            if hasattr(worldmirror, "enable_gs"):
                worldmirror.enable_gs = original_gs_state

            # Cleanup for Manual Sequential Offload
            if offload_scheme == "sequential_cpu_offload":
                 # Move heads back to CPU
                 heads = [
                        getattr(worldmirror, "cam_head", None),
                        getattr(worldmirror, "pts_head", None),
                        getattr(worldmirror, "depth_head", None),
                        getattr(worldmirror, "norm_head", None),
                        getattr(worldmirror, "gs_head", None),
                    ]
                 for head in heads:
                        if head is not None:
                            head.to("cpu")
                 
                 # Disable manual flag (reset state)
                 if hasattr(worldmirror.visual_geometry_transformer, "manual_offload"):
                         worldmirror.visual_geometry_transformer.manual_offload = False
                 
                 torch.cuda.empty_cache()

            if offload_scheme == "none" and original_device.type == "cpu":
                 # If we moved it manually to GPU, and it came from CPU (which is now default),
                 # maybe we should move it back to save memory for other nodes?
                 # Actually, usually ComfyUI models stay where they are put.
                 # But if user selected "cpu" in loader, they expect it in CPU.
                 # Let's move it back if it wasn't offloaded by accelerate (accelerate handles it).
                 worldmirror.to("cpu")
                 torch.cuda.empty_cache()

        print("✅ Inference complete")
        
        # ============================================================================
        # Post-Processing: Filtering & Sky Masking (Ported)
        # ============================================================================
        
        S, H, W = predictions["depth"].shape[1:4] # Get dimensions from depth map
        B_batch = predictions["depth"].shape[0]   # Although we usually have B=1 in this node structure

        # 1. Compute Sky Mask
        sky_mask_np = None
        if apply_sky_mask:
            print("🌤️ Computing sky masks...")
            sky_model_path = os.path.join(folder_paths.models_dir, "skyseg.onnx")
            
            # Download if missing
            if not os.path.exists(sky_model_path):
                print(f"⬇️ Downloading skyseg.onnx to {sky_model_path}...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", 
                    sky_model_path
                )
            
            if os.path.exists(sky_model_path):
                try:
                    skyseg_session = onnxruntime.InferenceSession(sky_model_path)
                    sky_mask_list = []
                    
                    # We need original images for sky seg ideally, but using tensor inputs converted back is fine
                    # images is [B, H_orig, W_orig, 3] or similar? No, images is [B, H, W, 3] from Comfy
                    # But note: we need the images that match the INFERENCE resolution (H,W) for mask consistency
                    
                    # Convert input tensor images to numpy for skyseg
                    # imgs_tensor is [1, S, 3, H, W]
                    for i in range(S):
                        img_np = imgs_tensor[0, i].permute(1, 2, 0).cpu().numpy() # [H, W, 3]
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        sky_mask_frame = segment_sky(img_np, skyseg_session)
                        # Resize mask to match H×W if needed (segment_sky handles it but double check)
                        if sky_mask_frame.shape[0] != H or sky_mask_frame.shape[1] != W:
                            sky_mask_frame = cv2.resize(sky_mask_frame, (W, H))
                        sky_mask_list.append(sky_mask_frame)
                    
                    sky_mask_np = np.stack(sky_mask_list, axis=0) # [S, H, W]
                    sky_mask_np = sky_mask_np > 0 # Binary: True = non-sky
                    print(f"✅ Sky masks computed for {S} frames")
                except Exception as e:
                    print(f"❌ Sky segmentation failed: {e}")
                    sky_mask_np = None
            else:
                print("❌ Failed to download skyseg.onnx")

        # 2. Compute Filter Mask
        
        # Prepare Distortion Mask (Moved up)
        # distortion_mask_np is already computed above
        if distortion_mask_np is not None:
             print("👺 Re-using pre-computed distortion mask for post-filtering...")
        
        print("🔍 Computing geometric filter mask...")
        pts3d_conf_np = predictions["pts3d_conf"][0].detach().cpu().numpy()
        depth_preds_np = predictions["depth"][0].detach().cpu().numpy()
        normal_preds_np = predictions["normals"][0].detach().cpu().numpy()
        
        final_mask = create_filter_mask(
            pts3d_conf=pts3d_conf_np,
            depth_preds=depth_preds_np,
            normal_preds=normal_preds_np,
            sky_mask=sky_mask_np,
            distortion_mask=distortion_mask_np,
            confidence_percentile=confidence_percentile,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=True,
            apply_edge_mask=filter_edges,
            apply_sky_mask=apply_sky_mask
        ) # [S, H, W] bool array
        
        # 3. Apply Limit to outputs
        # We need to filter 'pts3d' (point cloud) AND 'splats' (gaussians)
        
        # Filter Point Cloud [1, S, H, W, 3] -> Flat List of valid points
        # Actually, ComfyUI style output keeps structure usually, but for PLY saving we want list
        # We will RETURN the mask or filtered points. 
        # Existing code accesses predictions["pts3d"] which is [1, S, H, W, 3]
        
        # Let's flatten and filter pts3d for the output dictionary
        # BUT: predictions["pts3d"] maps to pixels. If we remove pixels, we lose grid structure.
        # The PLY saver handles flattening. We should zero out invalid points or mark them.
        # Or better: Provide the filtered list directly in ply_data.
        
        all_pts_list = []
        all_conf_list = []
        
        if "splats" in predictions:
            # Note: We DO NOT filter splats here with final_mask!
            # Why: If enable_prune=True (default), splats are a sparse point cloud and do not map 1:1 to pixels.
            # Using a pixel-grid mask (final_mask) on sparse splats is mathematically wrong and causes size mismatches.
            # We pass splats raw, trusting the Model's internal GaussianSplatRenderer to have pruned/filtered them.
            print(f"ℹ️ Passing native Gaussian Splats (Pruned/Internal Filter).")
            # Ensure consistency of list vs tensor
            pass
        
        # Prepare mask for Point Cloud filtering
        final_mask_flat = torch.from_numpy(final_mask.reshape(-1)).to(execution_device)

        # Filter Points 3D (for PLY_DATA)
        # pts3d is [1, S, H, W, 3]. Flatten to [S*H*W, 3] and filter
        filtered_pts = None
        if "pts3d" in predictions:
            pts = predictions["pts3d"][0].reshape(-1, 3)
            # pts_conf = predictions["pts3d_conf"][0].reshape(-1)
            
            filtered_pts = pts[final_mask_flat.to(pts.device)]
            # filtered_conf = pts_conf[final_mask_flat]

        ply_data = {
            "pts3d": predictions.get("pts3d"), # Raw structured points (useful for depth maps)
            "pts3d_filtered": filtered_pts, # Filtered flat points
            "pts3d_conf": predictions.get("pts3d_conf"),
            "splats": predictions.get("splats"), # Now contains FILTERED splats
            "images": imgs_tensor,
            "filter_mask": final_mask_flat, # Pass mask for color filtering in fallback
            "camera_poses": predictions.get("camera_poses"),
            "camera_intrs": predictions.get("camera_intrs"),
        } 
        

        
        depth_tensor = predictions.get("depth")
        if depth_tensor is not None:
            depth = depth_tensor[0]
            depth_min = depth.min()
            depth_max = depth.max()
            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            depth_rgb = depth_norm.repeat(1, 1, 1, 3)
            depth_out = depth_rgb.cpu().float()
        else:
            depth_out = torch.zeros(B, target_size, target_size, 3)
        
        normal_tensor = predictions.get("normals")
        if normal_tensor is not None:
            normals = normal_tensor[0]
            normals_out = ((normals + 1) / 2).cpu().float()
        else:
            normals_out = torch.zeros(B, target_size, target_size, 3)
        
        # Move to CPU for ComfyUI compatibility
        camera_poses_out = predictions.get("camera_poses")
        camera_intrs_out = predictions.get("camera_intrs")
        
        if camera_poses_out is not None:
            camera_poses_out = camera_poses_out.cpu().float()
        if camera_intrs_out is not None:
            camera_intrs_out = camera_intrs_out.cpu().float()
        
        # Add internal images to predictions for SplatRefiner
        predictions["images"] = imgs_tensor
        predictions["raw_images"] = images

        return (ply_data, depth_out, normals_out, camera_poses_out, camera_intrs_out, predictions)


class VNCCS_Equirect360ToViews:
    """Convert 360° equirectangular panorama to perspective views."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
            },
            "optional": {
                # Unlocked limits for Brute Force/Consensus experiments
                "fov": ("INT", {"default": 90, "min": 1, "max": 179}),
                "yaw_step": ("INT", {"default": 45, "min": 1, "max": 180}),
                "pitches": ("STRING", {"default": "0,-60,60"}),
                "output_size": ("INT", {"default": 518, "min": 252, "max": 4096, "step": 14}),
                "dynamic_fov": ("BOOLEAN", {"default": True, "tooltip": "Automatically reduce FOV looking up/down to minimize stretching"}),
                "sampling_mode": (["bilinear", "bicubic"], {
                    "default": "bilinear",
                    "tooltip": "Perspective extraction resampling. bilinear is safer for 3D reconstruction; bicubic is sharper but can create colored edge ringing."
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-view camera/intrinsics diagnostics for panorama slices."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("views", "intrinsics", "camera_poses")
    FUNCTION = "extract_views"
    CATEGORY = "VNCCS/3D"
    
    def extract_views(self, panorama, quality="Standard (518)", fov=90, yaw_step=45, pitches="0,-60,60", output_size=518,
                      dynamic_fov=True, yaw_offset=0.0, pose_convention="opencv_c2w",
                      pose_yaw_offset=0.0, pose_pitch_offset=0.0, pose_roll_offset=0.0,
                      intrinsics_focal_scale=1.0, sampling_mode="bilinear", debug_log=False):

        pitch_list = [int(p.strip()) for p in pitches.split(",")]
        
        img_np = (panorama[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        yaw_angles = list(range(0, 360, yaw_step))
        
        views = []
        total = len(yaw_angles) * len(pitch_list)
        
        print(f"🔄 Extracting {total} views from 360° panorama...")
        print(f"   - Settings: FOV={fov}, Step={yaw_step}, Output={output_size}")
        print(
            f"   - Features: DynamicFOV={dynamic_fov}, PoseConvention={pose_convention}, "
            f"PoseOffset=({pose_yaw_offset}, {pose_pitch_offset}, {pose_roll_offset}), "
            f"FocalScale={intrinsics_focal_scale}, Sampling={sampling_mode}"
        )
        if debug_log:
            print(
                "[Equirect360ToViews][DEBUG][BEGIN] "
                f"panorama_shape={tuple(panorama.shape)}, pitches={pitch_list}, "
                f"yaw_angles={yaw_angles}, output_size={output_size}"
            )

        intrinsics_list = []
        poses_list = []
        center_ray_mismatches = 0

        for yaw in yaw_angles:
            for pitch in pitch_list:
                effective_yaw = (yaw + yaw_offset) % 360
                view, focal = equirect_to_perspective(
                    pil_img,
                    fov_deg=fov,
                    yaw_deg=effective_yaw, # Apply Offset and Wrap
                    pitch_deg=pitch,
                    output_size=(output_size, output_size),
                    dynamic_fov=dynamic_fov,
                    sampling_mode=sampling_mode,
                )

                # Construct Intrinsics Matrix [3, 3]
                # [ fx,  0, cx ]
                # [  0, fy, cy ]
                # [  0,  0,  1 ]
                # WorldMirror expects intrinsics normalized by image size? 
                # According to worldmirror.py line 306: intrinsics[:, :, 0, 0] / w
                # So here we store absolute pixels, worldmirror will normalize them.
                cx, cy = output_size / 2.0, output_size / 2.0
                fx = focal * (output_size / 2.0) / math.tan(math.radians(fov)/2) # Wait, f is 1/tan(fov_rad/2)
                # Correct calculation: f_pixels = (output_size / 2.0) * focal
                f_pixels = (output_size / 2.0) * focal * intrinsics_focal_scale
                intra = np.array([
                    [f_pixels, 0, cx],
                    [0, f_pixels, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                intrinsics_list.append(intra)
                
                # Construct Pose Matrix [4, 4] (C2W)
                # Based on Yaw/Pitch/Roll logic in equirect_to_perspective
                # We use (yaw + yaw_offset) to match the view rotation
                pose = build_camera_pose_matrix(
                    pitch + pose_pitch_offset,
                    yaw + yaw_offset + pose_yaw_offset,
                    pose_roll_offset,
                    pose_convention,
                )
                poses_list.append(pose)
                
                view_np = np.array(view).astype(np.float32) / 255.0
                
                print(f"   - View: Pitch={pitch:3d}, Yaw={yaw:3d} | Proj: Rectilinear")
                if debug_log:
                    effective_fov = math.degrees(2.0 * math.atan(1.0 / max(float(focal), 1e-8)))
                    proj_R = build_rotation_matrix(pitch, effective_yaw, 0.0)
                    proj_center_ray = proj_R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    proj_center_ray = proj_center_ray / max(float(np.linalg.norm(proj_center_ray)), 1e-8)
                    pose_right = pose[:3, 0]
                    pose_up = pose[:3, 1]
                    pose_forward = pose[:3, 2]
                    pose_forward_n = pose_forward / max(float(np.linalg.norm(pose_forward)), 1e-8)
                    dot_forward = float(np.dot(proj_center_ray, pose_forward_n))
                    dot_backward = float(np.dot(proj_center_ray, -pose_forward_n))
                    if dot_forward < 0.999:
                        center_ray_mismatches += 1
                    pose_yaw_dbg, pose_pitch_dbg, pose_roll_dbg = _camera_debug_angles_from_pose(pose)
                    print(
                        f"[Equirect360ToViews][DEBUG][VIEW {len(views):03d}] "
                        f"yaw={yaw:.3f}, pitch={pitch:.3f}, effective_yaw={effective_yaw:.3f}, "
                        f"base_fov={float(fov):.3f}, effective_fov={effective_fov:.3f}, focal_norm={float(focal):.6f}, "
                        f"K=(fx={float(intra[0,0]):.4f}, fy={float(intra[1,1]):.4f}, "
                        f"cx={float(intra[0,2]):.4f}, cy={float(intra[1,2]):.4f}), "
                        f"pose_angles=(yaw={pose_yaw_dbg:.3f}, pitch={pose_pitch_dbg:.3f}, roll={pose_roll_dbg:.3f}), "
                        f"right={_fmt_vec3_np(pose_right)}, up={_fmt_vec3_np(pose_up)}, forward={_fmt_vec3_np(pose_forward)}, "
                        f"proj_center_ray={_fmt_vec3_np(proj_center_ray)}, "
                        f"proj_center_dot_pose_forward={dot_forward:.6f}, "
                        f"proj_center_dot_pose_backward={dot_backward:.6f}"
                    )
                     
                views.append(view_np)
        
        views_tensor = torch.from_numpy(np.stack(views, axis=0))
        intrinsics_tensor = torch.from_numpy(np.stack(intrinsics_list, axis=0))
        poses_tensor = torch.from_numpy(np.stack(poses_list, axis=0))
        
        print(f"✅ Extracted {total} views (Channels: {views_tensor.shape[-1]})")
        if debug_log:
            print(
                "[Equirect360ToViews][DEBUG][END] "
                f"views={tuple(views_tensor.shape)}, intrinsics={tuple(intrinsics_tensor.shape)}, "
                f"poses={tuple(poses_tensor.shape)}, center_ray_mismatches={center_ray_mismatches}"
            )
        
        return (views_tensor, intrinsics_tensor, poses_tensor)


def extract_splat_params(data):
    """
    Robustly extract Gaussian Splatting parameters from WorldMirror output.
    Handles batches, lists, and point cloud fallbacks.
    """
    if data is None: 
        print("❌ [extract_splat_params] ERROR: Input data is None")
        return None
    
    print(f"🔍 [extract_splat_params] Input keys: {list(data.keys())}")
    splats = data.get("splats")
    pts3d = data.get("pts3d")
    images = data.get("images")
    
    # Debug: print raw value of splats
    print(f"🔍 [extract_splat_params] splats raw value: {type(splats)}, is None: {splats is None}")
    if splats is not None:
        if isinstance(splats, dict):
            print(f"🔍 [extract_splat_params] splats dict keys: {list(splats.keys())}")
        elif isinstance(splats, (list, tuple)):
            print(f"🔍 [extract_splat_params] splats is list/tuple with {len(splats)} elements")
            if len(splats) > 0:
                print(f"🔍 [extract_splat_params] splats[0] type: {type(splats[0])}")
        else:
            print(f"🔍 [extract_splat_params] splats is {type(splats)}")
    
    device = torch.device("cpu") # Extract to CPU for saving
    
    # Try to extract from splats
    if splats is not None:
        print(f"🔍 [extract_splat_params] Splats found, type: {type(splats)}")
        
        # Handle list-wrapped splats from some ComfyUI custom types
        if isinstance(splats, list) and len(splats) > 0:
            print(f"🔍 [extract_splat_params] splats is list with {len(splats)} elements, taking first")
            splats = splats[0]
            
        if isinstance(splats, dict) and len(splats) > 0:
            print(f"✅ [extract_splat_params] Processing splats dict with keys: {list(splats.keys())}")
            
            # Extract parameters from splats dict
            keys = ["means", "scales", "quats", "sh", "colors", "opacities"]
            params = {}
            for k in keys:
                v = splats.get(k)
                if v is None: continue
                # Handle potential list wrapper from some versions
                if isinstance(v, list): v = v[0]
                # Handle batch dimension [B, N, ...] -> [N, ...], but do not
                # collapse SH tensors that are already [N, SH, 3].
                if v.dim() >= 3 and v.shape[0] == 1:
                    v = v[0]
                params[k] = v.detach().cpu().float()
            
            if "means" not in params:
                print("❌ [extract_splat_params] No 'means' in splats, falling back to pts3d")
            else:
                means = params["means"].reshape(-1, 3)
                
                scales = params.get("scales")
                if scales is not None:
                    scales = scales.reshape(-1, 3)
                    # Heuristic: if scales are mostly negative, they are in log space
                    if scales.to(torch.float32).mean() < -0.5:
                        print(f"🔍 [extract_splat_params] Detecting log-scales, applying exp()")
                        scales = torch.exp(scales)
                else:
                    print("⚠️ [extract_splat_params] No scales in splats, using default")
                    scales = torch.ones(means.shape[0], 3) * 0.01
                
                quats = params.get("quats")
                if quats is not None:
                    quats = quats.reshape(-1, 4)
                    if quats.shape[1] == 4 and quats.abs().sum() < 1e-6:
                        quats[:, 0] = 1.0  # Default to identity if all zeros
                else:
                    quats = torch.zeros(means.shape[0], 4)
                    quats[:, 0] = 1.0
                
                opacities = params.get("opacities")
                if opacities is not None:
                    opacities = opacities.reshape(-1)
                    # Heuristic: if opacities have values far outside [0, 1], they are logits
                    if opacities.min() < -2.0 or opacities.max() > 2.0:
                        print(f"🔍 [extract_splat_params] Detecting logit-opacities, applying sigmoid()")
                        opacities = torch.sigmoid(opacities)
                else:
                    opacities = torch.ones(means.shape[0]) * 0.9
                
                # Colors: convert to RGB [0, 1] if they look like SH coefficients
                colors_data = params.get("sh", params.get("colors"))
                if colors_data is not None:
                    if colors_data.dim() == 3:  # [N, SH, 3] -> DC is first SH
                        colors_data = colors_data[:, 0, :]
                    colors_data = colors_data.reshape(-1, 3)
                    
                    # Consistent with save_utils heuristic
                    c_np = colors_data.numpy()
                    if c_np.min() < -0.1 or c_np.max() > 1.1:
                        # Definitely SH
                        colors_data = colors_data * 0.28209479177387814 + 0.5
                    colors_data = torch.clamp(colors_data, 0.0, 1.0)
                else:
                    colors_data = torch.ones(means.shape[0], 3) * 0.5
                
                print(f"📊 [extract_splat_params] Gaussian Stats:")
                print(f"   - Means:  {means.shape} min={means.min(dim=0)[0].tolist()}, max={means.max(dim=0)[0].tolist()}")
                if scales is not None: print(f"   - Scales: {scales.shape}")
                if quats is not None: print(f"   - Quats:  {quats.shape}")
                if colors_data is not None: print(f"   - Colors: {colors_data.shape}")
                if opacities is not None: print(f"   - Opacity:{opacities.shape} min={opacities.min().item():.3f}")
                
                return means, scales, quats, colors_data, opacities
        else:
            print(f"⚠️ [extract_splat_params] splats is {type(splats)}, not a valid dict")
    
    # Fallback: Convert point cloud to dummy Gaussians
    if pts3d is not None or data.get("pts3d_filtered") is not None:
        print("🔍 [extract_splat_params] Using pts3d fallback (point cloud mode)")
        
        if data.get("pts3d_filtered") is not None:
             print("✨ Using FILTERED point cloud")
             means = data["pts3d_filtered"].detach().cpu().float()
        else:
             means = pts3d[0].view(-1, 3).detach().cpu().float()
             
        N = means.shape[0]
        
        # Cap at 2M to avoid crashing viewers
        if N > 2000000:
            idx = torch.randperm(N)[:2000000]
            means = means[idx]
            N = 2000000
        else:
            idx = None
            
        scales = torch.ones(N, 3) * 0.005  # Small visible splats
        quats = torch.zeros(N, 4)
        quats[:, 0] = 1.0  # Identity rotation
        opacities = torch.ones(N) * 10.0  # High value (will be sigmoid'ed to ~1.0)
        
        if images is not None:
            S = pts3d.shape[1]
            colors_data = images[0, :S].permute(0, 2, 3, 1).reshape(-1, 3)
            
            # Apply filter mask if available (Crucial for mismatch fix)
            mask = data.get("filter_mask")
            if mask is not None:
                # Ensure mask matches raw size
                if mask.shape[0] == colors_data.shape[0]:
                    print(f"✨ [extract_splat_params] Applying filter mask to colors | Mask: {mask.shape} | Colors: {colors_data.shape}")
                    colors_data = colors_data[mask.to(colors_data.device)]
                else:
                    print(f"⚠️ [extract_splat_params] Mask size {mask.shape[0]} != Colors size {colors_data.shape[0]}, ignoring")
            
            if idx is not None:
                colors_data = colors_data[idx]
            colors_data = colors_data.detach().cpu().float()
            
            # DEBUG: Color Stats
            if colors_data.numel() > 0:
                print(f"🎨 [extract_splat_params] Colors Stats: Min={colors_data.min():.3f}, Max={colors_data.max():.3f}, Mean={colors_data.mean():.3f}")
            else:
                print("🎨 [extract_splat_params] Colors Stats: empty tensor")
        else:
            colors_data = torch.ones(N, 3) * 0.5
        
        print(f"📊 [extract_splat_params] Point Cloud Stats: {N:,} points | GSPLAT_AVAILABLE={GSPLAT_AVAILABLE}")
            
        return means, scales, quats, colors_data, opacities
        
    print("❌ [extract_splat_params] No valid data found")
    return None



class VNCCS_SavePLY:
    """Save 3D reconstruction as PLY file."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "filename": ("STRING", {"default": "output"}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_ply"
    CATEGORY = "VNCCS/3D"
    OUTPUT_NODE = True
    
    def _rotation_matrix(self, rx, ry, rz):
        """Create rotation matrix from Euler angles (degrees)."""
        rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ], dtype=torch.float32)
        
        Ry = torch.tensor([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ], dtype=torch.float32)
        
        Rz = torch.tensor([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return Rz @ Ry @ Rx

    def _rotate_quaternions(self, quats, R):
        """
        Rotate quaternions by rotation matrix R.
        quats: [N, 4] (w, x, y, z) or (x, y, z, w)? gsplat usually (w, x, y, z) or (x, y, z, w).
        3DGS usually uses (w, x, y, z).
        """
        # Convert R to quaternion q_R checking typical conversion math
        # Minimal implementation of Matrix to Quat (assuming R is pure rotation)
        # R is [3, 3]
        
        # We need batch multiplication.
        # Use pytorch3d or similar logic if available? No, simple math.
        # q_new = q_R * q_old
        
        # 1. R -> q_R (w, x, y, z)
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        
        q_R = torch.tensor([w, x, y, z], device=quats.device, dtype=quats.dtype)
        
        # 2. Quaternion Multiplication (Hamilton Product)
        # (w1, x1, y1, z1) * (w2, x2, y2, z2)
        # Input quats is [N, 4]
        
        # q_R is [4]
        w1, x1, y1, z1 = q_R[0], q_R[1], q_R[2], q_R[3]
        w2, x2, y2, z2 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=1)

    def _get_unique_path(self, directory, filename, suffix, extension):
        filename = self._sanitize_filename(filename)
        counter = 1
        while True:
            # Use 5 digits like ComfyUI SaveImage
            full_filename = f"{filename}_{counter:05}_{suffix}.{extension}"
            path = os.path.join(directory, full_filename)
            if not os.path.exists(path):
                return path
            counter += 1

    def _sanitize_filename(self, filename):
        base = os.path.basename(str(filename or "output").replace("\\", "/"))
        base = os.path.splitext(base)[0]
        base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" ._")
        return base or "output"

    def _safe_quantile(self, values, q, max_values=1_000_000):
        flat = values.detach().cpu().float().reshape(-1)
        if flat.numel() > max_values:
            step = max(1, flat.numel() // max_values)
            flat = flat[::step][:max_values]
        return torch.quantile(flat, q)

    def _save_splat_preview_cache(self, path, means, scales, rotations, rgbs, opacities, chunk_size=1_000_000):
        """Write gsplat.js' compact 32-byte/gaussian binary preview cache."""
        splat_path = os.path.splitext(str(path))[0] + ".splat"
        tmp_path = f"{splat_path}.tmp"
        n = int(means.shape[0])
        sh_c0 = 0.28209479177387814

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        try:
            with open(tmp_path, "wb") as handle:
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    count = end - start
                    block = np.zeros((count, 32), dtype=np.uint8)
                    floats = block.view("<f4").reshape(count, 8)

                    floats[:, 0:3] = means[start:end].numpy()
                    floats[:, 3:6] = scales[start:end].numpy()

                    rgb = (rgbs[start:end] * sh_c0 + 0.5).clamp(0.0, 1.0)
                    block[:, 24:27] = (rgb.numpy() * 255.0 + 0.5).astype(np.uint8)

                    alpha = opacities[start:end]
                    if alpha.numel() and (float(alpha.min()) < 0.0 or float(alpha.max()) > 1.0):
                        alpha = torch.sigmoid(alpha)
                    block[:, 27] = (alpha.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)

                    quat = rotations[start:end]
                    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    block[:, 28:32] = (quat.clamp(-1.0, 1.0).numpy() * 128.0 + 128.0).clip(0, 255).astype(np.uint8)

                    handle.write(block.tobytes())

            os.replace(tmp_path, splat_path)
            size_mb = os.path.getsize(splat_path) / (1024 * 1024)
            print(f"⚡ [SavePLY] Wrote preview splat cache: {os.path.basename(splat_path)} ({size_mb:.2f} MB)")
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _save_gs_ply_safe(
        self,
        path,
        means,
        scales,
        rotations,
        rgbs,
        opacities,
        quantile_threshold=0.95,
        apply_scale_filter=True,
    ):
        from plyfile import PlyData, PlyElement

        means = means.detach().cpu().float().reshape(-1, 3)
        scales = scales.detach().cpu().float().reshape(-1, 3).clamp_min(1e-8)
        rotations = rotations.detach().cpu().float().reshape(-1, 4)
        rgbs = rgbs.detach().cpu().float().reshape(-1, 3)
        opacities = opacities.detach().cpu().float().reshape(-1)

        n = min(means.shape[0], scales.shape[0], rotations.shape[0], opacities.shape[0])
        if n == 0:
            raise ValueError("No Gaussian splats to save")
        if rgbs.shape[0] == 1 and n > 1:
            rgbs = rgbs.expand(n, -1)
        elif rgbs.shape[0] != n:
            n = min(n, rgbs.shape[0])
        if opacities.shape[0] == 1 and n > 1:
            opacities = opacities.expand(n)
        means, scales, rotations, rgbs, opacities = (
            means[:n], scales[:n], rotations[:n], rgbs[:n], opacities[:n]
        )

        if apply_scale_filter and quantile_threshold is not None:
            max_scale = scales.max(dim=-1)[0]
            scale_threshold = self._safe_quantile(max_scale, quantile_threshold)
            keep = max_scale <= scale_threshold
            kept = int(keep.sum().item())
            if kept <= 0:
                keep = torch.ones_like(max_scale, dtype=torch.bool)
                kept = int(keep.sum().item())
            if kept != n:
                print(f"📉 [SavePLY] Scale filter: {n} -> {kept} gaussians")

            means = means[keep]
            scales = scales[keep]
            rotations = rotations[keep]
            rgbs = rgbs[keep]
            opacities = opacities[keep]

        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        attributes += [f"f_dc_{i}" for i in range(3)]
        attributes += ["opacity"]
        attributes += [f"scale_{i}" for i in range(3)]
        attributes += [f"rot_{i}" for i in range(4)]

        elements = np.empty(means.shape[0], dtype=[(attribute, "f4") for attribute in attributes])
        means_np = means.numpy()
        rgbs_np = rgbs.numpy()
        opacities_np = opacities.numpy()
        scales_np = scales.log().numpy()
        rotations_np = rotations.numpy()

        elements["x"] = means_np[:, 0]
        elements["y"] = means_np[:, 1]
        elements["z"] = means_np[:, 2]
        elements["nx"] = 0.0
        elements["ny"] = 0.0
        elements["nz"] = 0.0
        for i in range(3):
            elements[f"f_dc_{i}"] = rgbs_np[:, i]
            elements[f"scale_{i}"] = scales_np[:, i]
        elements["opacity"] = opacities_np
        for i in range(4):
            elements[f"rot_{i}"] = rotations_np[:, i]

        tmp_path = f"{path}.tmp"
        try:
            PlyData([PlyElement.describe(elements, "vertex")]).write(tmp_path)
            os.replace(tmp_path, str(path))
            try:
                self._save_splat_preview_cache(path, means, scales, rotations, rgbs, opacities)
            except Exception as cache_error:
                print(f"⚠️ [SavePLY] Preview splat cache failed: {cache_error}")
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


    def save_ply(self, ply_data, filename="output", save_pointcloud=False, save_gaussians=True,
                 rotate_x=0.0, rotate_y=0.0, rotate_z=0.0):
        from src.utils.save_utils import save_scene_ply, save_gs_ply
        
        output_dir = folder_paths.get_output_directory()
        filename = self._sanitize_filename(filename)
        saved_files = []
        save_pointcloud = False
        save_gaussians = True
        rotate_x = rotate_y = rotate_z = 0.0
        
        R = None
        if rotate_x != 0 or rotate_y != 0 or rotate_z != 0:
            R = self._rotation_matrix(rotate_x, rotate_y, rotate_z).cpu()
            print(f"🔄 Applying rotation: X={rotate_x}°, Y={rotate_y}°, Z={rotate_z}°")
        
        if save_gaussians:
            # 1. Try Native Extraction (Primary)
            splats = ply_data.get("splats")
            native_success = False
            apply_scale_filter = not bool(ply_data.get("skip_scale_filter"))
            
            if splats is not None and isinstance(splats, dict):
                try:
                    # Native extraction matching infer.py
                    def get_tensor(k, dim):
                         v = splats.get(k)
                         if v is None: return None
                         if isinstance(v, list): v = v[0]
                         # if v.dim() > 2: v = v[0]  <-- REMOVED: broken for SH [N, 1, 3]
                         return v.reshape(-1, dim).detach().cpu().float()
                         
                    means = get_tensor("means", 3)
                    scales = get_tensor("scales", 3)
                    quats = get_tensor("quats", 4)
                    
                    print(f"🔍 [SavePLY] Inspecting Native Splats: Means={means.shape if means is not None else 'None'}")
                    
                    # Colors logic from infer.py
                    if "sh" in splats:
                        # Use SH if available (infer.py prefers SH)
                        colors = get_tensor("sh", 3) 
                    else:
                        colors = get_tensor("colors", 3)
                        
                    # FIX: Handle broadcasting if colors are global (1, C) but means are (N, 3)
                    if colors is not None and means is not None:
                        if colors.shape[0] == 1 and means.shape[0] > 1:
                            print(f"🎨 [SavePLY] Broadcasting global color {colors.shape} to {means.shape[0]} points")
                            colors = colors.repeat(means.shape[0], 1)
                         
                    opacities = get_tensor("opacities", 1).reshape(-1)
                    
                    if means is not None and scales is not None and quats is not None:
                         # Debug Stats
                         print(f"📊 [SavePLY Native] Stats:")
                         print(f"   - Means: {means.shape} [Min: {means.min():.3f}, Max: {means.max():.3f}]")
                         print(f"   - Scales: {scales.shape} [Min: {scales.min():.3f}, Max: {scales.max():.3f}, Mean: {scales.mean():.3f}]")
                         print(f"   - Opacity: {opacities.shape} [Min: {opacities.min():.3f}, Max: {opacities.max():.3f}]")

                         # Apply rotation to means AND quaternions
                         if R is not None:
                             means = (means @ R.T)
                             quats = self._rotate_quaternions(quats, R)
                         
                         gs_path = self._get_unique_path(output_dir, filename, "gaussians", "ply")
                         self._save_gs_ply_safe(
                             gs_path,
                             means,
                             scales,
                             quats,
                             colors,
                             opacities,
                             apply_scale_filter=apply_scale_filter,
                         )
                         saved_files.append(gs_path)
                         print(f"💾 [SavePLY] SUCCESS: Saved gaussians (Native): {os.path.basename(gs_path)} ({len(means)} pts)")
                         native_success = True
                except Exception as e:
                    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
                        raise RuntimeError(
                            "SavePLY failed: no space left on device while writing Gaussian PLY. "
                            "Free disk space or lower splat_upsample_max_points; fallback is disabled "
                            "for storage errors to avoid leaving a broken 0 KB PLY."
                        ) from e
                    print(f"⚠️ [SavePLY] Native extraction failed, falling back: {e}")
            
            # 2. Fallback to Helper (if splats missing or native failed)
            if not native_success:
                print("⚠️ [SavePLY] Using fallback extraction...")
                params = extract_splat_params(ply_data)
                if params:
                    means, scales, quats, colors, opacities = params
                    if R is not None:
                        means = (means.to(torch.float32) @ R.T).cpu()
                    
                    # FIX: Convert RGB to SH for correct color rendering in splat viewers
                    # Viewer: Color = 0.5 + 0.282 * SH
                    # Inverse: SH = (Color - 0.5) / 0.282
                    print("🎨 [SavePLY] Converting RGB to SH for Fallback Splats...")
                    SH_C0 = 0.28209479177387814
                    colors = (colors - 0.5) / SH_C0

                    gs_path = self._get_unique_path(output_dir, filename, "gaussians", "ply")
                    self._save_gs_ply_safe(
                        gs_path,
                        means,
                        scales,
                        quats,
                        colors,
                        opacities,
                        apply_scale_filter=apply_scale_filter,
                    )
                    saved_files.append(gs_path)
                    print(f"💾 [SavePLY] SUCCESS: Saved gaussians (Fallback): {os.path.basename(gs_path)} ({len(means)} pts)")
                else:
                    print("⚠️ [SavePLY] No splat data available after extraction")

        if save_pointcloud and ply_data.get("pts3d") is not None:
            pts3d = ply_data["pts3d"]
            images = ply_data["images"]
            means = pts3d[0].view(-1, 3).cpu()
            S = pts3d.shape[1]
            colors = images[0, :S].permute(0, 2, 3, 1).reshape(-1, 3).cpu()
            
            if R is not None:
                means = (means.to(torch.float32) @ R.T).cpu()
            
            # Use unique path generation to avoid overwriting
            pc_path = self._get_unique_path(output_dir, filename, "pointcloud", "ply")
            print(f"⏳ [SavePLY] Saving PointCloud PLY to: {pc_path}")
            
            # CRITICAL FIX for Pale Colors:
            # save_gs_ply expects SH coefficients. Viewer does: Color = 0.5 + 0.282 * SH
            # We have RGB [0..1]. So we need to Inverse Transform: SH = (RGB - 0.5) / 0.282
            # Otherwise 0->0.5 (Gray), 1->0.78 (Light Gray).
            SH_C0 = 0.28209479177387814
            colors_sh = (colors - 0.5) / SH_C0
            
            # We use save_gs_ply even for fallback points to get splat visualization
            # Creating dummy scales/quats/opacity for the point cloud
            from src.utils.save_utils import save_gs_ply
            
            N = len(means)
            dummy_scales = torch.ones(N, 3) * -4.6 # exp(-4.6) ~ 0.01
            dummy_quats = torch.zeros(N, 4)
            dummy_quats[:, 0] = 1.0
            dummy_opacities = torch.ones(N) * 100.0 # Opaque
            
            self._save_gs_ply_safe(pc_path, means, dummy_scales, dummy_quats, colors_sh, dummy_opacities)
            # save_scene_ply(pc_path, means, colors) # OLD method (simple points)
            
            saved_files.append(pc_path)
            print(f"💾 [SavePLY] SUCCESS: Saved pointcloud: {os.path.basename(pc_path)} ({len(means)} pts)")
        
        if not saved_files:
            return ("",)
            
        # Prioritize returning the Gaussian file for preview
        for f in saved_files:
            if "_gaussians" in f:
                return (f,)
                
        return (saved_files[0],)
class VNCCS_BackgroundPreview:
    """
    Preview Gaussian Splatting PLY files with interactive gsplat.js viewer.
    
    Displays 3D Gaussian Splats in an interactive WebGL viewer with orbit controls.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file"
                }),
                "camera_poses": ("TENSOR", {
                    "tooltip": "Optional: camera poses tensor from WorldMirror V2. Used to initialize the viewer camera."
                }),
                "camera_intrinsics": ("TENSOR", {
                    "tooltip": "Optional: camera intrinsics tensor from WorldMirror V2. Used for viewer FOV."
                }),
                "coordinate_basis": (["auto", "worldmirror", "hyworld2_worldgen"], {
                    "default": "auto",
                    "tooltip": "Coordinate basis of the PLY/camera tensors. Use hyworld2_worldgen for official HYWorld2 trainer outputs."
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("video_path", "ply_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "VNCCS/3D"
    OUTPUT_IS_LIST = (False, False)
    
    @classmethod
    def IS_CHANGED(cls, ply_path=None, coordinate_basis="auto", **kwargs):
        coordinate_basis = cls._normalize_coordinate_basis(coordinate_basis)
        state = [str(ply_path or ""), coordinate_basis]
        if ply_path:
            try:
                stat = os.stat(ply_path)
                state.append(f"{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                state.append("missing")
        return "|".join(state)

    @staticmethod
    def _normalize_coordinate_basis(coordinate_basis):
        value = str(coordinate_basis or "auto").strip()
        if not value:
            return "auto"
        if value not in {"auto", "worldmirror", "hyworld2_worldgen"}:
            print(f"[VNCCS_BackgroundPreview] Unknown coordinate_basis={coordinate_basis!r}; falling back to auto")
            return "auto"
        return value

    def _tensor_to_list(self, value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().float()
            return value.tolist()
        if isinstance(value, np.ndarray):
            return value.astype(np.float32).tolist()
        if isinstance(value, (list, tuple)):
            return value
        return None

    def _camera_intrinsics_to_preview(self, camera_intrinsics):
        if camera_intrinsics is None or not hasattr(camera_intrinsics, "detach"):
            return None
        intrs = camera_intrinsics.detach().cpu().float()
        if intrs.dim() == 4:
            intrs = intrs[0]
        if intrs.dim() == 3:
            intrs = intrs[0]
        if intrs.shape[-2:] != (3, 3):
            return None
        return intrs.tolist()

    def _infer_coordinate_basis(self, ply_path, coordinate_basis="auto"):
        requested = self._normalize_coordinate_basis(coordinate_basis)
        print(f"[HYWorld2 BasisDiag] Preview requested_basis={requested}, ply={ply_path}")
        if requested != "auto":
            print(f"[HYWorld2 BasisDiag] Preview resolved_basis={requested} (explicit)")
            return requested
        try:
            path = Path(ply_path)
            candidates = [
                path.parent.parent / "train_command.json",
                path.parent.parent / "train_config.json",
                path.parent / "train_command.json",
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                with open(candidate, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                basis = data.get("ply_basis") or data.get("camera_pose_basis")
                print(f"[HYWorld2 BasisDiag] Preview metadata={candidate}, declared_basis={basis!r}")
                if isinstance(basis, str):
                    basis = basis.lower()
                    if "hyworld2" in basis:
                        print("[HYWorld2 BasisDiag] Preview resolved_basis=hyworld2_worldgen (metadata)")
                        return "hyworld2_worldgen"
                    if "worldmirror" in basis:
                        print("[HYWorld2 BasisDiag] Preview resolved_basis=worldmirror (metadata)")
                        return "worldmirror"
            if (path.parent / "trainer_cameras.json").exists() or path.name.startswith("point_cloud_"):
                print("[HYWorld2 BasisDiag] Preview resolved_basis=hyworld2_worldgen (filename fallback)")
                return "hyworld2_worldgen"
        except Exception as exc:
            print(f"[VNCCS_BackgroundPreview] coordinate_basis auto-detect skipped: {type(exc).__name__}: {exc}")
        return "worldmirror"

    def _camera_poses_to_preview_basis(self, poses, coordinate_basis):
        if coordinate_basis != "hyworld2_worldgen":
            return poses
        basis = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=poses.dtype,
            device=poses.device,
        )
        return basis.unsqueeze(0) @ poses

    def _camera_poses_to_preview_extrinsics(self, camera_poses, coordinate_basis="worldmirror"):
        if camera_poses is None or not hasattr(camera_poses, "detach"):
            return None
        poses = camera_poses.detach().cpu().float()
        if poses.dim() == 4:
            poses = poses[0]
        if poses.dim() == 2:
            poses = poses.unsqueeze(0)
        if poses.dim() != 3 or poses.shape[-2] not in (3, 4) or poses.shape[-1] != 4:
            return None
        if poses.shape[0] == 0:
            print("[VNCCS_BackgroundPreview][diag] camera_poses is empty; skipping extrinsics.")
            return None

        if poses.shape[-2] == 3:
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=poses.dtype).view(1, 1, 4)
            poses = torch.cat([poses, bottom.repeat(poses.shape[0], 1, 1)], dim=1)
        poses = self._camera_poses_to_preview_basis(poses, coordinate_basis)

        c2w = poses[0].clone()
        centers = poses[:, :3, 3]
        c2w[:3, 3] = centers.mean(dim=0)

        # WorldMirror/Panorama nodes output c2w. The browser viewer receives
        # w2c-style extrinsics and reconstructs camera center as -R^T t.
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        w2c = torch.eye(4, dtype=c2w.dtype)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -(R.T @ t)
        print(f"[HYWorld2 BasisDiag] Preview camera_coordinate_basis={coordinate_basis}")
        print(f"[HYWorld2 BasisDiag] Preview representative_c2w=\n{c2w.numpy()}")
        print(f"[HYWorld2 BasisDiag] Preview emitted_w2c=\n{w2c.numpy()}")
        print(
            "[HYWorld2 BasisDiag] Preview camera axes: "
            f"right={c2w[:3, 0].tolist()}, down={c2w[:3, 1].tolist()}, "
            f"forward={c2w[:3, 2].tolist()}, center={c2w[:3, 3].tolist()}, "
            f"det={torch.det(c2w[:3, :3]).item():.6f}"
        )
        return w2c.tolist()

    def _view_info_for_path(self, path):
        output_dir = folder_paths.get_output_directory()
        temp_dir = folder_paths.get_temp_directory()

        path_norm = path.replace("\\", "/")
        output_dir_norm = output_dir.replace("\\", "/")
        temp_dir_norm = temp_dir.replace("\\", "/")

        if path_norm.startswith(output_dir_norm):
            rel_path = os.path.relpath(path, output_dir).replace("\\", "/")
            file_type = "output"
        elif path_norm.startswith(temp_dir_norm):
            rel_path = os.path.relpath(path, temp_dir).replace("\\", "/")
            file_type = "temp"
        else:
            rel_path = os.path.basename(path)
            file_type = "output"

        return {
            "rel_path": rel_path,
            "subfolder": os.path.dirname(rel_path).replace("\\", "/"),
            "filename": os.path.basename(rel_path),
            "type": file_type,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
        }

    def preview(
        self,
        ply_path=None,
        camera_poses=None,
        camera_intrinsics=None,
        coordinate_basis="auto",
        **kwargs,
    ):
        """Prepare PLY file for gsplat.js preview."""
        coordinate_basis = self._normalize_coordinate_basis(coordinate_basis)
        import glob

        # If no path provided, we can't preview
        if not ply_path:
            return {"ui": {}, "result": ("", "")}
        
        # Validate ply_path
        if not os.path.exists(ply_path):
            print(f"[VNCCS_BackgroundPreview] PLY file not found: {ply_path}")
            return {"ui": {"error": [f"File not found: {ply_path}"]}, "result": ("", "")}

        output_dir = folder_paths.get_output_directory()
        ply_info = self._view_info_for_path(ply_path)
        resolved_coordinate_basis = self._infer_coordinate_basis(ply_path, coordinate_basis)
        camera_coordinate_basis = resolved_coordinate_basis

        try:
            is_gaussian_ply = _ply_header_has_gaussian_fields(ply_path)
        except Exception:
            is_gaussian_ply = False

        if is_gaussian_ply:
            if resolved_coordinate_basis == "hyworld2_worldgen":
                preview_path = _convert_gaussian_ply_to_worldmirror_preview_basis(ply_path)
                frontend_coordinate_basis = "worldmirror"
            else:
                preview_path = ply_path
                frontend_coordinate_basis = resolved_coordinate_basis
            preview_info = ply_info
            preview_format = "ply"
            if preview_path != ply_path:
                preview_info = self._view_info_for_path(preview_path)
                print("[VNCCS_BackgroundPreview] Gaussian PLY detected; loading preview-basis PLY.")
            else:
                print("[VNCCS_BackgroundPreview] Gaussian PLY detected; loading original PLY directly.")
        else:
            frontend_coordinate_basis = resolved_coordinate_basis
            preview_path = os.path.splitext(ply_path)[0] + ".splat"
            if os.path.exists(preview_path):
                preview_info = self._view_info_for_path(preview_path)
                preview_format = "splat"
            else:
                preview_path = ply_path
                preview_info = ply_info
                preview_format = "ply"
        
        print(f"🔍 [VNCCS_BackgroundPreview] Preparing UI Data:")
        print(f"   - Full Path: {ply_path}")
        print(f"   - Filename: {ply_info['filename']}")
        print(f"   - Subfolder: {ply_info['subfolder']}")
        print(f"   - Type: {ply_info['type']}")
        print(f"   - Size: {ply_info['size_mb']:.2f} MB")
        if preview_path != ply_path:
            print(f"   - Preview cache: {preview_info['filename']} ({preview_info['size_mb']:.2f} MB)")
        print(f"   - Resolved coordinate basis: {resolved_coordinate_basis}")
        if frontend_coordinate_basis != resolved_coordinate_basis:
            print(f"   - Frontend coordinate basis: {frontend_coordinate_basis}")
        if hasattr(camera_poses, "shape"):
            print(f"   - camera_poses shape: {tuple(camera_poses.shape)}")
        else:
            print("   - camera_poses shape: None")
        if hasattr(camera_intrinsics, "shape"):
            print(f"   - camera_intrinsics shape: {tuple(camera_intrinsics.shape)}")
        else:
            print("   - camera_intrinsics shape: None")
        _log_preview_ply_diagnostics(ply_path, "source")
        if preview_path != ply_path and preview_format == "splat":
            _log_preview_splat_diagnostics(preview_path, "preview")
        elif preview_path != ply_path and preview_format == "ply":
            _log_preview_ply_diagnostics(preview_path, "preview")

        # Find latest recorded video from the viewer, if one was exported.
        video_path = ""
        try:
            pattern = os.path.join(output_dir, "gaussian-recording-*.mp4")
            video_files = glob.glob(pattern)
            if video_files:
                video_files.sort(key=os.path.getmtime, reverse=True)
                video_path = os.path.abspath(video_files[0])
                print(f"   - Found video: {os.path.basename(video_path)}")
        except Exception as e:
            print(f"   ⚠️ Error finding video: {e}")
            
        ui_data = {
            "filename": [ply_info["filename"]],
            "subfolder": [ply_info["subfolder"]],
            "type": [ply_info["type"]],
            "ply_path": [ply_info["rel_path"]],
            "file_size_mb": [ply_info["size_mb"]],
            "preview_filename": [preview_info["filename"]],
            "preview_subfolder": [preview_info["subfolder"]],
            "preview_type": [preview_info["type"]],
            "preview_path": [preview_info["rel_path"]],
            "preview_file_size_mb": [preview_info["size_mb"]],
            "preview_format": [preview_format],
            "coordinate_basis": [frontend_coordinate_basis],
            "basis_diagnostic_version": ["basis-diag-v6-fullscreen-preserve-20260713"],
        }
        
        # Add camera parameters if provided
        extrinsics = self._camera_poses_to_preview_extrinsics(camera_poses, camera_coordinate_basis)
        # Do not forward WorldMirror intrinsics to the browser viewer by default:
        # viewer_gaussian.html switches into native-resolution rendering whenever
        # intrinsics are present, which is much heavier for multi-million splats.
        if extrinsics is not None:
            ui_data["extrinsics"] = [extrinsics]
        
        print(f"✅ [VNCCS_BackgroundPreview] UI data ready. Returning to frontend.")
        return {"ui": ui_data, "result": (video_path, ply_path)}




class VNCCS_LoadPLYFile:
    """Load a PLY path and optional camera JSON for preview/testing nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to a .ply file.",
                }),
            },
            "optional": {
                "camera_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional camera JSON. If empty, tries trainer_cameras.json near the PLY.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("ply_path", "camera_poses", "camera_intrinsics", "info")
    FUNCTION = "load"
    CATEGORY = "VNCCS/3D"

    @classmethod
    def IS_CHANGED(cls, ply_path, camera_json="", **kwargs):
        values = []
        for raw_path in (ply_path, camera_json):
            value = str(raw_path or "").strip().strip('"')
            path = Path(value) if value else None
            if path and path.exists():
                stat = path.stat()
                values.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
            else:
                values.append(value)
        return "|".join(values)

    def load(self, ply_path, camera_json=""):
        ply_path = str(ply_path or "").strip().strip('"')
        if not ply_path:
            raise ValueError("Load PLY File requires ply_path.")
        path = Path(ply_path)
        if not path.exists():
            raise ValueError(f"PLY file not found: {path}")
        if path.suffix.lower() != ".ply":
            raise ValueError(f"Load PLY File expects a .ply file: {path}")

        camera_path = Path(str(camera_json).strip().strip('"')) if camera_json else None
        if not camera_path or not camera_path.exists():
            camera_path = _find_preview_camera_json_for_ply(path)

        if camera_path and camera_path.exists():
            poses, intrs = _load_preview_camera_tensors_from_json(camera_path)
        else:
            poses, intrs = torch.empty((0, 4, 4)), torch.empty((0, 3, 3))

        info = {
            "ply_path": str(path),
            "camera_json": str(camera_path) if camera_path else "",
            "camera_count": int(poses.shape[0]) if poses.ndim >= 1 else 0,
        }
        print(f"[Load PLY File] PLY: {path}")
        print(f"[Load PLY File] Camera JSON: {info['camera_json'] or 'not found'}")
        print(f"[Load PLY File] Cameras: {info['camera_count']}")
        return (str(path), poses, intrs, json.dumps(info, indent=2))


class VNCCS_DecomposePLYData:
    """
    Extract individual components from PLY_DATA.
    
    Useful for accessing camera poses, intrinsics, point clouds, and splat parameters
    without going through the preview node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
            },
            "optional": {
                "view_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "TENSOR")
    RETURN_NAMES = ("camera_pose", "camera_intrinsics", "pts3d", "pts3d_conf")
    FUNCTION = "decompose"
    CATEGORY = "VNCCS/3D"
    
    def decompose(self, ply_data, view_index=0):
        """Extract camera and point cloud data from PLY data."""
        
        camera_pose = None
        camera_intrs = None
        pts3d = None
        pts3d_conf = None
        
        # Extract camera pose for specified view
        if ply_data.get("camera_poses") is not None:
            poses = ply_data["camera_poses"]
            # Shape: [B, S, 4, 4]
            S = poses.shape[1]
            idx = min(view_index, S - 1)
            camera_pose = poses[0, idx].cpu().float()  # [4, 4]
            print(f"[VNCCS_DecomposePLYData] Extracted camera pose for view {idx}")
        
        # Extract camera intrinsics for specified view
        if ply_data.get("camera_intrs") is not None:
            intrs = ply_data["camera_intrs"]
            # Shape: [B, S, 3, 3]
            S = intrs.shape[1]
            idx = min(view_index, S - 1)
            camera_intrs = intrs[0, idx].cpu().float()  # [3, 3]
            print(f"[VNCCS_DecomposePLYData] Extracted camera intrinsics for view {idx}")
        
        # Extract 3D points for specified view
        if ply_data.get("pts3d") is not None:
            pts = ply_data["pts3d"]
            # Shape: [B, S, H, W, 3]
            S = pts.shape[1]
            idx = min(view_index, S - 1)
            pts3d = pts[0, idx].cpu().float()  # [H, W, 3]
            print(f"[VNCCS_DecomposePLYData] Extracted pts3d for view {idx}")
        
        # Extract 3D point confidence for specified view
        if ply_data.get("pts3d_conf") is not None:
            conf = ply_data["pts3d_conf"]
            # Shape: [B, S, H, W]
            S = conf.shape[1]
            idx = min(view_index, S - 1)
            pts3d_conf = conf[0, idx].cpu().float()  # [H, W]
            print(f"[VNCCS_DecomposePLYData] Extracted pts3d_conf for view {idx}")
        
        return (camera_pose, camera_intrs, pts3d, pts3d_conf)


class VNCCS_PLYSceneRenderer:
    """
    Render multiple views from PLY/Gaussian splat for scene restoration.
    
    Loads Gaussian data from a PLY file (same format as the preview widget uses).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {"forceInput": True, "tooltip": "Path to Gaussian PLY file from SavePLY node"}),
            },
            "optional": {
                "coverage_mode": (["minimal", "balanced", "ideal", "testing", "camera_control"], {"default": "balanced"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "fov": ("FLOAT", {"default": 90.0, "min": 30.0, "max": 120.0, "step": 5.0}),
                "use_gsplat": ("BOOLEAN", {"default": True, "tooltip": "Use official gsplat library for faster/better rendering if available."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("views", "camera_poses", "camera_intrinsics")
    FUNCTION = "render_views"
    CATEGORY = "VNCCS/3D"
    OUTPUT_IS_LIST = (True, False, False)
    
    def _load_gaussian_ply(self, ply_path):
        """Load Gaussian parameters from PLY file (same format as viewer uses)."""
        from plyfile import PlyData
        
        print(f"🔍 [PLYSceneRenderer] Loading PLY: {ply_path}")
        ply = PlyData.read(ply_path)
        vertex = ply['vertex']
        
        N = len(vertex.data)
        print(f"   - Loaded {N:,} vertices")
        
        # Extract positions
        means = np.column_stack([vertex['x'], vertex['y'], vertex['z']]).astype(np.float32)
        
        # Extract scales (stored as log in PLY, need to exp)
        if 'scale_0' in vertex.data.dtype.names:
            scales = np.column_stack([
                np.exp(vertex['scale_0']),
                np.exp(vertex['scale_1']),
                np.exp(vertex['scale_2'])
            ]).astype(np.float32)
            print(f"   - Scales: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")
        else:
            # Fallback for simple point clouds
            scales = np.ones((N, 3), dtype=np.float32) * 0.01
            print("   - ⚠️ No scales found, using default 0.01")
        
        # Extract rotations
        if 'rot_0' in vertex.data.dtype.names:
            quats = np.column_stack([
                vertex['rot_0'], vertex['rot_1'], 
                vertex['rot_2'], vertex['rot_3']
            ]).astype(np.float32)
        else:
            quats = np.zeros((N, 4), dtype=np.float32)
            quats[:, 0] = 1.0  # Identity rotation
        
        # Extract colors (SH DC -> RGB)
        if 'f_dc_0' in vertex.data.dtype.names:
            sh_dc = np.column_stack([
                vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
            ]).astype(np.float32)
            # Convert SH DC to RGB: RGB = SH * 0.28209 + 0.5
            colors = sh_dc * 0.28209479177387814 + 0.5
            colors = np.clip(colors, 0.0, 1.0)
            print(f"   - Colors (from SH): min={colors.min():.3f}, max={colors.max():.3f}")
        elif 'red' in vertex.data.dtype.names:
            # Simple RGB point cloud
            colors = np.column_stack([
                vertex['red'], vertex['green'], vertex['blue']
            ]).astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            print(f"   - Colors (RGB): min={colors.min():.3f}, max={colors.max():.3f}")
        else:
            colors = np.ones((N, 3), dtype=np.float32) * 0.5
            print("   - ⚠️ No colors found, using gray")
        
        # Extract opacities (stored as logit in PLY, need sigmoid)
        if 'opacity' in vertex.data.dtype.names:
            opacity_raw = vertex['opacity'].astype(np.float32)
            # If values are outside [0, 1], they are logits
            if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
                opacities = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid
                print(f"   - Opacities (from logit): min={opacities.min():.3f}, max={opacities.max():.3f}")
            else:
                opacities = np.clip(opacity_raw, 0.0, 1.0)
                print(f"   - Opacities (direct): min={opacities.min():.3f}, max={opacities.max():.3f}")
        else:
            opacities = np.ones(N, dtype=np.float32) * 0.9
            print("   - ⚠️ No opacities found, using 0.9")
        
        return means, scales, quats, colors, opacities
    
    def _build_intrinsics(self, width, height, fov_deg, device):
        """Build intrinsic matrix from FOV and image size."""
        fov_rad = math.radians(fov_deg)
        focal = width / (2.0 * math.tan(fov_rad / 2.0))
        
        K = torch.zeros(3, 3, device=device)
        K[0, 0] = focal  # fx
        K[1, 1] = focal  # fy
        K[0, 2] = width / 2.0  # cx
        K[1, 2] = height / 2.0  # cy
        K[2, 2] = 1.0
        return K
    
    def _rotation_matrix_y(self, angle_deg, device):
        """Create rotation matrix around Y axis."""
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=torch.float32, device=device)
    
    def _rotation_matrix_x(self, angle_deg, device):
        """Create rotation matrix around X axis."""
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=torch.float32, device=device)
    
    def _apply_rotation_to_pose(self, pose, rot_matrix):
        """Apply rotation to camera pose (c2w matrix)."""
        new_pose = pose.clone()
        # Rotate the rotation part of the pose
        new_pose[:3, :3] = pose[:3, :3] @ rot_matrix.T
        return new_pose
    
    def _translate_pose(self, pose, translation):
        """Translate camera position in world space."""
        new_pose = pose.clone()
        new_pose[:3, 3] = new_pose[:3, 3] + translation
        return new_pose
    
    def _make_look_at_pose(self, position, target, device):
        """Create a camera pose looking from position toward target."""
        forward = target - position
        forward = forward / (torch.norm(forward) + 1e-8)
        
        # Use Y-down convention (common in vision)
        up = torch.tensor([0.0, -1.0, 0.0], device=device)
        right = torch.linalg.cross(forward, up)
        right_norm = torch.norm(right)
        
        # Handle case when forward is parallel to up
        if right_norm < 1e-6:
            up = torch.tensor([0.0, 0.0, 1.0], device=device)
            right = torch.linalg.cross(forward, up)
        
        right = right / (torch.norm(right) + 1e-8)
        up = torch.linalg.cross(right, forward)
        
        pose = torch.eye(4, device=device)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = forward # Fix: OpenCV style expects +Z forward
        pose[:3, 3] = position
        return pose
    
    def _generate_corner_positions(self, center, size, base_height, shrink=0.35, device=None):
        """Generate 4 corner positions inside room bounds."""
        half_x = size[0] * shrink
        half_z = size[2] * shrink
        
        corners = [
            center + torch.tensor([half_x, 0, half_z], device=device),   # +X +Z
            center + torch.tensor([-half_x, 0, half_z], device=device),  # -X +Z
            center + torch.tensor([-half_x, 0, -half_z], device=device), # -X -Z
            center + torch.tensor([half_x, 0, -half_z], device=device),  # +X -Z
        ]
        
        # Set Y to base camera height
        for c in corners:
            c[1] = base_height
        
        return corners
    
    def _generate_edge_positions(self, center, size, base_height, shrink=0.35, device=None):
        """Generate 4 edge midpoint positions inside room bounds."""
        half_x = size[0] * shrink
        half_z = size[2] * shrink
        
        edges = [
            center + torch.tensor([half_x, 0, 0], device=device),   # +X edge
            center + torch.tensor([-half_x, 0, 0], device=device),  # -X edge
            center + torch.tensor([0, 0, half_z], device=device),   # +Z edge
            center + torch.tensor([0, 0, -half_z], device=device),  # -Z edge
        ]
        
        for e in edges:
            e[1] = base_height
        
        return edges
    
    def _generate_coverage_views(self, scene_bounds, coverage_mode, seed, device):
        """
        Generate camera views optimized for >90% scene coverage.
        
        Strategy:
        - minimal: 4 corners looking inward + 2 pitch variants = 6 views
        - balanced: corners + edges + pitch variants = 10 views
        - ideal: corners + edges + center + multiple pitches = 14 views
        - testing: ideal + 360 rotation from center (looking outward) = 26 views
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        views = []
        center = (scene_bounds["min"] + scene_bounds["max"]) / 2
        size = scene_bounds["max"] - scene_bounds["min"]
        base_height = center[1].item()  # Use center Y as camera height
        
        # Always include corner positions (best for triangulation)
        corners = self._generate_corner_positions(center, size, base_height, shrink=0.35, device=device)
        
        if coverage_mode == "minimal":
            # 4 corners looking at center (diagonal coverage)
            for pos in corners:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 2 corner views with pitch down (floor/objects) from opposite corners
            for pos in [corners[0], corners[2]]:
                floor_target = center.clone()
                floor_target[1] = scene_bounds["min"][1]  # Look at floor level
                pose = self._make_look_at_pose(pos, floor_target, device)
                views.append(pose)
                
        elif coverage_mode == "balanced":
            # 4 corners looking at center
            for pos in corners:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 4 edge positions
            edges = self._generate_edge_positions(center, size, base_height, shrink=0.35, device=device)
            for pos in edges:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 2 views looking down at floor from corners
            for pos in [corners[0], corners[2]]:
                floor_target = center.clone()
                floor_target[1] = scene_bounds["min"][1]
                pose = self._make_look_at_pose(pos, floor_target, device)
                views.append(pose)
                
        elif coverage_mode == "ideal" or coverage_mode == "testing":  # ideal logic applies to testing too
            # 4 corners looking at center
            for pos in corners:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # 4 edge positions
            edges = self._generate_edge_positions(center, size, base_height, shrink=0.35, device=device)
            for pos in edges:
                pose = self._make_look_at_pose(pos, center, device)
                views.append(pose)
            
            # Center view looking around (4 directions: +X, -X, +Z, -Z)
            center_h = center.clone()
            center_h[1] = base_height
            look_directions = [
                center_h + torch.tensor([1, 0, 0], device=device),
                center_h + torch.tensor([-1, 0, 0], device=device),
                center_h + torch.tensor([0, 0, 1], device=device),
                center_h + torch.tensor([0, 0, -1], device=device),
            ]
            for target in look_directions:
                pose = self._make_look_at_pose(center_h, target, device)
                views.append(pose)
            
            # 2 random high-coverage views
            for _ in range(2):
                rand_pos = corners[np.random.randint(0, 4)] + torch.randn(3, device=device) * 0.1
                rand_target = center + torch.randn(3, device=device) * 0.2
                pose = self._make_look_at_pose(rand_pos, rand_target, device)
                views.append(pose)
            
            # 4 views looking DOWN at floor (cover horizontal surfaces)
            for pos in corners:
                floor_target = center.clone()
                floor_target[1] = scene_bounds["min"][1]
                pose = self._make_look_at_pose(pos, floor_target, device)
                views.append(pose)
            
            # 2 views looking UP at ceiling (cover overhead)
            for pos in [corners[1], corners[3]]:
                ceiling_target = center.clone()
                ceiling_target[1] = scene_bounds["max"][1]
                pose = self._make_look_at_pose(pos, ceiling_target, device)
                views.append(pose)
        
        if coverage_mode == "testing":
             # "testing" mode EXTENDS "ideal" mode by adding a 360 spin
             # Similar to Equirect360ToViews but inside the room bounds
             
             # Calculate spin radius (slightly smaller than room bounds)
             radius = min(size[0], size[2]) * 0.3
             num_frames = 12 # 30 degree steps
             
             for i in range(num_frames):
                 angle = 2 * np.pi * i / num_frames
                 # Camera position is CENTER
                 cam_pos_center = center.clone()
                 cam_pos_center[1] = base_height
                 
                 # Look OUTWARD at a point on circle
                 # Note: in many 3D systems Z is forward/back, X is left/right. 
                 # Let's rotate in XZ plane.
                 target_x = center[0] + radius * np.cos(angle)
                 target_z = center[2] + radius * np.sin(angle)
                 target_pos = torch.tensor([target_x, base_height, target_z], device=device)
                 
                 # Camera at center, looking at target
                 pose = self._make_look_at_pose(cam_pos_center, target_pos, device)
                 views.append(pose)
        
        elif coverage_mode == "camera_control":
             # "camera_control" mode: 4 Cardinal Views (0, 90, 180, 270)
             # Designed for creating training datasets that match panorama unrolling.
             
             # Camera position is CENTER
             cam_pos_center = center.clone()
             cam_pos_center[1] = base_height
             
             # 4 Directions: Front (+Z), Right (+X), Back (-Z), Left (-X)
             # Matches standard panorama unwrapping:
             # Front (0°), Right (90°), Back (180°), Left (270°)
             
             targets = [
                 cam_pos_center + torch.tensor([0, 0, 1], device=device),  # Front (+Z)
                 cam_pos_center + torch.tensor([1, 0, 0], device=device),  # Right (+X)
                 cam_pos_center + torch.tensor([0, 0, -1], device=device), # Back (-Z)
                 cam_pos_center + torch.tensor([-1, 0, 0], device=device), # Left (-X)
             ]
             
             for target in targets:
                 pose = self._make_look_at_pose(cam_pos_center, target, device)
                 views.append(pose)

        return views
    
    def _get_scene_bounds(self, means):
        """Get scene bounding box from gaussian means."""
        bounds_min = means.min(axis=0)
        bounds_max = means.max(axis=0)
        return {"min": torch.from_numpy(bounds_min), "max": torch.from_numpy(bounds_max)}
    
    def render_views(self, ply_path, coverage_mode="balanced", width=1024, height=1024, fov=90.0, seed=0, use_gsplat=True):
        """Render multiple views from PLY file using ModernGL or gsplat renderer."""
        import time
        import os
        
        print(f"🎥 [PLYSceneRenderer] RENDER START")
        print(f"   - PLY File: {os.path.basename(ply_path)}")
        print(f"   - Coverage Mode: {coverage_mode}")
        print(f"   - Resolution: {width}x{height}, FOV: {fov}")
        print(f"   - Engine: {'gsplat' if use_gsplat and GSPLAT_AVAILABLE else ('FastPLY' if not use_gsplat else 'FastPLY (fallback)')}")
        
        if not os.path.exists(ply_path):
            raise ValueError(f"PLY file not found: {ply_path}")
        
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - Device: {device}")
        
        # Load Gaussian data from PLY file
        means_np, scales_np, quats_np, colors_np, opacities_np = self._load_gaussian_ply(ply_path)
        N = len(means_np)
        
        # Get scene bounds for view generation
        scene_bounds = self._get_scene_bounds(means_np)
        scene_bounds["min"] = scene_bounds["min"].to(device)
        scene_bounds["max"] = scene_bounds["max"].to(device)
        print(f"   - Scene bounds: {scene_bounds['min'].tolist()} to {scene_bounds['max'].tolist()}")
        
        # Build intrinsics
        K = self._build_intrinsics(width, height, fov, device)
        print(f"   - Intrinsics K:\n{K.cpu().numpy()}")
        
        # Generate camera poses using corner/edge strategy for optimal triangulation
        all_poses = self._generate_coverage_views(scene_bounds, coverage_mode, seed, device)
        print(f"   - Generated {len(all_poses)} views")
        
        # Stack poses and intrinsics
        poses_tensor = torch.stack(all_poses)  # [V, 4, 4]
        Ks_tensor = K.unsqueeze(0).expand(len(all_poses), -1, -1)  # [V, 3, 3]
        
        print(f"   Rendering {len(all_poses)} views at {width}x{height}...")
        
        rendered_images = []
        
        # Prepare Tensors on GPU
        print(f"   - Moving {N:,} points to GPU...")
        means_t = torch.from_numpy(means_np).to(device)
        colors_t = torch.from_numpy(colors_np).to(device)
        opacities_t = torch.from_numpy(opacities_np).to(device)
        scales_t = torch.from_numpy(scales_np).to(device)
        quats_t = torch.from_numpy(quats_np).to(device) # Needed for gsplat
        
        print(f"   📊 Gaussian Stats:")
        print(f"      - Points: {N:,}")
        print(f"      - Means:  avg={means_np.mean(axis=0)}, range=[{means_np.min(axis=0)}, {means_np.max(axis=0)}]")
        
        render_start = time.time()
        
        # --------------------------------------------------------------------
        # RENDER ENGINE 1: gsplat (Official Implementation)
        # --------------------------------------------------------------------
        if use_gsplat and GSPLAT_AVAILABLE:
            try:
                from gsplat import rasterization
                
                # gsplat expects view-matrices (world-to-camera) and specific conventions
                # c2w is camera-to-world. w2c = inverse(c2w)
                
                for i, pose in enumerate(all_poses):
                    # Compute World-to-Camera (Extrinsics)
                    w2c = torch.inverse(pose) # [4, 4]
                    
                    # gsplat expects [B, 4, 4] or single view
                    # We render one by one to save VRAM (just in case) or batched?
                    # Let's do one by one to be safe and consistent with loop structure
                    
                    # Prepare arguments for gsplat.rasterization
                    # Note: gsplat API varies by version (v0.1 vs v1.0)
                    # Assuming v1.0 convention: means, quats, scales, opacities, colors, viewmats, Ks, width, height
                    
                    # Check gsplat version signature roughly by try-catch or assumption
                    # We'll assume standard 1.0ish signature
                    
                    # Need to handle SH vs Color?
                    # colors_t is RGB [N, 3]. gsplat supports 'colors' arg if sh_degree=None?
                    # Actually, usually 'colors' is not a direct arg in rasterization if SHs are expected.
                    # But if we pass sh_degree=0, we can pass SHs (which are just RGB / 0.282...)
                    # Or maybe pre-computed colors?
                    
                    # Let's try simple call. If fails, user will see error.
                    # Actually, gsplat usually takes 'means', 'quats', 'scales', 'opacities', 'colors' (if provided?)
                    
                    # Wait, 'colors' are view-dependent. gsplat usually takes 'shs'.
                    # We have RGB colors. We can convert them to SH degree 0.
                    # SH_0 = RGB / 0.28209479177387814 - 0.5/0.282... ?
                    # Wait, formula earlier was RGB = SH * C0 + 0.5.
                    # So SH = (RGB - 0.5) / C0.
                    
                    # But we can also use 'colors' argument in some versions?
                    # Let's try to assume we can pass colors if we set sh_degree=None or something.
                    # Re-checking worldmirror usage...
                    # WorldMirror uses `gsplat.project_gaussians` then `gsplat.rasterize_gaussians`.
                    # Let's stick to simple Rasterizer wrapper from WorldMirror if accessible?
                    # No, we don't have access to the wrapper easily here (different context).
                    
                    # Let's fallback to FastPLYRenderer logic but powered by gsplat if I can't guess API?
                    # No, user wants gsplat specifically.
                    
                    # Let's use the simplest efficient method: batch processing if possible?
                    # No, let's keep loop.
                    
                    viewmat = w2c.unsqueeze(0) # [1, 4, 4]
                    K_in = K.unsqueeze(0) # [1, 3, 3]
                    
                    # Convert colors to SH (deg 0)
                    # RGB to SH0: (RGB - 0.5) / 0.28209479177387814
                    shs_0 = (colors_t - 0.5) / 0.28209479177387814
                    shs_view = shs_0.unsqueeze(1) # [N, 1, 3] (deg 0)
                    
                    # Call rasterization
                    # (means, quats, scales, opacities, colors, viewmats, Ks, width, height)
                    # WARNING: signature depends on version.
                    # v1.0.0+: rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, ...)
                    
                    render_colors, render_alphas, info = rasterization(
                        means=means_t,
                        quats=quats_t,
                        scales=scales_t,
                        opacities=opacities_t, # flatten?
                        colors=shs_view, # pass SHs here? Or is there a 'colors' arg?
                        # If we pass 'colors' it usually expects pre-computed colors.
                        # Let's try passing 'shs' if keyword arg? 
                        # Actually, looking at docs (mental check):
                        # rasterization(means, quats, scales, opacities, colors=None, shs=None, ...)
                        # If colors is None and shs is None, it fails.
                        # If shs provided, it computes colors.
                        shs=shs_view,
                        viewmats=viewmat,
                        Ks=K_in,
                        width=width,
                        height=height,
                        packed=False # usually False for simple batch
                    )
                    
                    # render_colors: [1, H, W, 3]
                    img_tensor = render_colors[0] # [H, W, 3]
                    
                    # Clamp
                    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                    
                    rendered_images.append(img_tensor.cpu().unsqueeze(0))
                    
                    if (i + 1) % 5 == 0:
                        print(f"   ✓ [gsplat] Rendered {i + 1}/{len(all_poses)} views")
            
            except Exception as e:
                print(f"❌ [VNCCS] gsplat rendering failed: {e}")
                print("   ⚠️ Falling back to FastPLYRenderer...")
                # Fallback logic below
                use_gsplat = False # Trigger fallback
        
        # --------------------------------------------------------------------
        # RENDER ENGINE 2: FastPLYRenderer (Fallback / Default)
        # --------------------------------------------------------------------
        if not use_gsplat or not GSPLAT_AVAILABLE:
            if use_gsplat:
                 print("   ⚠️ gsplat requested but not available/failed. Using FastPLYRenderer.")
            
            # Initialize Fast Renderer
            print(f"   - Initializing FastPLYRenderer...")
            renderer = FastPLYRenderer(device)
            
            for i, pose in enumerate(all_poses):
                # Render using Fast PyTorch Splatter
                img_tensor = renderer.render(
                    means=means_t,
                    colors=colors_t,
                    opacities=opacities_t,
                    scales=scales_t,
                    c2w=pose,
                    width=width,
                    height=height,
                    fov_deg=fov
                )
                
                # DEBUG: Check if image is not black
                mean_val = img_tensor.mean().item()
                if (i + 1) % 5 == 0 or i == 0:
                     print(f"      - View {i}: mean pixel value {mean_val:.2f}")
                     
                rendered_images.append(img_tensor.cpu().unsqueeze(0))
                
                if (i + 1) % 5 == 0:
                    print(f"   ✓ [FastPLY] Rendered {i + 1}/{len(all_poses)} views")
        
        # Cleanup large tensors before finishing
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        # Move camera data to CPU
        poses_out = poses_tensor.cpu().float()
        Ks_out = Ks_tensor.cpu().float()
        
        total_render_time = time.time() - render_start
        total_total_time = time.time() - start_time
        print(f"✅ [PLYSceneRenderer] FINISHED")
        print(f"   - Render time: {total_render_time:.2f}s ({total_render_time/len(rendered_images):.3f}s/view)")
        print(f"   - Total time:  {total_total_time:.2f}s")
        
        return (rendered_images, poses_out, Ks_out)


class VNCCS_SplatRefiner:
    """
    Post-inference Gaussian Splatting Refinement.
    Uses backpropagation to tune splat parameters against the source images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_splats": ("VNCCS_SPLAT",),
                "iterations": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 10}),
                "lr": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001}),
            },
            "optional": {
                "images": ("IMAGE",),
                "optimize_means": ("BOOLEAN", {"default": True}),
                "optimize_scales": ("BOOLEAN", {"default": True}),
                "optimize_opacities": ("BOOLEAN", {"default": True}),
                "optimize_colors": ("BOOLEAN", {"default": True}),
                "consensus_bonus": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Experimental: Penalize points that move too far from their consensus neighbors. Range 1.0-5.0 recommended for V2."}),
                "optimize_camera": ("BOOLEAN", {"default": True, "tooltip": "V2+: Allow minor camera pose adjustments to fix ghosting (doubled objects)."}),
                "geometry_source": (["gaussians", "direct_pts3d"], {"default": "gaussians", "tooltip": "gaussians: use positions from input splats. direct_pts3d: RE-SYNC to raw 3D head (fixes broken/ghosted objects if projection failed)."}),
            }
        }
    
    RETURN_TYPES = ("PLY_DATA", "TENSOR")
    RETURN_NAMES = ("ply_data", "camera_poses")
    FUNCTION = "refine"
    CATEGORY = "VNCCS/3D"
    
    def refine(self, raw_splats, images, iterations=100, lr=0.001, 
               optimize_means=True, optimize_scales=True, optimize_opacities=True, optimize_colors=True, 
               consensus_bonus=1.0, optimize_camera=True, geometry_source="gaussians"):
        import torch.optim as optim
        
        # 1. Prepare Data
        # raw_splats is the 'predictions' dict from WorldMirror
        splats = raw_splats.get("splats")
        if splats is None:
            raise ValueError("No splat data found in raw_splats input.")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Prepare Ground Truth Images
        # Priority: use 'images' stored inside raw_splats (already tiled/resized/aligned)
        # Fallback: use external 'images' input
        internal_imgs = raw_splats.get("images")
        
        if internal_imgs is not None:
            # internal_imgs is [1, V, 3, H, W]
            gt_images_raw = internal_imgs[0]
            print(f"   - Found internal tiled images: {gt_images_raw.shape}")
        elif images is not None:
             # ComfyUI images are [V, H, W, 3] -> [V, 3, H, W]
             gt_images_raw = images.permute(0, 3, 1, 2)
             print(f"   - Using external source images: {gt_images_raw.shape}")
        else:
            raise ValueError("No ground truth images provided (neither internal nor external).")
        
        # 2. Setup Parameters for Optimization
        # CRITICAL: Prepare ALL data inside inference_mode(False) to avoid "Inference tensor" errors
        with torch.inference_mode(False):
            # 2.1 Prepare Ground Truth with explicit clone to break inference bond
            gt_images = gt_images_raw.detach().clone().to(device).float()
            
            # CRITICAL FIX: Match batch size for HD/Ultra tiling
            # Note: viewmats might not be on GPU yet, using raw_splats key as proxy
            V_render = raw_splats["rendered_extrinsics"][0].shape[0]
            if gt_images.shape[0] != V_render:
                print(f"⚠️ [SplatRefiner] Batch mismatch! Ground Truth={gt_images.shape[0]}, Render={V_render}.")
                if V_render % gt_images.shape[0] == 0:
                    repeat_val = V_render // gt_images.shape[0]
                    gt_images = gt_images.repeat_interleave(repeat_val, dim=0)
                    print(f"   -> Broadcasted to {gt_images.shape}")
                else:
                    raise ValueError(f"Cannot match GT images {gt_images.shape[0]} to render views {V_render}.")
            
            B_views = gt_images.shape[0]
            # STABLE OPTIMIZATION: Convert to unbounded space
            # 1. Means (Positions) - reduced LR later
            means = splats["means"][0].detach().clone().float().to(device).requires_grad_(optimize_means)
            quats = splats["quats"][0].detach().clone().float().to(device) # Rotation remains fixed for stability
            
            # 2. Scales - Optimize in log-space to ensure they stay positive
            log_scales = torch.log(splats["scales"][0].detach().clone().float().to(device) + 1e-8)
            if optimize_scales:
                log_scales.requires_grad_(True)
                
            # 3. Opacities - Optimize in logit-space to ensure they stay in [0, 1]
            # WorldMirror opacities are already sigmoid-activated.
            opacities_raw = splats["opacities"][0].detach().clone().float().to(device)
            logit_opacities = torch.logit(opacities_raw.clamp(1e-6, 1.0 - 1e-6))
            if optimize_opacities:
                logit_opacities.requires_grad_(True)
                
            # 4. Colors (SH) - Boosted LR later
            sh = splats["sh"][0].detach().clone().float().to(device).requires_grad_(optimize_colors)
            
            # V3.2: GEOMETRIC RESCUE MODE
            # If user wants to re-sync to the Direct 3D branch, we swap the means before starting.
            if geometry_source == "direct_pts3d" and "pts3d" in raw_splats:
                # pts3d is [1, S, H, W, 3]. Flatten to match dense splats
                direct_means = raw_splats["pts3d"][0].detach().clone().reshape(-1, 3).to(device).float()
                if direct_means.shape[0] == means.shape[0]:
                    print(f"   🎯 [SplatRefiner] GEOMETRIC RESCUE: Re-syncing Gaussians to Direct 3D head.")
                    means = direct_means.requires_grad_(optimize_means)
                else:
                    print(f"   ⚠️ [SplatRefiner] Rescue failed: shape mismatch. Pruned splats not supported for re-sync.")
            
            # 3. Setup Camera Data (Rendered Views)
            viewmats = raw_splats["rendered_extrinsics"][0].detach().clone().to(device).float() # [V, 4, 4]
            if optimize_camera:
                viewmats.requires_grad_(True)
            
            Ks = raw_splats["rendered_intrinsics"][0].detach().clone().to(device).float()       # [V, 3, 3]
            H, W = gt_images.shape[2], gt_images.shape[3]
            
            # 3.2 Prepare Distortion Masks (if available from WorldMirror)
            distortion_masks = raw_splats.get("distortion_masks")
            if distortion_masks is not None:
                distortion_masks = distortion_masks[0].detach().clone().to(device).float() # [V, H, W]
                print(f"   - Active Distortion Correction: weighted loss enabled.")

            # 4. Optimization Loop with Parameter-Specific Learning Rates (V3 Force-Sync)
            param_groups = []
            if optimize_means:
                param_groups.append({"params": [means], "lr": lr * 0.5})      
            if optimize_scales:
                param_groups.append({"params": [log_scales], "lr": lr * 1.0})
            if optimize_opacities:
                param_groups.append({"params": [logit_opacities], "lr": lr * 1.0})
            if optimize_colors:
                param_groups.append({"params": [sh], "lr": lr * 1.0})         
            if optimize_camera:
                # BOOSTED LR: Camera poses are the primary source of ghosting in V3 deep-dive
                param_groups.append({"params": [viewmats], "lr": lr * 2.0})   
            
            if not param_groups:
                print("⚠️ [SplatRefiner] No parameters selected for optimization. Skipping.")
                return (raw_splats, raw_splats["camera_poses"])

            optimizer = optim.Adam(param_groups)
            
            # Determine actual renderer to use
            from src.models.models.rasterization import Rasterizer
            rasterizer = Rasterizer()
            
            print(f"🔥 [SplatRefiner] Marble Stable Refinement V2 Start: {iterations} iterations")
            print(f"   - Target: {B_views} views at {W}x{H}")
            print(f"   - Points: {len(means):,}")
            print(f"   - Features: SSIM+L1 Hybrid, Camera Opt={optimize_camera}, Strict Anchoring")
            
            # Helper for SSIM Loss (Distortion-Weighted V3)
            def ssim(img1, img2, window_size=11, size_average=True, mask=None):
                import torch.nn.functional as F
                mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
                mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
                mu1_sq = mu1.pow(2)
                mu2_sq = mu2.pow(2)
                mu1_mu2 = mu1 * mu2
                sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
                sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
                sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
                C1 = 0.01 ** 2
                C2 = 0.03 ** 2
                ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                
                if mask is not None:
                    # Weight map by distortion mask
                    if mask.dim() == 3: mask = mask.unsqueeze(1)
                    return (ssim_map * mask).sum() / (mask.sum() * ssim_map.shape[1] + 1e-8)
                
                return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

            # Epsilon-Anchor for positions (Relaxed for V2.1: 15cm radius)
            EPSILON_RADIUS = 0.15 
            original_means = means.clone().detach() 
            
            with torch.enable_grad():
                for i in range(iterations):
                    optimizer.zero_grad()
                    
                    # TRANSFORM PARAMETERS
                    curr_scales = torch.exp(log_scales)
                    curr_opacities = torch.sigmoid(logit_opacities)
                    
                    # Render
                    rendered_colors, _, _ = rasterizer.rasterize_batches(
                        [means], [quats], [curr_scales], [curr_opacities], [sh], 
                        viewmats.unsqueeze(0), Ks.unsqueeze(0), W, H
                    )
                    
                    # [V, 3, H, W]
                    pred = rendered_colors[0].permute(0, 3, 1, 2)
                    
                    # HYBRID LOSS: L1 + SSIM (Distortion-Weighted V3)
                    l1_diff = torch.abs(pred - gt_images)
                    if distortion_masks is not None:
                        mask_w = distortion_masks.unsqueeze(1)
                        l1_loss = (l1_diff * mask_w).sum() / (mask_w.sum() * 3 + 1e-8)
                        ssim_val = ssim(pred, gt_images, mask=distortion_masks)
                    else:
                        l1_loss = l1_diff.mean()
                        ssim_val = ssim(pred, gt_images)
                    
                    loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)
                    
                    # Strict Geometric Anchoring (L2 penalty if outside epsilon)
                    if optimize_means:
                        diff = means - original_means
                        dist = torch.norm(diff, dim=-1)
                        # Penalize points moving > EPSILON_RADIUS
                        penalty = torch.mean(torch.clamp(dist - EPSILON_RADIUS, min=0)**2)
                        loss += penalty * 10.0
                    
                    # Consensus Bonus (Local density/structure maintenance)
                    if consensus_bonus > 0:
                        # Re-using the distance-from-origin penalty but scaled by user
                        loss += torch.mean(torch.norm(means - original_means, dim=-1)) * consensus_bonus * 0.1
                    
                    # Optimization step
                    loss.backward()
                    optimizer.step()
                    
                    # POST-STEP CLAMPING
                    with torch.no_grad():
                        # Keep points within logical bounds (Relaxed for V2.1: 1.5m limit)
                        diff = means - original_means
                        dist = torch.norm(diff, dim=-1, keepdim=True)
                        clip_mask = (dist > 1.5).squeeze(-1) 
                        if clip_mask.any():
                            means[clip_mask] = original_means[clip_mask] + (diff[clip_mask] / dist[clip_mask]) * 1.5
                    
                    if (i+1) % 50 == 0 or i == 0:
                        print(f"   [Iter {i+1:4d}] Loss: {loss.item():.6f} (L1: {l1_loss.item():.4f}, SSIM: {ssim_val.item():.4f})")

            # FINAL CLAMPING AND ACTIVATION
            final_means = means.detach()
            final_scales = torch.exp(log_scales).detach()
            final_opacities = torch.sigmoid(logit_opacities).detach()
            final_sh = sh.detach()
            final_poses = viewmats.detach().cpu()
        
        # 5. Pack results back into a PLY-compatible structure
        # We mimic the WorldMirror output structure so SavePLY works
        new_splats = {
            "means": [final_means.cpu()],
            "quats": [quats.detach().cpu()],
            "scales": [final_scales.cpu()],
            "opacities": [final_opacities.cpu()],
            "sh": [final_sh.cpu()],
        }
        
        refined_ply_data = {
            "splats": new_splats,
            "images": images.detach().cpu().permute(0, 3, 1, 2).unsqueeze(0) # Store as [1, V, 3, H, W]
        }
        
        print("✅ [SplatRefiner] Refinement Complete.")
        
        return (refined_ply_data, final_poses)



class VNCCS_WorldMirror3D_Official:
    """
    Clean WorldMirror 3D Reconstruction — mirrors the official Tencent infer.py exactly.
    No conditioning, no tiling, no consensus. Standard 'crop' preprocessing.
    Only use_direct_points is kept as an enhancement option.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WORLDMIRROR_MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "target_size": ("INT", {"default": 518, "min": 252, "max": 1024, "step": 14}),
                "use_gsplat": ("BOOLEAN", {"default": True, "tooltip": "Enable Gaussian Splatting renderer. If disabled, falls back to Point Cloud only."}),
                "use_direct_points": ("BOOLEAN", {"default": False, "tooltip": "Use pts3d (globally consistent pointmap) for Gaussian positions instead of depth reprojection. May improve multi-view overlap alignment."}),
                "confidence_percentile": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "Filter bottom X% low-confidence points."}),
                "filter_edges": ("BOOLEAN", {"default": True, "tooltip": "Remove artifact points at depth discontinuities."}),
                "edge_depth_threshold": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("PLY_DATA", "IMAGE", "IMAGE", "TENSOR", "TENSOR", "VNCCS_SPLAT")
    RETURN_NAMES = ("ply_data", "depth_maps", "normal_maps", "camera_poses", "camera_intrinsics", "raw_splats")
    FUNCTION = "run_inference"
    CATEGORY = "VNCCS/3D"
    
    def run_inference(self, model, images, target_size=518, use_gsplat=True,
                      use_direct_points=False, confidence_percentile=10.0,
                      filter_edges=True, edge_depth_threshold=0.03):
        from torchvision import transforms
        
        # Ensure target_size is divisible by 14
        target_size = (target_size // 14) * 14
        
        worldmirror = model["model"]
        device = model["device"]
        
        # =====================================================================
        # Image Preprocessing — EXACT replica of official prepare_images_to_tensor
        # Strategy: "crop" — resize width to target_size, center-crop height
        # =====================================================================
        B, H_orig, W_orig, C = images.shape
        
        converter = transforms.ToTensor()
        patch_size = 14
        tensor_list = []
        
        for i in range(B):
            img = images[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Handle RGBA (blend with white background, like official)
            if pil_img.mode == "RGBA":
                white_bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
                pil_img = Image.alpha_composite(white_bg, pil_img)
            pil_img = pil_img.convert("RGB")
            
            orig_w, orig_h = pil_img.size
            
            # Official "crop" strategy: set width to target_size, proportional height
            new_w = target_size
            new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
            
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            tensor_img = converter(pil_img)  # [3, H, W] in [0, 1]
            
            # Center crop height if larger than target_size
            if new_h > target_size:
                crop_start = (new_h - target_size) // 2
                tensor_img = tensor_img[:, crop_start:crop_start + target_size, :]
            
            tensor_list.append(tensor_img)
        
        # Stack and batch — official format: [1, S, 3, H, W]
        imgs_tensor = torch.stack(tensor_list).unsqueeze(0)
        
        # =====================================================================
        # Device Management (simple — no offload complexity)
        # =====================================================================
        execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_device = next(worldmirror.parameters()).device
        
        if original_device != execution_device:
            worldmirror.to(execution_device)
        
        imgs_tensor = imgs_tensor.to(execution_device)
        
        B_batch, S, C_ch, H, W = imgs_tensor.shape
        print(f"🚀 [Official] Running WorldMirror inference on {S} images ({H}x{W})...")
        
        # =====================================================================
        # Inference — EXACT replica of official infer.py
        # cond_flags = [0, 0, 0] — NO conditioning
        # No stabilization, no filter_mask, no consensus
        # =====================================================================
        views = {"img": imgs_tensor}
        cond_flags = [0, 0, 0]  # Official default — no conditioning
        
        # Override GS state
        original_gs_state = getattr(worldmirror, "enable_gs", True)
        
        try:
            if use_gsplat and not GSPLAT_AVAILABLE:
                print("⚠️ gsplat requested but library not found. Falling back to Point Cloud.")
                worldmirror.enable_gs = False
            else:
                worldmirror.enable_gs = use_gsplat
            
            use_amp = execution_device.type == "cuda"
            amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float32
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    predictions = worldmirror(
                        views=views,
                        cond_flags=cond_flags,
                        use_direct_points=use_direct_points,
                    )
        finally:
            worldmirror.enable_gs = original_gs_state
            if original_device.type == "cpu":
                worldmirror.to("cpu")
                torch.cuda.empty_cache()
        
        print("✅ [Official] Inference complete")
        
        # =====================================================================
        # Post-Processing — matching official infer.py filter logic
        # =====================================================================
        S = predictions["depth"].shape[1]
        H = predictions["depth"].shape[2]
        W = predictions["depth"].shape[3]
        
        # Compute filter mask (confidence + edges)
        pts3d_conf_np = predictions["pts3d_conf"][0].detach().cpu().numpy()
        depth_preds_np = predictions["depth"][0].detach().cpu().numpy()
        normal_preds_np = predictions["normals"][0].detach().cpu().numpy()
        
        final_mask = create_filter_mask(
            pts3d_conf=pts3d_conf_np,
            depth_preds=depth_preds_np,
            normal_preds=normal_preds_np,
            sky_mask=None,
            distortion_mask=None,
            confidence_percentile=confidence_percentile,
            edge_normal_threshold=5.0,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=True,
            apply_edge_mask=filter_edges,
            apply_sky_mask=False,
        )
        
        final_mask_flat = torch.from_numpy(final_mask.reshape(-1)).to(execution_device)
        
        # Filter Point Cloud
        filtered_pts = None
        if "pts3d" in predictions:
            pts = predictions["pts3d"][0].reshape(-1, 3)
            filtered_pts = pts[final_mask_flat.to(pts.device)]
        
        # Build PLY data output
        ply_data = {
            "pts3d": predictions.get("pts3d"),
            "pts3d_filtered": filtered_pts,
            "pts3d_conf": predictions.get("pts3d_conf"),
            "splats": predictions.get("splats"),
            "images": imgs_tensor,
            "filter_mask": final_mask_flat,
            "camera_poses": predictions.get("camera_poses"),
            "camera_intrs": predictions.get("camera_intrs"),
        }
        
        # Depth maps
        depth_tensor = predictions.get("depth")
        if depth_tensor is not None:
            depth = depth_tensor[0]
            depth_min = depth.min()
            depth_max = depth.max()
            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            depth_rgb = depth_norm.repeat(1, 1, 1, 3)
            depth_out = depth_rgb.cpu().float()
        else:
            depth_out = torch.zeros(S, target_size, target_size, 3)
        
        # Normal maps
        normal_tensor = predictions.get("normals")
        if normal_tensor is not None:
            normals = normal_tensor[0]
            normals_out = ((normals + 1) / 2).cpu().float()
        else:
            normals_out = torch.zeros(S, target_size, target_size, 3)
        
        # Camera outputs
        camera_poses_out = predictions.get("camera_poses")
        camera_intrs_out = predictions.get("camera_intrs")
        if camera_poses_out is not None:
            camera_poses_out = camera_poses_out.cpu().float()
        if camera_intrs_out is not None:
            camera_intrs_out = camera_intrs_out.cpu().float()
        
        # Store images in predictions for downstream compatibility
        predictions["images"] = imgs_tensor
        predictions["raw_images"] = images
        
        return (ply_data, depth_out, normals_out, camera_poses_out, camera_intrs_out, predictions)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": VNCCS_LoadWorldMirrorModel,
    "VNCCS_WorldMirror3D": VNCCS_WorldMirror3D,
    "VNCCS_WorldMirror3D_Official": VNCCS_WorldMirror3D_Official,
    "VNCCS_Equirect360ToViews": VNCCS_Equirect360ToViews,
    "VNCCS_SavePLY": VNCCS_SavePLY,
    "VNCCS_LoadPLYFile": VNCCS_LoadPLYFile,
    "VNCCS_BackgroundPreview": VNCCS_BackgroundPreview,
    "VNCCS_DecomposePLYData": VNCCS_DecomposePLYData,
    "VNCCS_PLYSceneRenderer": VNCCS_PLYSceneRenderer,
    "VNCCS_SplatRefiner": VNCCS_SplatRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": "🌐 Load WorldMirror Model",
    "VNCCS_WorldMirror3D": "🏔️ WorldMirror 3D Reconstruction",
    "VNCCS_WorldMirror3D_Official": "🔬 WorldMirror 3D (Official)",
    "VNCCS_Equirect360ToViews": "🔄 360° Panorama to Views",
    "VNCCS_SavePLY": "💾 Save PLY File",
    "VNCCS_BackgroundPreview": "👁️ Background Preview",
    "VNCCS_DecomposePLYData": "📦 Decompose PLY Data",
    "VNCCS_PLYSceneRenderer": "🎥 PLY Scene Renderer",
    "VNCCS_SplatRefiner": "🔥 Splat Refiner (Backprop)",
}

NODE_CATEGORY_MAPPINGS = {
    "VNCCS_LoadWorldMirrorModel": "VNCCS/3D",
    "VNCCS_WorldMirror3D": "VNCCS/3D",
    "VNCCS_WorldMirror3D_Official": "VNCCS/3D",
    "VNCCS_Equirect360ToViews": "VNCCS/3D",
    "VNCCS_SavePLY": "VNCCS/3D",
    "VNCCS_BackgroundPreview": "VNCCS/3D",
    "VNCCS_DecomposePLYData": "VNCCS/3D",
    "VNCCS_PLYSceneRenderer": "VNCCS/3D",
    "VNCCS_SplatRefiner": "VNCCS/3D",
}

NODE_DISPLAY_NAME_MAPPINGS["VNCCS_LoadPLYFile"] = "Load PLY File"
NODE_CATEGORY_MAPPINGS["VNCCS_LoadPLYFile"] = "VNCCS/3D"
