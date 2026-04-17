# WorldStereo ComfyUI Nodes — Design Spec
Date: 2026-04-17

## Overview

Integrate FuchengSu/WorldStereo into the ComfyUI_HYWorld2 project as 3 new nodes. WorldStereo generates camera-controlled multi-view video from a single image using the WAN DiT backbone. Its output (video frames + camera poses + intrinsics) feeds directly into the existing WorldMirror V2 node for 3D reconstruction.

## Repository Integration

Clone `https://github.com/FuchengSu/WorldStereo` into `worldstereo/` at the project root (same pattern as `worldmirror/`). Add `worldstereo/` to `sys.path` in the node file. No submodule — plain directory clone.

## Models Downloaded

All downloads via `huggingface_hub.snapshot_download()` into `folder_paths.models_dir` on first use.

| Repo | Local path | Size |
|---|---|---|
| `hanshanxude/WorldStereo` (subfolder per model_type) | `models/WorldStereo/<model_type>/` | 10.9–34.9 GB |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | `models/Wan2.1-I2V-14B-480P/` | ~40 GB |
| `Ruicheng/moge-2-vitl-normal` | `models/MoGe/` | ~1–2 GB |

The Wan2.1 base model (VAE, T5, CLIP) is shared across all WorldStereo model types — downloaded once.

## New Dependencies

Add to `requirements.txt`:
- `diffusers>=0.36.0`
- `transformers>=5.2.0`
- `bitsandbytes>=0.43.0` (fp8/fp4 quantization)
- `imageio[ffmpeg]`
- `kornia`
- `einops` (already present)
- `omegaconf` (already present)

MoGe installed via `pip install git+https://github.com/microsoft/MoGe.git` — add to install instructions.

## Nodes

### VNCCS_LoadWorldStereoModel

**Category:** `VNCCS/Video`

**Inputs (all optional):**
- `model_type`: enum `[worldstereo-camera, worldstereo-memory, worldstereo-memory-dmd]`, default `worldstereo-camera`
- `precision`: enum `[bf16, fp8, fp4]`, default `bf16`
- `offload_mode`: enum `[none, model_cpu_offload, sequential_cpu_offload]`, default `model_cpu_offload`
- `device`: enum `[cuda, cpu]`, default `cuda`

**Outputs:**
- `WORLDSTEREO_MODEL` — dict `{"pipeline": <pipeline>, "moge": <MoGeModel>, "device": str, "model_type": str}`

**Logic:**
1. Download WorldStereo transformer weights (subfolder = model_type)
2. Download Wan2.1-I2V-14B-480P-Diffusers (base_model components)
3. Download MoGe
4. Load `WorldStereo.from_pretrained()` — assembles pipeline internally from both repos
5. Apply precision:
   - `bf16`: `pipeline.transformer.to(torch.bfloat16)`, VAE to bf16, T5/CLIP stay fp32 on CPU
   - `fp8`: `BitsAndBytesConfig(load_in_8bit=True)` on transformer only
   - `fp4`: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4")` on transformer only
6. Apply offload_mode via diffusers pipeline methods:
   - `model_cpu_offload`: `pipeline.enable_model_cpu_offload()`
   - `sequential_cpu_offload`: `pipeline.enable_sequential_cpu_offload()`
7. Load MoGe separately, keep on CPU until inference

---

### VNCCS_CameraTrajectoryBuilder

**Category:** `VNCCS/Video`

**Inputs (all optional):**
- `preset`: enum `[circular, zoom_in, zoom_out, forward, custom]`, default `circular`
- `num_frames`: INT, default 25, min 4, max 81
- `radius`: FLOAT, default 1.0 (for circular/zoom)
- `speed`: FLOAT, default 1.0 (for forward)
- `elevation_deg`: FLOAT, default 0.0 (camera tilt for circular)
- `fov_deg`: FLOAT, default 70.0
- `image_width`: INT, default 768
- `image_height`: INT, default 480
- `custom_json`: STRING, default `""` (raw extrinsics JSON, used when preset=custom)

**Outputs:**
- `CAMERA_TRAJECTORY` — dict `{"extrinsics": Tensor[N,4,4], "intrinsics": Tensor[N,3,3]}`

**Preset implementations (all produce C2W 4×4 extrinsics):**
- `circular`: camera orbits around origin at given radius and elevation
- `zoom_in` / `zoom_out`: camera moves along Z axis toward/away from scene center
- `forward`: camera translates forward along view direction
- `custom`: parse extrinsics from `custom_json` string (list of 4×4 matrices)

Intrinsics matrix built from `fov_deg`, `image_width`, `image_height` — same for all frames.

---

### VNCCS_WorldStereoGenerate

**Category:** `VNCCS/Video`

**Required inputs:**
- `model`: `WORLDSTEREO_MODEL`
- `image`: `IMAGE` (single frame, `[1,H,W,3]`)
- `trajectory`: `CAMERA_TRAJECTORY`

**Optional inputs:**
- `num_inference_steps`: INT, default 20 (4 for memory-dmd)
- `guidance_scale`: FLOAT, default 5.0
- `seed`: INT, default -1 (random)
- `negative_prompt`: STRING, default `""`
- `output_fps`: INT, default 16 (for preview video only)

**Outputs:**
- `video_frames`: `IMAGE` `[N,H,W,3]` → connects to WorldMirror V2 `images`
- `camera_poses`: `TENSOR` `[N,4,4]` → connects to WorldMirror V2 `camera_poses`
- `camera_intrinsics`: `TENSOR` `[N,3,3]` → connects to WorldMirror V2 `camera_intrinsics`

**Inference flow:**
1. Preprocess input image to pipeline resolution (resize to multiple of 14, max side = image_width from trajectory)
2. Run MoGe on preprocessed image → depth map (on GPU during inference, back to CPU after)
3. Prepare `meta_info` dict from trajectory extrinsics, intrinsics, depth — matching `load_single_view_data()` format
4. Call `worldstereo.pipeline(**meta_info, negative_prompt=..., generator=..., output_type="pt")`
5. Extract frames: `output.frames[0].float()` → permute `[N,C,H,W]` → `[N,H,W,C]` → ComfyUI IMAGE
6. Return extrinsics and intrinsics from trajectory as-is (already `[N,4,4]` and `[N,3,3]`)
7. Clear GPU cache after inference

**num_inference_steps default:** auto-set to 4 if model_type is `worldstereo-memory-dmd`, else 20.

---

## File Changes

| File | Change |
|---|---|
| `worldstereo/` | New directory — cloned repo |
| `nodes/world_stereo.py` | New file — 3 nodes |
| `nodes/__init__.py` | Add WorldStereo import |
| `requirements.txt` | Add diffusers, bitsandbytes, imageio[ffmpeg], kornia |

## Downstream Workflow

```
[Single Image]
      ↓
VNCCS_LoadWorldStereoModel  +  VNCCS_CameraTrajectoryBuilder
      ↓                                  ↓
            VNCCS_WorldStereoGenerate
               ↓           ↓          ↓
         video_frames  cam_poses  cam_intrs
               ↓           ↓          ↓
            VNCCS_WorldMirrorV2_3D (existing node)
               ↓
         PLY / depth / normals / splats
```

## Constraints

- No torchao — quantization via bitsandbytes + diffusers BitsAndBytesConfig only
- worldstereo-camera is the only variant realistically usable on 16 GB VRAM with model_cpu_offload
- T5 encoder (fp32, ~20 GB) must remain on CPU via offloading — this is handled automatically by enable_model_cpu_offload()
- MoGe loaded on CPU by default, moved to GPU only during depth estimation call
