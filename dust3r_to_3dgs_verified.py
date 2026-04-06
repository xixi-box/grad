#!/usr/bin/env python3
"""
DUSt3R to 3DGS / COLMAP exporter with quality evaluation.

Main features:
1. Read input images, run DUSt3R inference and global alignment
2. Export COLMAP style cameras.bin / images.bin / points3D.bin / points3D.ply
3. Support two coordinate modes:
   - dust3r: images, intrinsics, depth unified to DUSt3R internal resolution
   - original: images, intrinsics, depth unified to original image resolution
4. Optional quality evaluation report for input validation

Usage:
    python dust3r_to_3dgs.py --input ./images --output ./data --coord_system dust3r --save_depth --evaluate
    python dust3r_to_3dgs.py --input ./images --output ./data --coord_system original --save_depth --evaluate --overwrite
"""
import os
import sys

# Auto-add dust3r path (supports local dev and cloud deployment)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_dust3r_path = os.path.join(_script_dir, "dust3r")
if os.path.exists(_dust3r_path) and _dust3r_path not in sys.path:
    sys.path.insert(0, _dust3r_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description="DUSt3R to 3DGS / COLMAP with evaluation")
    parser.add_argument("-i", "--input", required=True, help="Input image directory")
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(__file__), "data", "raw"),
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="DUSt3R model path"
    )
    parser.add_argument("--resolution", type=int, default=512, help="DUSt3R processing resolution")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--base_scale", type=float, default=0.5, help="Global alignment base_scale")
    parser.add_argument(
        "--coord_system",
        choices=["dust3r", "original"],
        default="dust3r",
        help="Export coordinate system: dust3r=DUSt3R internal, original=original image"
    )
    parser.add_argument("--save_depth", action="store_true", help="Save depth maps as .npy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Clear output directory if exists")
    parser.add_argument("--max_points", type=int, default=1_000_000, help="Max points in exported point cloud")

    # evaluation
    parser.add_argument("--evaluate", action="store_true", help="Run quality evaluation after export")
    parser.add_argument("--eval_sample_points", type=int, default=20000, help="Max sample points for evaluation")
    parser.add_argument("--eval_knn_k", type=int, default=16, help="kNN k for local neighborhood evaluation")
    parser.add_argument("--eval_depth_tol_rel", type=float, default=0.05, help="Depth consistency relative threshold")
    parser.add_argument("--eval_depth_tol_abs", type=float, default=0.01, help="Depth consistency absolute threshold")
    return parser.parse_args()


# ---------- basic utils ----------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def ensure_hwc_image(img):
    """Convert DUSt3R output image to HWC RGB np.ndarray."""
    img = tensor_to_numpy(img)
    if img.ndim != 3:
        raise ValueError(f"scene.imgs image dimension error, expected 3D, got {img.shape}")

    # CHW -> HWC
    if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]

    if img.shape[-1] != 3:
        raise ValueError(f"Cannot convert to HWC RGB image, current shape={img.shape}")
    return img


def array_to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = (img * 255.0).clip(0, 255)
    return img.astype(np.uint8)


def resize_image_to_wh(img: np.ndarray, target_wh):
    target_w, target_h = target_wh
    pil = Image.fromarray(array_to_uint8_rgb(img))
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(pil)


def resize_depth_to_wh(depth: np.ndarray, target_wh):
    target_w, target_h = target_wh
    depth = np.asarray(depth, dtype=np.float32)
    pil = Image.fromarray(depth, mode="F")
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def safe_inverse_pose(c2w, idx):
    try:
        return np.linalg.inv(c2w)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Pose matrix {idx} is not invertible: {e}") from e


def prepare_output_dirs(output_dir: str, save_depth: bool, overwrite: bool):
    if overwrite and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    img_out = os.path.join(output_dir, "images")
    sparse_out = os.path.join(output_dir, "sparse", "0")
    depth_out = os.path.join(output_dir, "depths")
    report_out = os.path.join(output_dir, "reports")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(sparse_out, exist_ok=True)
    os.makedirs(report_out, exist_ok=True)
    if save_depth:
        os.makedirs(depth_out, exist_ok=True)

    return img_out, sparse_out, depth_out, report_out


def list_input_images(input_dir: str):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(IMG_EXTS)])
    if len(files) < 2:
        raise ValueError("At least 2 images are required")
    return files


def load_original_images(input_dir: str, file_names):
    images = []
    sizes = []
    for fname in file_names:
        src = os.path.join(input_dir, fname)
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img)
            images.append(arr)
            sizes.append((arr.shape[1], arr.shape[0]))  # (w, h)
    return images, sizes


# ---------- export prep ----------
def build_export_views(coord_system, output_names, original_images, real_sizes, intrinsics, rgb_imgs, depth_maps):
    """
    Unify export triplets: export image / export intrinsics / export depth
    Ensure all three are in the same coordinate system.
    """
    export_images = []
    export_sizes = []
    export_intrinsics = []
    export_depths = [] if depth_maps is not None else None

    for i, name in enumerate(output_names):
        rgb_img = ensure_hwc_image(rgb_imgs[i])
        dust3r_h, dust3r_w = rgb_img.shape[:2]
        K = np.asarray(intrinsics[i], dtype=np.float64).copy()

        if coord_system == "dust3r":
            out_img = array_to_uint8_rgb(rgb_img)
            out_w, out_h = dust3r_w, dust3r_h
            out_K = K
            if depth_maps is not None:
                out_depth = np.asarray(tensor_to_numpy(depth_maps[i]), dtype=np.float32)
        else:  # original
            real_w, real_h = real_sizes[i]
            out_img = original_images[i]
            scale_x = real_w / dust3r_w
            scale_y = real_h / dust3r_h
            out_w, out_h = real_w, real_h
            out_K = K.copy()
            out_K[0, 0] *= scale_x
            out_K[1, 1] *= scale_y
            out_K[0, 2] *= scale_x
            out_K[1, 2] *= scale_y
            if depth_maps is not None:
                out_depth = resize_depth_to_wh(tensor_to_numpy(depth_maps[i]), (real_w, real_h))

        export_images.append(out_img)
        export_sizes.append((out_w, out_h))
        export_intrinsics.append(out_K)
        if export_depths is not None:
            export_depths.append(out_depth)

    return export_images, export_sizes, export_intrinsics, export_depths


def save_export_images(img_out_dir, output_names, export_images):
    for name, img in zip(output_names, export_images):
        dst = os.path.join(img_out_dir, name)
        Image.fromarray(array_to_uint8_rgb(img)).save(dst, format="JPEG", quality=95)


def build_camera_records(output_names, export_sizes, export_intrinsics):
    records = []
    for i, name in enumerate(output_names):
        w, h = export_sizes[i]
        K = np.asarray(export_intrinsics[i], dtype=np.float64)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        records.append((i + 1, w, h, fx, fy, cx, cy, name))
    return records


def write_cameras_bin(sparse_out, cameras):
    with open(os.path.join(sparse_out, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<I", len(cameras)))  # uint32 (COLMAP standard)
        for cid, w, h, fx, fy, cx, cy, _ in cameras:
            f.write(struct.pack("<iiII", cid, 1, w, h))  # PINHOLE (uint32 for w,h)
            f.write(struct.pack("<4d", fx, fy, cx, cy))


def write_images_bin(sparse_out, poses, cameras):
    with open(os.path.join(sparse_out, "images.bin"), "wb") as f:
        f.write(struct.pack("<I", len(poses)))  # uint32 (COLMAP standard)
        for i, pose in enumerate(poses):
            c2w = np.asarray(pose, dtype=np.float64)
            if c2w.shape != (4, 4):
                raise ValueError(f"Pose matrix {i} dimension error: {c2w.shape}")
            w2c = safe_inverse_pose(c2w, i)
            q = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # xyzw
            t = w2c[:3, 3]
            name = cameras[i][7]
            name_bytes = name.encode("utf-8") + b"\0"

            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<4d", q[3], q[0], q[1], q[2]))  # wxyz
            f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<i", i + 1))
            f.write(name_bytes)
            f.write(struct.pack("<I", 0))  # uint32 (COLMAP standard)


def save_depths(depth_out_dir, output_names, export_depths):
    for name, depth in zip(output_names, export_depths):
        np.save(os.path.join(depth_out_dir, name.replace(".jpg", ".npy")), np.asarray(depth, dtype=np.float32))


def aggregate_points(pts3d_list, masks, rgb_imgs):
    all_pts, all_cols = [], []
    per_view_counts = []

    if len(pts3d_list) != len(masks) or len(pts3d_list) != len(rgb_imgs):
        raise RuntimeError("pts3d_list / masks / rgb_imgs count mismatch")

    for i, (pts, mask, img) in enumerate(zip(pts3d_list, masks, rgb_imgs)):
        pts_np = np.asarray(tensor_to_numpy(pts), dtype=np.float32).reshape(-1, 3)
        mask_np = np.asarray(tensor_to_numpy(mask)).reshape(-1).astype(bool)
        img_np = ensure_hwc_image(img)
        col_np = array_to_uint8_rgb(img_np).reshape(-1, 3)

        if len(pts_np) != len(mask_np):
            raise ValueError(f"View {i}: pts and mask count mismatch: {len(pts_np)} vs {len(mask_np)}")
        if len(col_np) != len(mask_np):
            raise ValueError(f"View {i}: color and mask count mismatch: {len(col_np)} vs {len(mask_np)}")

        valid_pts = pts_np[mask_np]
        valid_cols = col_np[mask_np]
        per_view_counts.append(int(len(valid_pts)))

        if len(valid_pts) > 0:
            all_pts.append(valid_pts)
            all_cols.append(valid_cols)

    if not all_pts:
        raise RuntimeError("No valid points to export, check DUSt3R output or masks")

    pts_all = np.concatenate(all_pts, axis=0)
    cols_all = np.concatenate(all_cols, axis=0)
    return pts_all, cols_all, per_view_counts


def deduplicate_and_downsample_points(pts_all, cols_all, max_points, seed):
    rounded = np.round(pts_all, 3)
    _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    pts_unique = pts_all[uniq_idx]
    cols_unique = cols_all[uniq_idx]

    rng = np.random.default_rng(seed)
    sampled = False
    if len(pts_unique) > max_points:
        keep = rng.choice(len(pts_unique), max_points, replace=False)
        keep = np.sort(keep)
        pts_unique = pts_unique[keep]
        cols_unique = cols_unique[keep]
        sampled = True

    return pts_unique, cols_unique, len(uniq_idx), sampled


def write_points3d_bin(sparse_out, pts_all, cols_all):
    with open(os.path.join(sparse_out, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<I", len(pts_all)))  # uint32 (COLMAP standard)
        for i, (pt, col) in enumerate(zip(pts_all, cols_all)):
            f.write(struct.pack("<I", i + 1))  # uint32 (COLMAP standard)
            f.write(struct.pack("<3d", float(pt[0]), float(pt[1]), float(pt[2])))
            f.write(struct.pack("<3B", int(col[0]), int(col[1]), int(col[2])))
            f.write(struct.pack("<d", 0.0))  # error
            f.write(struct.pack("<I", 0))    # track length, uint32 (COLMAP standard)


def write_points3d_ply(sparse_out, pts_all, cols_all):
    vertex = np.array(
        [(float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2])) for p, c in zip(pts_all, cols_all)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    PlyData([PlyElement.describe(vertex, "vertex")]).write(os.path.join(sparse_out, "points3D.ply"))




def verify_export_consistency(img_out_dir, cameras):
    """
    Check if exported images match cameras.bin width/height.
    Raise error if any mismatch.
    """
    print("[*] Verifying export consistency (images vs cameras.bin)...")
    summary = []
    mismatch = []

    for cid, w, h, fx, fy, cx, cy, name in cameras:
        img_path = os.path.join(img_out_dir, name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing exported image: {img_path}")

        with Image.open(img_path) as img:
            real_w, real_h = img.size

        ok = (real_w == w and real_h == h)
        row = {
            "camera_id": int(cid),
            "file": name,
            "image_width": int(real_w),
            "image_height": int(real_h),
            "camera_width": int(w),
            "camera_height": int(h),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "ok": bool(ok),
        }
        summary.append(row)

        status = "OK" if ok else "MISMATCH"
        print(f"[VERIFY] {name}: image=({real_w},{real_h}) | camera=({w},{h}) -> {status}")

        if not ok:
            mismatch.append(row)

    if mismatch:
        lines = ["Export inconsistency: following images do not match cameras.bin metadata:"]
        for row in mismatch:
            lines.append(
                f"- {row['file']}: image=({row['image_width']},{row['image_height']}), "
                f"camera=({row['camera_width']},{row['camera_height']})"
            )
        raise RuntimeError("\n".join(lines))

    print("[OK] Image sizes match cameras.bin")
    return summary


def save_export_manifest(output_dir, coord_system, cameras, consistency_rows):
    """
    Save export manifest for verification.
    """
    manifest = {
        "coord_system": coord_system,
        "num_cameras": int(len(cameras)),
        "cameras": consistency_rows,
    }
    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[*] Export manifest saved: {manifest_path}")
    return manifest_path

# ---------- evaluation ----------
def build_projection_data(pts3d_list, masks, export_images, export_depths, export_intrinsics, poses):
    data = []
    num_views = len(export_images)
    for i in range(num_views):
        pts = np.asarray(tensor_to_numpy(pts3d_list[i]), dtype=np.float32).reshape(-1, 3)
        mask = np.asarray(tensor_to_numpy(masks[i])).reshape(-1).astype(bool)
        H, W = export_images[i].shape[:2]

        if len(pts) != len(mask):
            raise ValueError(f"Evaluation: view {i} pts and mask count mismatch")
        if len(pts) != H * W:
            # In original mode, pts still from DUSt3R resolution, not required to match original image pixel count
            pass

        data.append({
            "pts": pts,
            "mask": mask,
            "K": np.asarray(export_intrinsics[i], dtype=np.float64),
            "c2w": np.asarray(poses[i], dtype=np.float64),
            "image_shape": (H, W),
            "depth": None if export_depths is None else np.asarray(export_depths[i], dtype=np.float32),
        })
    return data


def camera_center_from_c2w(c2w):
    return np.asarray(c2w[:3, 3], dtype=np.float64)


def project_world_points(points_world, K, c2w):
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid_z = z > 1e-8

    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    uv[valid_z, 0] = K[0, 0] * (pts_cam[valid_z, 0] / z[valid_z]) + K[0, 2]
    uv[valid_z, 1] = K[1, 1] * (pts_cam[valid_z, 1] / z[valid_z]) + K[1, 2]
    return uv, z, valid_z


def sample_points_for_evaluation(pts_all, sample_n, seed):
    n = len(pts_all)
    if n <= sample_n:
        return pts_all
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, sample_n, replace=False)
    return pts_all[np.sort(idx)]


def evaluate_coordinate_consistency(export_images, export_depths, export_intrinsics):
    per_view = []
    all_ok = True
    for i, img in enumerate(export_images):
        H, W = img.shape[:2]
        depth_ok = True
        if export_depths is not None:
            dH, dW = export_depths[i].shape[:2]
            depth_ok = (H == dH and W == dW)
        K = np.asarray(export_intrinsics[i])
        intr_ok = bool(K[0, 0] > 0 and K[1, 1] > 0 and 0 <= K[0, 2] <= W and 0 <= K[1, 2] <= H)
        ok = bool(depth_ok and intr_ok)
        all_ok = all_ok and ok
        per_view.append({
            "view_id": i,
            "image_hw": [H, W],
            "depth_hw": None if export_depths is None else list(export_depths[i].shape[:2]),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "ok": ok,
        })
    return {"all_ok": bool(all_ok), "per_view": per_view}


def evaluate_pose_stability(poses):
    centers = np.array([camera_center_from_c2w(np.asarray(p)) for p in poses], dtype=np.float64)
    if len(centers) < 2:
        return {
            "num_views": len(centers),
            "path_length": 0.0,
            "median_step": 0.0,
            "max_step": 0.0,
            "step_outlier_ratio": 0.0,
        }

    steps = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
    median_step = float(np.median(steps)) if len(steps) else 0.0
    max_step = float(np.max(steps)) if len(steps) else 0.0
    path_length = float(np.sum(steps)) if len(steps) else 0.0
    thresh = max(median_step * 3.0, 1e-8)
    outlier_ratio = float(np.mean(steps > thresh)) if len(steps) else 0.0

    return {
        "num_views": int(len(centers)),
        "path_length": path_length,
        "median_step": median_step,
        "max_step": max_step,
        "step_outlier_ratio": outlier_ratio,
    }


def evaluate_depth_consistency(proj_data, sample_points_world, depth_tol_rel=0.05, depth_tol_abs=0.01):
    if not proj_data or proj_data[0]["depth"] is None:
        return {"available": False}

    abs_errors = []
    rel_errors = []
    pass_flags = []
    per_view_stats = []

    for i, item in enumerate(proj_data):
        K = item["K"]
        c2w = item["c2w"]
        H, W = item["image_shape"]
        depth = item["depth"]

        uv, z, valid_z = project_world_points(sample_points_world, K, c2w)
        in_bounds = valid_z & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        if not np.any(in_bounds):
            per_view_stats.append({"view_id": i, "num_samples": 0, "pass_rate": 0.0})
            continue

        uu = np.rint(uv[in_bounds, 0]).astype(int)
        vv = np.rint(uv[in_bounds, 1]).astype(int)
        depth_gt = depth[vv, uu]
        valid_depth = np.isfinite(depth_gt) & (depth_gt > 1e-8)
        if not np.any(valid_depth):
            per_view_stats.append({"view_id": i, "num_samples": int(np.sum(in_bounds)), "pass_rate": 0.0})
            continue

        z_sel = z[in_bounds][valid_depth]
        d_sel = depth_gt[valid_depth]
        err_abs = np.abs(z_sel - d_sel)
        err_rel = err_abs / np.maximum(np.abs(d_sel), 1e-8)
        passed = (err_abs <= depth_tol_abs) | (err_rel <= depth_tol_rel)

        abs_errors.append(err_abs)
        rel_errors.append(err_rel)
        pass_flags.append(passed)
        per_view_stats.append({
            "view_id": i,
            "num_samples": int(len(err_abs)),
            "mean_abs_error": float(np.mean(err_abs)),
            "median_abs_error": float(np.median(err_abs)),
            "mean_rel_error": float(np.mean(err_rel)),
            "pass_rate": float(np.mean(passed)),
        })

    if not abs_errors:
        return {"available": True, "num_valid_samples": 0}

    abs_errors = np.concatenate(abs_errors)
    rel_errors = np.concatenate(rel_errors)
    pass_flags = np.concatenate(pass_flags)
    return {
        "available": True,
        "num_valid_samples": int(len(abs_errors)),
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "p90_abs_error": float(np.percentile(abs_errors, 90)),
        "mean_rel_error": float(np.mean(rel_errors)),
        "median_rel_error": float(np.median(rel_errors)),
        "pass_rate": float(np.mean(pass_flags)),
        "per_view": per_view_stats,
    }


def evaluate_multiview_stability(proj_data, sample_points_world, depth_tol_rel=0.05, depth_tol_abs=0.01):
    if not proj_data or proj_data[0]["depth"] is None:
        return {"available": False}

    valid_counts = np.zeros(len(sample_points_world), dtype=np.int32)
    visible_counts = np.zeros(len(sample_points_world), dtype=np.int32)

    for item in proj_data:
        K = item["K"]
        c2w = item["c2w"]
        H, W = item["image_shape"]
        depth = item["depth"]

        uv, z, valid_z = project_world_points(sample_points_world, K, c2w)
        in_bounds = valid_z & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        visible_counts += in_bounds.astype(np.int32)
        if not np.any(in_bounds):
            continue

        uu = np.rint(uv[in_bounds, 0]).astype(int)
        vv = np.rint(uv[in_bounds, 1]).astype(int)
        depth_gt = depth[vv, uu]
        valid_depth = np.isfinite(depth_gt) & (depth_gt > 1e-8)
        if not np.any(valid_depth):
            continue

        z_sel = z[in_bounds][valid_depth]
        d_sel = depth_gt[valid_depth]
        err_abs = np.abs(z_sel - d_sel)
        err_rel = err_abs / np.maximum(np.abs(d_sel), 1e-8)
        passed = (err_abs <= depth_tol_abs) | (err_rel <= depth_tol_rel)

        idx_in_bounds = np.flatnonzero(in_bounds)
        idx_valid = idx_in_bounds[valid_depth]
        valid_counts[idx_valid] += passed.astype(np.int32)

    has_visibility = visible_counts > 0
    stability = np.zeros(len(sample_points_world), dtype=np.float64)
    stability[has_visibility] = valid_counts[has_visibility] / visible_counts[has_visibility]

    return {
        "available": True,
        "num_points": int(len(sample_points_world)),
        "num_visible_points": int(np.sum(has_visibility)),
        "mean_visible_views": float(np.mean(visible_counts[has_visibility])) if np.any(has_visibility) else 0.0,
        "mean_valid_views": float(np.mean(valid_counts[has_visibility])) if np.any(has_visibility) else 0.0,
        "mean_stability": float(np.mean(stability[has_visibility])) if np.any(has_visibility) else 0.0,
        "median_stability": float(np.median(stability[has_visibility])) if np.any(has_visibility) else 0.0,
        "stable_ratio_ge_0_5": float(np.mean(stability[has_visibility] >= 0.5)) if np.any(has_visibility) else 0.0,
    }


def evaluate_local_geometry(points_world, k=16):
    n = len(points_world)
    if n < max(8, k + 1):
        return {"available": False, "num_points": int(n)}

    tree = cKDTree(points_world)
    dists, idxs = tree.query(points_world, k=min(k + 1, n))
    # First column is self, remove it
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    neigh_mean = np.mean(dists, axis=1)
    neigh_std = np.std(dists, axis=1)

    # normalized deviation to neighborhood centroid
    neigh_pts = points_world[idxs]
    neigh_centers = np.mean(neigh_pts, axis=1)
    dev = np.linalg.norm(points_world - neigh_centers, axis=1) / np.maximum(neigh_mean, 1e-8)

    outlier_thresh = np.median(dev) + 2.0 * np.std(dev)
    outlier_ratio = float(np.mean(dev > outlier_thresh))

    return {
        "available": True,
        "num_points": int(n),
        "k": int(dists.shape[1]),
        "mean_neighbor_dist": float(np.mean(neigh_mean)),
        "median_neighbor_dist": float(np.median(neigh_mean)),
        "mean_neighbor_std": float(np.mean(neigh_std)),
        "mean_normalized_deviation": float(np.mean(dev)),
        "median_normalized_deviation": float(np.median(dev)),
        "p90_normalized_deviation": float(np.percentile(dev, 90)),
        "outlier_ratio": outlier_ratio,
    }


def score_report(coord_res, pose_res, depth_res, mv_res, local_res, per_view_counts, raw_point_count, dedup_count, final_count):
    score = 0
    reasons = []

    # 1. Coordinate consistency 0-2
    if coord_res["all_ok"]:
        score += 2
        reasons.append("Coordinate consistency: image/intrinsics/depth size match")
    else:
        reasons.append("Coordinate consistency: view size or intrinsics anomaly")

    # 2. Pose stability 0-2
    outlier_ratio = pose_res.get("step_outlier_ratio", 1.0)
    if outlier_ratio <= 0.05:
        score += 2
        reasons.append("Camera trajectory: smooth")
    elif outlier_ratio <= 0.15:
        score += 1
        reasons.append("Camera trajectory: usable with minor jumps")
    else:
        reasons.append("Camera trajectory: obvious jumps")

    # 3. Depth consistency 0-2
    if depth_res.get("available", False) and depth_res.get("num_valid_samples", 0) > 0:
        pass_rate = depth_res.get("pass_rate", 0.0)
        if pass_rate >= 0.70:
            score += 2
            reasons.append("Depth consistency: good")
        elif pass_rate >= 0.45:
            score += 1
            reasons.append("Depth consistency: moderate")
        else:
            reasons.append("Depth consistency: weak")
    else:
        reasons.append("Depth consistency: not evaluated (depth not saved)")

    # 4. Multiview stability 0-2
    if mv_res.get("available", False):
        st = mv_res.get("mean_stability", 0.0)
        if st >= 0.60:
            score += 2
            reasons.append("Multiview stability: good")
        elif st >= 0.35:
            score += 1
            reasons.append("Multiview stability: moderate")
        else:
            reasons.append("Multiview stability: weak")
    else:
        reasons.append("Multiview stability: not evaluated (depth not saved)")

    # 5. Local geometry quality 0-2
    if local_res.get("available", False):
        out_ratio = local_res.get("outlier_ratio", 1.0)
        if out_ratio <= 0.12:
            score += 2
            reasons.append("Local geometry: low outlier ratio")
        elif out_ratio <= 0.25:
            score += 1
            reasons.append("Local geometry: usable")
        else:
            reasons.append("Local geometry: many outliers")
    else:
        reasons.append("Local geometry: insufficient samples")

    # 6. Point cloud validity/redundancy control 0-2
    median_points = float(np.median(per_view_counts)) if per_view_counts else 0.0
    dedup_ratio = dedup_count / max(raw_point_count, 1)
    if median_points > 1000 and final_count > 10000 and dedup_ratio >= 0.10:
        score += 2
        reasons.append("Point cloud validity: sufficient valid points")
    elif median_points > 200 and final_count > 2000:
        score += 1
        reasons.append("Point cloud validity: barely sufficient")
    else:
        reasons.append("Point cloud validity: insufficient points")

    if score >= 10:
        grade = "High"
        recommendation = "High quality input, ready for structure scoring and first pruning experiment."
    elif score >= 8:
        grade = "Medium"
        recommendation = "Usable input, suggest prototype experiment first, focus on anomaly views and boundary regions."
    else:
        grade = "Low"
        recommendation = "Weak quality input, suggest checking pose, depth consistency and coordinate system before pruning."

    return {
        "score_0_to_12": int(score),
        "grade": grade,
        "reasons": reasons,
        "recommendation": recommendation,
    }


def run_evaluation(args, report_out, export_images, export_depths, export_intrinsics, poses,
                   pts3d_list, masks, pts_final, per_view_counts,
                   raw_point_count, dedup_count, final_count):
    print("[*] Running quality evaluation...")
    coord_res = evaluate_coordinate_consistency(export_images, export_depths, export_intrinsics)
    pose_res = evaluate_pose_stability(poses)

    sample_points = sample_points_for_evaluation(pts_final, args.eval_sample_points, args.seed)
    proj_data = build_projection_data(pts3d_list, masks, export_images, export_depths, export_intrinsics, poses)
    depth_res = evaluate_depth_consistency(
        proj_data, sample_points,
        depth_tol_rel=args.eval_depth_tol_rel,
        depth_tol_abs=args.eval_depth_tol_abs,
    )
    mv_res = evaluate_multiview_stability(
        proj_data, sample_points,
        depth_tol_rel=args.eval_depth_tol_rel,
        depth_tol_abs=args.eval_depth_tol_abs,
    )
    local_res = evaluate_local_geometry(sample_points, k=args.eval_knn_k)

    summary = score_report(
        coord_res, pose_res, depth_res, mv_res, local_res,
        per_view_counts, raw_point_count, dedup_count, final_count
    )

    report = {
        "config": {
            "coord_system": args.coord_system,
            "save_depth": bool(args.save_depth),
            "eval_sample_points": int(min(len(pts_final), args.eval_sample_points)),
            "eval_knn_k": int(args.eval_knn_k),
            "eval_depth_tol_rel": float(args.eval_depth_tol_rel),
            "eval_depth_tol_abs": float(args.eval_depth_tol_abs),
        },
        "summary": summary,
        "coordinate_consistency": coord_res,
        "pose_stability": pose_res,
        "depth_consistency": depth_res,
        "multiview_stability": mv_res,
        "local_geometry": local_res,
        "point_stats": {
            "raw_point_count": int(raw_point_count),
            "deduplicated_point_count": int(dedup_count),
            "final_point_count": int(final_count),
            "per_view_valid_points": [int(x) for x in per_view_counts],
            "median_per_view_valid_points": float(np.median(per_view_counts)) if per_view_counts else 0.0,
        },
    }

    json_path = os.path.join(report_out, "quality_report.json")
    txt_path = os.path.join(report_out, "quality_report.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        s = report["summary"]
        f.write("DUSt3R Quality Evaluation Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Score: {s['score_0_to_12']} / 12\n")
        f.write(f"Grade: {s['grade']}\n")
        f.write(f"Recommendation: {s['recommendation']}\n\n")

        f.write("[Key Metrics]\n")
        f.write(f"- Coordinate consistency: {'Pass' if coord_res['all_ok'] else 'Anomaly'}\n")
        f.write(f"- Pose jump ratio: {pose_res.get('step_outlier_ratio', 0.0):.4f}\n")
        if depth_res.get("available", False):
            f.write(f"- Depth consistency pass rate: {depth_res.get('pass_rate', 0.0):.4f}\n")
            f.write(f"- Depth median abs error: {depth_res.get('median_abs_error', 0.0):.6f}\n")
        else:
            f.write("- Depth consistency: Not evaluated\n")
        if mv_res.get("available", False):
            f.write(f"- Multiview mean stability: {mv_res.get('mean_stability', 0.0):.4f}\n")
        else:
            f.write("- Multiview stability: Not evaluated\n")
        if local_res.get("available", False):
            f.write(f"- Local geometry outlier ratio: {local_res.get('outlier_ratio', 0.0):.4f}\n")
        else:
            f.write("- Local geometry: Not evaluated\n")
        f.write(f"- Final point count: {final_count}\n\n")

        f.write("[Reasons]\n")
        for x in s["reasons"]:
            f.write(f"- {x}\n")

    print(f"[*] Evaluation complete: {txt_path}")
    return report


# ---------- main ----------
def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"[*] DUSt3R -> 3DGS | Input: {args.input} | Output: {args.output} | Mode: {args.coord_system}")

    img_out, sparse_out, depth_out, report_out = prepare_output_dirs(args.output, args.save_depth, args.overwrite)
    input_files = list_input_images(args.input)
    original_images, real_sizes = load_original_images(args.input, input_files)
    output_names = [f"{i:05d}.jpg" for i in range(len(input_files))]
    print(f"[*] Input image count: {len(input_files)}")

    print("[*] Loading model...")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA specified but not available, switching to CPU")
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    try:
        model = (
            AsymmetricCroCo3DStereo.from_pretrained(args.model).to(device)
            if os.path.exists(args.model)
            else AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}") from e

    # DUSt3R load_images reads from disk; convert and save original images to temp dir to avoid EXIF/format issues
    tmp_input_dir = os.path.join(args.output, "_dust3r_input")
    os.makedirs(tmp_input_dir, exist_ok=True)
    for name, img in zip(output_names, original_images):
        Image.fromarray(array_to_uint8_rgb(img)).save(os.path.join(tmp_input_dir, name), format="JPEG", quality=95)

    print("[*] DUSt3R inference...")
    images = load_images(tmp_input_dir, size=args.resolution)
    if len(images) != len(output_names):
        raise RuntimeError(f"load_images count mismatch: expected {len(output_names)}, got {len(images)}")

    pairs = make_pairs(images, scene_graph="swin-4-2", prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1)

    print("[*] Global alignment...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, base_scale=args.base_scale)
    loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
    try:
        print(f"[*] Alignment complete (loss: {float(loss):.4f})")
    except Exception:
        print(f"[*] Alignment complete (loss: {loss})")

    intrinsics = tensor_to_numpy(scene.get_intrinsics())
    poses = tensor_to_numpy(scene.get_im_poses())
    depth_maps = scene.get_depthmaps() if args.save_depth else None
    pts3d_list = scene.get_pts3d()
    masks = scene.get_masks()
    rgb_imgs = scene.imgs

    num_views = len(output_names)
    if len(poses) != num_views or len(intrinsics) != num_views or len(rgb_imgs) != num_views:
        raise RuntimeError(
            f"View count mismatch: images={num_views}, poses={len(poses)}, intrinsics={len(intrinsics)}, rgb_imgs={len(rgb_imgs)}"
        )

    export_images, export_sizes, export_intrinsics, export_depths = build_export_views(
        args.coord_system, output_names, original_images, real_sizes, intrinsics, rgb_imgs, depth_maps
    )

    print("[*] Exporting images/ ...")
    save_export_images(img_out, output_names, export_images)

    print("[*] Exporting cameras.bin ...")
    cameras = build_camera_records(output_names, export_sizes, export_intrinsics)
    write_cameras_bin(sparse_out, cameras)

    consistency_rows = verify_export_consistency(img_out, cameras)
    save_export_manifest(args.output, args.coord_system, cameras, consistency_rows)

    print("[*] Exporting images.bin ...")
    write_images_bin(sparse_out, poses, cameras)

    if args.save_depth:
        print("[*] Saving depth maps ...")
        save_depths(depth_out, output_names, export_depths)

    print("[*] Aggregating point cloud ...")
    pts_raw, cols_raw, per_view_counts = aggregate_points(pts3d_list, masks, rgb_imgs)
    raw_point_count = int(len(pts_raw))
    pts_final, cols_final, dedup_count, sampled = deduplicate_and_downsample_points(
        pts_raw, cols_raw, args.max_points, args.seed
    )
    final_count = int(len(pts_final))

    print("[*] Exporting points3D.bin / points3D.ply ...")
    write_points3d_bin(sparse_out, pts_final, cols_final)
    write_points3d_ply(sparse_out, pts_final, cols_final)

    if args.evaluate:
        run_evaluation(
            args, report_out,
            export_images, export_depths, export_intrinsics, poses,
            pts3d_list, masks, pts_final, per_view_counts,
            raw_point_count, dedup_count, final_count,
        )

    # Cleanup temp input directory
    try:
        shutil.rmtree(tmp_input_dir)
    except Exception:
        pass

    print(f"[OK] Done! Views={num_views}, Raw points={raw_point_count}, Dedup points={dedup_count}, Final points={final_count}")
    if sampled:
        print(f"[*] Randomly downsampled due to exceeding max_points={args.max_points}")
    print(f"    Output directory: {args.output}")
    print(f"    Coordinate mode: {args.coord_system}")
    if args.evaluate:
        print(f"    Evaluation report: {os.path.join(report_out, 'quality_report.txt')}")


if __name__ == "__main__":
    main()