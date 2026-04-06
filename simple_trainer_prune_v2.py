import json
import math
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

GSPLAT_SRC_ROOT = os.path.join(ROOT_DIR, "gsplat")
if os.path.isdir(os.path.join(GSPLAT_SRC_ROOT, "gsplat")) and GSPLAT_SRC_ROOT not in sys.path:
    sys.path.insert(0, GSPLAT_SRC_ROOT)

DUST3R_SRC_ROOT = os.path.join(ROOT_DIR, "dust3r")
if os.path.isdir(DUST3R_SRC_ROOT) and DUST3R_SRC_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_SRC_ROOT)
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)

# Try to import fused_ssim, fallback to torchmetrics if not available
try:
    from fused_ssim import fused_ssim
    HAS_FUSED_SSIM = True
except ImportError:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    _ssim_metric = None
    _ssim_device = None
    def fused_ssim(pred, target, padding="valid"):
        """Fallback SSIM using torchmetrics with automatic device handling."""
        global _ssim_metric, _ssim_device
        if _ssim_metric is None or _ssim_device != pred.device:
            _ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
            _ssim_device = pred.device
        return _ssim_metric(pred, target)
    HAS_FUSED_SSIM = False

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat.exporter import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # ========== Structure-Aware Pruning Parameters ==========
    # Enable structure-aware pruning
    enable_prune: bool = False
    # Directory containing dense depth maps (.npy files, named as 00000.npy etc.)
    dense_depth_dir: Optional[str] = None
    # Pruning step (training progress ratio, e.g., 0.7 means 70% of max_steps)
    prune_step_ratio: float = 0.7
    # Pruning threshold for structure score S_i
    prune_tau: float = 0.3
    # Structure score weights S_i = lambda_d * D_i + lambda_v * V_i + lambda_n * N_i
    prune_lambda_d: float = 0.5
    prune_lambda_v: float = 0.3
    prune_lambda_n: float = 0.2
    # Sigma for depth consistency score (ratio of scene depth range)
    prune_sigma_d_ratio: float = 0.05
    # Sigma for neighborhood geometric consistency
    prune_sigma_n: float = 0.1
    # Number of nearest neighbors for N_i calculation
    prune_k_neighbors: int = 8
    # Maximum number of candidate neighbors used for approximate KNN when N is large
    prune_neighbor_candidate_size: int = 65536
    # Chunk size used in distance computation to avoid OOM
    prune_neighbor_chunk_size: int = 4096
    # Depth error threshold for valid view (ratio of scene depth range)
    prune_depth_err_ratio: float = 0.1
    # Minimum opacity contribution threshold
    prune_min_opacity: float = 0.01
    # Save pruning visualization: standard 3DGS ply (before/after)
    prune_save_ply: bool = False
    # Save RGB-colored point cloud for CloudCompare visualization
    prune_save_rgb_ply: bool = False
    # Save eval render images to disk (GT + render side-by-side PNG)
    save_eval_images: bool = False
    # Disable tensorboard scalar logging (speeds up training on cloud)
    disable_tb: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
        parser: Parser,
        init_type: str = "sfm",
        init_num_pts: int = 100_000,
        init_extent: float = 3.0,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        opacities_lr: float = 5e-2,
        quats_lr: float = 1e-3,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 2.5e-3 / 20,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        sparse_grad: bool = False,
        visible_adam: bool = False,
        batch_size: int = 1,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def _build_pruned_optimizers(
        splats: torch.nn.ParameterDict,
        cfg: Config,
        world_size: int = 1,
) -> Dict[str, torch.optim.Optimizer]:
    """Rebuild optimizers after pruning. Optimizer state is intentionally reset."""
    BS = cfg.batch_size * world_size
    if cfg.sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif cfg.visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam

    def _make(name: str, lr: float):
        return optimizer_class(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )

    optimizers = {
        "means": _make("means", cfg.means_lr * cfg.global_scale * math.sqrt(BS)),
        "scales": _make("scales", cfg.scales_lr * math.sqrt(BS)),
        "quats": _make("quats", cfg.quats_lr * math.sqrt(BS)),
        "opacities": _make("opacities", cfg.opacities_lr * math.sqrt(BS)),
    }
    if "sh0" in splats:
        optimizers["sh0"] = _make("sh0", cfg.sh0_lr * math.sqrt(BS))
    if "shN" in splats:
        optimizers["shN"] = _make("shN", cfg.shN_lr * math.sqrt(BS))
    if "features" in splats:
        optimizers["features"] = _make("features", cfg.sh0_lr * math.sqrt(BS))
    if "colors" in splats:
        optimizers["colors"] = _make("colors", cfg.sh0_lr * math.sqrt(BS))
    return optimizers


@torch.no_grad()
def _compute_neighbor_score(
        means: Tensor,
        scene_scale: float,
        cfg: Config,
) -> Tuple[Tensor, Dict[str, Union[int, str]]]:
    """Approximate neighborhood score without materializing a full N x N distance matrix."""
    N = means.shape[0]
    device = means.device
    k = min(cfg.prune_k_neighbors, max(N - 1, 1))
    if N <= 1 or k <= 0:
        return torch.ones(N, device=device), {"mode": "degenerate", "candidate_size": int(N), "k": int(k)}

    candidate_size = min(cfg.prune_neighbor_candidate_size, N)
    chunk_size = max(256, cfg.prune_neighbor_chunk_size)

    if candidate_size < N:
        candidate_idx = torch.randperm(N, device=device)[:candidate_size]
        candidate_means = means[candidate_idx]
        mode = "approx_subset"
    else:
        candidate_idx = torch.arange(N, device=device)
        candidate_means = means
        mode = "exact_chunked"

    knn_dists = []
    knn_global_idx = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        query = means[start:end]
        dists = torch.cdist(query, candidate_means)
        query_global = torch.arange(start, end, device=device)[:, None]
        self_mask = candidate_idx[None, :] == query_global
        dists = dists.masked_fill(self_mask, float("inf"))

        k_eff = min(k, candidate_means.shape[0] - 1 if candidate_means.shape[0] > 1 else 1)
        vals, idx_local = torch.topk(dists, k=k_eff, largest=False, dim=1)
        knn_dists.append(vals)
        knn_global_idx.append(candidate_idx[idx_local])

    knn_dists = torch.cat(knn_dists, dim=0)
    knn_global_idx = torch.cat(knn_global_idx, dim=0)
    neighbor_means = means[knn_global_idx]
    neighbor_centers = neighbor_means.mean(dim=1)
    dist_to_center = torch.norm(means - neighbor_centers, dim=-1)
    neighbor_radius = knn_dists.mean(dim=1).clamp_min(1e-8)
    neighbor_deviation = dist_to_center / neighbor_radius

    sigma_n = max(cfg.prune_sigma_n * scene_scale, 1e-8)
    N_i = torch.exp(-neighbor_deviation / sigma_n)
    meta = {
        "mode": mode,
        "candidate_size": int(candidate_size),
        "k": int(k),
        "chunk_size": int(chunk_size),
    }
    return N_i, meta


@torch.no_grad()
def structure_aware_prune(
        splats: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        parser: Parser,
        trainset: Dataset,
        scene_scale: float,
        cfg: Config,
        device: str,
        step: int,
        world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer], Dict]:
    """
    Structure-aware Gaussian pruning module.

    Computes structure confidence score
    S_i = lambda_d * D_i + lambda_v * V_i + lambda_n * N_i
    using depth consistency, per-view stability, and neighborhood consistency.

    Optimizer state is intentionally rebuilt after pruning.
    """
    N = len(splats["means"])
    means = splats["means"]
    opacities = torch.sigmoid(splats["opacities"])

    num_train_images = len(trainset)
    indices = trainset.indices
    camtoworlds_np = parser.camtoworlds[indices]

    if camtoworlds_np.shape[1] == 3:
        camtoworlds_all = torch.from_numpy(camtoworlds_np).float().to(device)
        camtoworlds_all = torch.cat([
            camtoworlds_all,
            torch.zeros(num_train_images, 1, 4, device=device),
        ], dim=1)
        camtoworlds_all[:, 3, 3] = 1.0
    else:
        camtoworlds_all = torch.from_numpy(camtoworlds_np).float().to(device)

    camera_ids = [parser.camera_ids[idx] for idx in indices]
    Ks_all = torch.stack([
        torch.from_numpy(parser.Ks_dict[cid]).float().to(device)
        for cid in camera_ids
    ])
    imsizes = [parser.imsize_dict[cid] for cid in camera_ids]
    viewmats = torch.linalg.inv(camtoworlds_all)

    means_homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=-1)
    means_cam_full = torch.einsum('mij,nj->mni', viewmats, means_homo)
    points_cam = means_cam_full[:, :, :3]
    depths_cam = points_cam[:, :, 2]

    fx = Ks_all[:, 0, 0].unsqueeze(1)
    fy = Ks_all[:, 1, 1].unsqueeze(1)
    cx = Ks_all[:, 0, 2].unsqueeze(1)
    cy = Ks_all[:, 1, 2].unsqueeze(1)

    z_safe = torch.clamp(depths_cam, min=1e-10)
    u = fx * points_cam[:, :, 0] / z_safe + cx
    v = fy * points_cam[:, :, 1] / z_safe + cy

    valid_proj_mask = torch.zeros(num_train_images, N, dtype=torch.bool, device=device)
    for img_idx, (w, h) in enumerate(imsizes):
        valid_proj_mask[img_idx] = (
                (u[img_idx] >= 0) & (u[img_idx] < w) &
                (v[img_idx] >= 0) & (v[img_idx] < h) &
                (depths_cam[img_idx] > cfg.near_plane) & (depths_cam[img_idx] < cfg.far_plane)
        )

    valid_depths = depths_cam[valid_proj_mask]
    if valid_depths.numel() > 0:
        depth_min = valid_depths.min().item()
        depth_max = valid_depths.max().item()
        depth_range = max(depth_max - depth_min, scene_scale)
    else:
        depth_range = scene_scale
    sigma_d = max(cfg.prune_sigma_d_ratio * depth_range, 1e-8)
    depth_err_threshold = cfg.prune_depth_err_ratio * depth_range

    depth_errors = torch.zeros(N, device=device)
    valid_depth_counts = torch.zeros(N, device=device)
    valid_view_counts = torch.zeros(N, device=device)
    measured_view_counts = torch.zeros(N, device=device)

    for img_idx in range(num_train_images):
        width, height = imsizes[img_idx]
        u_cam = u[img_idx]
        v_cam = v[img_idx]
        depth_cam = depths_cam[img_idx]
        valid = valid_proj_mask[img_idx]

        per_view_measured = torch.zeros(N, dtype=torch.bool, device=device)
        per_view_valid = torch.zeros(N, dtype=torch.bool, device=device)

        # 直接从 dense_depth_dir 读取深度图，不走 Dataset
        depth_gt = None
        if cfg.dense_depth_dir is not None:
            depth_path = os.path.join(cfg.dense_depth_dir, f"{indices[img_idx]:05d}.npy")
            if os.path.exists(depth_path):
                depth_gt = torch.from_numpy(np.load(depth_path)).float().to(device)

        if depth_gt is not None:
            if depth_gt.ndim == 3:
                depth_gt = depth_gt.squeeze(0)

            # 自动对齐尺度
            valid_idx = torch.where(valid)[0]
            if valid_idx.numel() > 10:
                u_s = u_cam[valid_idx].long().clamp(0, depth_gt.shape[1] - 1)
                v_s = v_cam[valid_idx].long().clamp(0, depth_gt.shape[0] - 1)
                gt_sampled = depth_gt[v_s, u_s]
                cam_sampled = depth_cam[valid_idx]
                gt_median = gt_sampled.median()
                cam_median = cam_sampled.median()
                if gt_median > 1e-6:
                    depth_gt = depth_gt * (cam_median / gt_median)

            # resize
            depth_h, depth_w = depth_gt.shape
            if depth_h != height or depth_w != width:
                depth_gt = F.interpolate(
                    depth_gt.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            # 采样并计算误差
            valid_idx = torch.where(valid)[0]
            if valid_idx.numel() > 0:
                u_valid = u_cam[valid_idx]
                v_valid = v_cam[valid_idx]
                u_norm = 2.0 * u_valid / max(width - 1, 1) - 1.0
                v_norm = 2.0 * v_valid / max(height - 1, 1) - 1.0
                grid = torch.stack([u_norm, v_norm], dim=-1).view(1, -1, 1, 2)
                depth_sampled = F.grid_sample(
                    depth_gt.unsqueeze(0).unsqueeze(0),
                    grid,
                    mode="bilinear",
                    align_corners=True,
                    padding_mode="border",
                ).view(-1)

                sampled_valid = (
                        (depth_sampled > cfg.near_plane) &
                        (depth_sampled < cfg.far_plane)
                )
                per_view_measured[valid_idx] = sampled_valid
                if sampled_valid.any():
                    err_valid = torch.abs(depth_cam[valid_idx] - depth_sampled)
                    measured_idx = valid_idx[sampled_valid]
                    measured_err = err_valid[sampled_valid]
                    depth_errors[measured_idx] += measured_err
                    valid_depth_counts[measured_idx] += 1.0
                    per_view_valid[measured_idx] = measured_err < depth_err_threshold

        measured_view_counts += per_view_measured.float()
        valid_view_counts += per_view_valid.float()

    mean_depth_errors = depth_errors / (valid_depth_counts + 1e-10)
    D_i = torch.exp(-mean_depth_errors / sigma_d)
    D_i = torch.where(valid_depth_counts > 0, D_i, torch.zeros_like(D_i))

    visible_views = valid_proj_mask.sum(dim=0).float()
    V_i = valid_view_counts / (visible_views + 1e-10)
    V_i = torch.where(visible_views > 0, V_i, torch.zeros_like(V_i))
    V_i = torch.clamp(V_i, 0.0, 1.0)

    N_i, neighbor_meta = _compute_neighbor_score(means, scene_scale, cfg)

    lambda_sum = cfg.prune_lambda_d + cfg.prune_lambda_v + cfg.prune_lambda_n
    if lambda_sum <= 0:
        raise ValueError("Pruning weights must sum to a positive value.")
    w_d = cfg.prune_lambda_d / lambda_sum
    w_v = cfg.prune_lambda_v / lambda_sum
    w_n = cfg.prune_lambda_n / lambda_sum
    S_i = w_d * D_i + w_v * V_i + w_n * N_i

    no_depth_mask = valid_depth_counts == 0
    opacity_sufficient_mask = opacities > cfg.prune_min_opacity
    if no_depth_mask.any():
        S_i[no_depth_mask] = 0.5 * opacities[no_depth_mask] + 0.5 * N_i[no_depth_mask]
    S_i = torch.where(opacity_sufficient_mask, S_i, S_i * 0.5)

    keep_mask = S_i >= cfg.prune_tau
    num_kept = keep_mask.sum().item()

    # 保底：至少保留 10% 或 10000 个
    min_keep = max(10000, int(N * 0.10))
    if num_kept < min_keep:
        print(f"[Pruning] WARNING: Only {num_kept} kept, falling back to top-{min_keep}")
        topk_idx = torch.topk(S_i, k=min_keep).indices
        keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
        keep_mask[topk_idx] = True
        num_kept = min_keep

    num_pruned = (~keep_mask).sum().item()

    print(f"[Pruning] Step {step}: Before pruning {N} Gaussians")
    print(f"[Pruning] Step {step}: Pruned {num_pruned} Gaussians, kept {num_kept}")
    print(f"[Pruning] Score distribution: D_i mean={D_i.mean().item():.4f} (std={D_i.std().item():.4f})")
    print(f"[Pruning]                   V_i mean={V_i.mean().item():.4f} (std={V_i.std().item():.4f})")
    print(f"[Pruning]                   N_i mean={N_i.mean().item():.4f} (std={N_i.std().item():.4f})")
    print(f"[Pruning]                   S_i mean={S_i.mean().item():.4f} (std={S_i.std().item():.4f})")
    print(f"[Pruning] Gaussians with no depth data: {no_depth_mask.sum().item()}")
    print(
        f"[Pruning] Neighbor search mode: {neighbor_meta['mode']} (candidates={neighbor_meta['candidate_size']}, k={neighbor_meta['k']})")

    new_splats = torch.nn.ParameterDict()
    for key in splats.keys():
        new_splats[key] = torch.nn.Parameter(splats[key][keep_mask].clone())
    new_splats = new_splats.to(device)

    new_optimizers = _build_pruned_optimizers(new_splats, cfg, world_size=world_size)

    stats = {
        "prune_step": step,
        "num_before": N,
        "num_after": num_kept,
        "num_pruned": num_pruned,
        "prune_ratio": num_pruned / N if N > 0 else 0.0,
        "D_mean": D_i.mean().item(),
        "D_std": D_i.std().item(),
        "V_mean": V_i.mean().item(),
        "V_std": V_i.std().item(),
        "N_mean": N_i.mean().item(),
        "N_std": N_i.std().item(),
        "S_mean": S_i.mean().item(),
        "S_std": S_i.std().item(),
        "visible_views_mean": visible_views.mean().item(),
        "valid_view_counts_mean": valid_view_counts.mean().item(),
        "measured_view_counts_mean": measured_view_counts.mean().item(),
        "no_depth_count": no_depth_mask.sum().item(),
        "low_opacity_count": (~opacity_sufficient_mask).sum().item(),
        "sigma_d": sigma_d,
        "sigma_n": max(cfg.prune_sigma_n * scene_scale, 1e-8),
        "tau": cfg.prune_tau,
        "lambda_d": w_d,
        "lambda_v": w_v,
        "lambda_n": w_n,
        "neighbor_search": neighbor_meta,
        "optimizer_state_reset": True,
        "strategy_state_reset_required": True,
    }

    return new_splats, new_optimizers, stats


def _save_rgb_ply(path: str, means: torch.Tensor, sh0: torch.Tensor) -> None:
    """Save Gaussian centers as a colored point cloud (CloudCompare compatible).
    Converts SH DC component to RGB. No external dependencies beyond numpy/struct.
    sh0 shape: [N, 1, 3]
    """
    import struct as _struct
    pts = means.detach().cpu().numpy().astype("float32")
    rgb = (sh0[:, 0, :].detach().cpu().numpy() * 0.28209479177387814 + 0.5)
    rgb = (rgb.clip(0.0, 1.0) * 255).astype("uint8")
    n = len(pts)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        buf = bytearray(n * 15)
        for i in range(n):
            offset = i * 15
            _struct.pack_into("<fff", buf, offset, pts[i, 0], pts[i, 1], pts[i, 2])
            buf[offset + 12] = int(rgb[i, 0])
            buf[offset + 13] = int(rgb[i, 1])
            buf[offset + 14] = int(rgb[i, 2])
        f.write(buf)
    print(f"[Pruning] RGB point cloud saved: {path}")


class Runner:
    """Engine for training and testing."""

    def __init__(
            self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        # When prune is enabled, also load depth data
        load_depths = cfg.depth_loss
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=load_depths,
            dense_depth_dir=cfg.dense_depth_dir if cfg.enable_prune else None,
        )
        self.trainset_prune = self.trainset  # Use same dataset for pruning
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            masks: Optional[Tensor] = None,
            rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
            camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
            **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 to avoid memory issues with large images
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                valid_depth_mask = (depths_gt > 0.0) & (depths > 0.0)
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = torch.where(depths_gt > 0.0, 1.0 / depths_gt, torch.zeros_like(depths_gt))
                if valid_depth_mask.any():
                    depthloss = F.l1_loss(disp[valid_depth_mask], disp_gt[valid_depth_mask]) * self.scene_scale
                else:
                    depthloss = torch.tensor(0.0, device=device, requires_grad=True)
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0 and not cfg.disable_tb:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                        "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                    step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # ========== Structure-Aware Pruning ==========
            # Execute pruning at prune_step (default 70% of max_steps)
            prune_step = int(cfg.max_steps * cfg.prune_step_ratio)
            if cfg.enable_prune and step == prune_step:
                # Save point cloud before pruning
                if cfg.prune_save_ply and world_rank == 0:
                    print(f"[Pruning] Saving point cloud before pruning...")
                    if self.cfg.app_opt:
                        rgb = self.app_module(
                            features=self.splats["features"],
                            embed_ids=None,
                            dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                            sh_degree=sh_degree_to_use,
                        )
                        rgb = rgb + self.splats["colors"]
                        rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                        sh0 = rgb_to_sh(rgb)
                        shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                    else:
                        sh0 = self.splats["sh0"]
                        shN = self.splats["shN"]
                    export_splats(
                        means=self.splats["means"],
                        scales=self.splats["scales"],
                        quats=self.splats["quats"],
                        opacities=self.splats["opacities"],
                        sh0=sh0,
                        shN=shN,
                        format="ply",
                        save_to=f"{self.ply_dir}/point_cloud_before_prune_{step}.ply",
                    )
                    if cfg.prune_save_rgb_ply:
                        _save_rgb_ply(
                            f"{self.ply_dir}/point_cloud_before_prune_{step}_rgb.ply",
                            self.splats["means"], sh0,
                        )

                # Execute pruning
                self.splats, self.optimizers, prune_stats = structure_aware_prune(
                    splats=self.splats,
                    optimizers=self.optimizers,
                    parser=self.parser,
                    trainset=self.trainset_prune,
                    scene_scale=self.scene_scale,
                    cfg=cfg,
                    device=device,
                    step=step,
                    world_size=world_size,
                )

                # Reinitialize densification strategy state after pruning because
                # Gaussian count and optimizer state have changed.
                self.cfg.strategy.check_sanity(self.splats, self.optimizers)
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.strategy_state = self.cfg.strategy.initialize_state(
                        scene_scale=self.scene_scale
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.strategy_state = self.cfg.strategy.initialize_state()
                else:
                    assert_never(self.cfg.strategy)

                # Save pruning stats
                if world_rank == 0:
                    with open(f"{self.stats_dir}/prune_stats_{step}.json", "w") as f:
                        json.dump(prune_stats, f)

                    # Log to tensorboard
                    self.writer.add_scalar("prune/num_before", prune_stats["num_before"], step)
                    self.writer.add_scalar("prune/num_after", prune_stats["num_after"], step)
                    self.writer.add_scalar("prune/D_mean", prune_stats["D_mean"], step)
                    self.writer.add_scalar("prune/V_mean", prune_stats["V_mean"], step)
                    self.writer.add_scalar("prune/N_mean", prune_stats["N_mean"], step)
                    self.writer.add_scalar("prune/S_mean", prune_stats["S_mean"], step)
                    self.writer.flush()

                # Save point cloud after pruning
                if cfg.prune_save_ply and world_rank == 0:
                    print(f"[Pruning] Saving point cloud after pruning...")
                    if self.cfg.app_opt:
                        rgb = self.app_module(
                            features=self.splats["features"],
                            embed_ids=None,
                            dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                            sh_degree=sh_degree_to_use,
                        )
                        rgb = rgb + self.splats["colors"]
                        rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                        sh0 = rgb_to_sh(rgb)
                        shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                    else:
                        sh0 = self.splats["sh0"]
                        shN = self.splats["shN"]
                    export_splats(
                        means=self.splats["means"],
                        scales=self.splats["scales"],
                        quats=self.splats["quats"],
                        opacities=self.splats["opacities"],
                        sh0=sh0,
                        shN=shN,
                        format="ply",
                        save_to=f"{self.ply_dir}/point_cloud_after_prune_{step}.ply",
                    )
                    if cfg.prune_save_rgb_ply:
                        _save_rgb_ply(
                            f"{self.ply_dir}/point_cloud_after_prune_{step}_rgb.ply",
                            self.splats["means"], sh0,
                        )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images (optional, gated by save_eval_images)
                if cfg.save_eval_images:
                    canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                    canvas = (canvas * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                        canvas,
                    )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i: i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
            self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
                        / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
