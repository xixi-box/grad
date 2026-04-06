import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


# -----------------------------------------------------------------------------
# Local COLMAP binary reader (no pycolmap dependency)
# -----------------------------------------------------------------------------

CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_bytes(fid, num_bytes: int) -> bytes:
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Failed to read {num_bytes} bytes, got {len(data)}.")
    return data


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    return struct.unpack(fmt, _read_bytes(fid, num_bytes))


def _read_c_string(fid) -> str:
    chars = []
    while True:
        ch = _read_bytes(fid, 1)
        if ch == b"\x00":
            break
        chars.append(ch)
    return b"".join(chars).decode("utf-8")


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion order is [qw, qx, qy, qz]."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qw * qz,
                2 * qx * qz + 2 * qw * qy,
            ],
            [
                2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qx * qz - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


@dataclass
class Camera:
    camera_id: int
    model_id: int
    camera_type: str
    width: int
    height: int
    params: np.ndarray

    @property
    def fx(self) -> float:
        if self.camera_type == "SIMPLE_PINHOLE":
            return float(self.params[0])
        return float(self.params[0])

    @property
    def fy(self) -> float:
        if self.camera_type == "SIMPLE_PINHOLE":
            return float(self.params[0])
        return float(self.params[1])

    @property
    def cx(self) -> float:
        if self.camera_type == "SIMPLE_PINHOLE":
            return float(self.params[1])
        return float(self.params[2])

    @property
    def cy(self) -> float:
        if self.camera_type == "SIMPLE_PINHOLE":
            return float(self.params[2])
        return float(self.params[3])

    @property
    def k1(self) -> float:
        if self.camera_type in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            return float(self.params[3])
        if self.camera_type in ("RADIAL", "RADIAL_FISHEYE"):
            return float(self.params[3])
        if self.camera_type in ("OPENCV", "OPENCV_FISHEYE"):
            return float(self.params[4])
        return 0.0

    @property
    def k2(self) -> float:
        if self.camera_type in ("RADIAL", "RADIAL_FISHEYE"):
            return float(self.params[4])
        if self.camera_type in ("OPENCV", "OPENCV_FISHEYE"):
            return float(self.params[5])
        return 0.0

    @property
    def p1(self) -> float:
        if self.camera_type == "OPENCV":
            return float(self.params[6])
        return 0.0

    @property
    def p2(self) -> float:
        if self.camera_type == "OPENCV":
            return float(self.params[7])
        return 0.0

    @property
    def k3(self) -> float:
        if self.camera_type == "OPENCV_FISHEYE":
            return float(self.params[6])
        return 0.0

    @property
    def k4(self) -> float:
        if self.camera_type == "OPENCV_FISHEYE":
            return float(self.params[7])
        return 0.0


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

    def R(self) -> np.ndarray:
        return _qvec_to_rotmat(self.qvec)


@dataclass
class Point3D:
    point3D_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: np.ndarray  # [track_len, 2], columns: image_id, point2D_idx


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "<IiQQ")
            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(_read_next_bytes(fid, 8 * num_params, "<" + "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(
                camera_id=int(camera_id),
                model_id=int(model_id),
                camera_type=model_name,
                width=int(width),
                height=int(height),
                params=params,
            )
    return cameras


def read_images_binary(path: str) -> Dict[int, ImageRecord]:
    images: Dict[int, ImageRecord] = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(fid, 4, "<I")[0]
            qvec = np.array(_read_next_bytes(fid, 32, "<4d"), dtype=np.float64)
            tvec = np.array(_read_next_bytes(fid, 24, "<3d"), dtype=np.float64)
            camera_id = _read_next_bytes(fid, 4, "<I")[0]
            name = _read_c_string(fid)
            num_points2D = _read_next_bytes(fid, 8, "<Q")[0]

            xys = np.zeros((num_points2D, 2), dtype=np.float64)
            point3D_ids = np.full((num_points2D,), -1, dtype=np.int64)
            for i in range(num_points2D):
                x, y, point3D_id = _read_next_bytes(fid, 24, "<2dq")
                xys[i] = [x, y]
                point3D_ids[i] = point3D_id

            images[int(image_id)] = ImageRecord(
                image_id=int(image_id),
                qvec=qvec,
                tvec=tvec,
                camera_id=int(camera_id),
                name=name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3d_binary(path: str) -> Dict[int, Point3D]:
    points3D: Dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        num_points3D = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(num_points3D):
            point3D_id = _read_next_bytes(fid, 8, "<Q")[0]
            xyz = np.array(_read_next_bytes(fid, 24, "<3d"), dtype=np.float64)
            rgb = np.array(_read_next_bytes(fid, 3, "<3B"), dtype=np.uint8)
            error = float(_read_next_bytes(fid, 8, "<d")[0])
            track_length = _read_next_bytes(fid, 8, "<Q")[0]
            track = np.zeros((track_length, 2), dtype=np.uint32)
            for i in range(track_length):
                image_id, point2D_idx = _read_next_bytes(fid, 8, "<2I")
                track[i] = [image_id, point2D_idx]

            points3D[int(point3D_id)] = Point3D(
                point3D_id=int(point3D_id),
                xyz=xyz,
                rgb=rgb,
                error=error,
                track=track,
            )
    return points3D


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        cameras_path = os.path.join(colmap_dir, "cameras.bin")
        images_path = os.path.join(colmap_dir, "images.bin")
        points3D_path = os.path.join(colmap_dir, "points3D.bin")
        if not os.path.exists(cameras_path):
            raise FileNotFoundError(f"Missing cameras.bin: {cameras_path}")
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Missing images.bin: {images_path}")
        if not os.path.exists(points3D_path):
            raise FileNotFoundError(f"Missing points3D.bin: {points3D_path}")

        cameras = read_cameras_binary(cameras_path)
        imdata = read_images_binary(images_path)
        points3d_dict = read_points3d_binary(points3D_path)

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        camtypes_seen = set()
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            camtypes_seen.add(type_)
            if type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            else:
                raise ValueError(
                    f"Only SIMPLE_PINHOLE/PINHOLE/SIMPLE_RADIAL/RADIAL/OPENCV/OPENCV_FISHEYE are supported, got {type_}"
                )

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if not all(t in ("SIMPLE_PINHOLE", "PINHOLE") for t in camtypes_seen):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names_unsorted = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names_unsorted)
        image_names = [image_names_unsorted[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": True,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and len(image_files) > 0 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        point_ids_sorted = sorted(points3d_dict.keys())
        points = np.array([points3d_dict[pid].xyz for pid in point_ids_sorted], dtype=np.float32)
        points_err = np.array([points3d_dict[pid].error for pid in point_ids_sorted], dtype=np.float32)
        points_rgb = np.array([points3d_dict[pid].rgb for pid in point_ids_sorted], dtype=np.uint8)
        point3D_id_to_point3D_idx = {pid: idx for idx, pid in enumerate(point_ids_sorted)}
        point3D_id_to_images = {pid: points3d_dict[pid].track for pid in point_ids_sorted}
        name_to_image_id = {imdata[k].name: k for k in imdata}
        image_id_to_name = {v: k for k, v in name_to_image_id.items()}

        point_indices: Dict[str, List[int]] = dict()
        for point_id, data in point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[int(image_id)]
                point_idx = point3D_id_to_point3D_idx[int(point_id)]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            camera_model_name = cameras[camera_id].camera_type

            if camera_model_name != "OPENCV_FISHEYE":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camera_model_name == "OPENCV_FISHEYE":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camera_model_name)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        dense_depth_dir: Optional[str] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.dense_depth_dir = dense_depth_dir
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # Load dense depth map from .npy file if available
            if self.dense_depth_dir is not None:
                image_name = self.parser.image_names[index]
                # Try different naming conventions
                base_name = os.path.splitext(image_name)[0]
                depth_path = os.path.join(self.dense_depth_dir, f"{base_name}.npy")
                if os.path.exists(depth_path):
                    depth_map = np.load(depth_path)  # [H, W]
                    # Optionally resize to match image
                    if depth_map.shape[0] != image.shape[0] or depth_map.shape[1] != image.shape[1]:
                        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                    data["depth_map"] = torch.from_numpy(depth_map).float()
                else:
                    # Fallback to sparse SfM points if dense depth not found
                    data["depth_map"] = None

            # Also load sparse SfM points for comparison (if available)
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(image_name, np.array([], dtype=np.int64))
            if len(point_indices) > 0:
                points_world = self.parser.points[point_indices]
                points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
                depths = points_cam[:, 2]  # (M,)
                # filter out points outside the image
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
                data["points"] = torch.from_numpy(points).float()
                data["depths"] = torch.from_numpy(depths).float()
            else:
                # No sparse points available
                data["points"] = torch.zeros((0, 2), dtype=torch.float32)
                data["depths"] = torch.zeros((0,), dtype=torch.float32)

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
