# DUSt3R to 3DGS Pipeline

将 DUSt3R (NAVER 深度估计模型) 输出转换为 3D Gaussian Splatting (3DGS) 格式，支持 COLMAP 标准导出与质量评估。

## 功能特性

- **DUSt3R 推理**: 从图像自动估计深度和相机位姿
- **全局对齐**: 多视图一致性优化
- **COLMAP 导出**: 标准 `cameras.bin` / `images.bin` / `points3D.bin` 格式
- **3DGS 训练**: 带剪枝的 Gaussian Splatting 训练器
- **质量评估**: 输入验证与重建质量报告
- **实时可视化**: gsplat 查看器支持

## 环境要求

- Python 3.10+
- CUDA 12.1+ (推荐 12.5+ 支持 RTX 50 系列)
- PyTorch 2.0+

## 安装

### 1. 安装 PyTorch (CUDA 版本)

```bash
# CUDA 12.5+ / 13.x (支持 RTX 50 系列)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 构建 DUSt3R

```bash
cd dust3r/croco/models/curope
python setup.py build_ext --inplace
cd ../../../..
```

### 4. 安装 gsplat

```bash
# 从源码编译安装
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
```

**RTX 50 系列 (sm_120 Blackwell) 注意：**

预编译的 gsplat wheel 包尚未支持 sm_120 架构，需要从源码编译（上面的命令即可）。

### 5. 下载模型权重

```bash
# 创建权重目录
mkdir -p checkpoints

# 下载 DUSt3R 模型
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -O checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

## 快速开始

### 1. 运行 DUSt3R 并导出 COLMAP 格式

```bash
python dust3r_to_3dgs_verified_fixed.py \
    --input ./data/input/ \
    --output ./data/raw/ \
    --coord_system original \
    --save_depth \
    --evaluate
```

**参数说明**:
- `--input`: 输入图像目录
- `--output`: 输出目录
- `--coord_system`: 坐标系统 (`dust3r` 或 `original`)
- `--save_depth`: 保存深度图为 `.npy` 文件
- `--evaluate`: 运行质量评估
- `--overwrite`: 覆盖已有输出

### 2. 训练 3DGS

```bash
python simple_trainer_prune_v2.py \
    --data_dir ./data/raw \
    --exp_name exp_tau030 \
    --tau 0.03
```

### 3. 可视化结果

```bash
python gsplat_viewer.py --data_dir ./data/raw
```

## 项目结构

```
.
├── dust3r/                      # DUSt3R 子模块
├── data/
│   ├── input/                   # 输入图像
│   └── raw/                     # DUSt3R 输出
│       ├── images/              # 处理后的图像
│       ├── depths/              # 深度图 (.npy)
│       ├── sparse/0/            # COLMAP 格式
│       │   ├── cameras.bin
│       │   ├── images.bin
│       │   └── points3D.bin
│       └── reports/             # 质量评估报告
├── checkpoints/                 # 模型权重
├── dust3r_to_3dgs_verified_fixed.py  # 主入口脚本
├── simple_trainer_prune_v2.py   # 3DGS 训练器
├── gsplat_viewer.py             # 可视化工具
└── requirements.txt             # 依赖列表
```

## RTX 50 系列特别说明

RTX 50 系列 (Blackwell, sm_120) 需要从源码编译 gsplat：

**原因：**
- 预编译的 gsplat wheel 包尚未支持 sm_120 架构

**解决方案：**
```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
```

从源码编译后会自动支持 RTX 50 系列。

## 已知问题

| 问题 | 解决方案 |
|------|----------|
| `torch.cuda.amp.autocast` 弃用警告 | 使用 `torch.amp.autocast('cuda', ...)` |
| gsplat 预编译包不支持 RTX 50 系列 | 从源码编译即可 |
| 大文件超过 GitHub 限制 | 使用 Git LFS 或排除大文件 |

## 参考文献

- [DUSt3R: Geometric 3D Vision Made Easy](https://github.com/naver/dust3r)
- [gsplat: CUDA Accelerated Gaussian Splatting](https://github.com/nerfstudio-project/gsplat)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

本项目仅供学术研究使用。DUSt3R 和 gsplat 各自遵循其原始许可证。