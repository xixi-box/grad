# PRD: DUSt3R to 3DGS Pipeline 展示网站

## 1. 项目概述

### 1.1 项目背景
本项目是一个将 DUSt3R 深度估计模型输出转换为 3D Gaussian Splatting 格式的工具链。需要为 GitHub 仓库创建一个专业的展示网站，提升项目的可发现性和专业度。

### 1.2 项目目标
- 展示项目核心功能和技术亮点
- 提供清晰的使用指南和快速开始入口
- 展示项目效果（输入/输出对比）
- 引导用户 Star、Fork 和使用项目

### 1.3 目标用户
- 计算机视觉/3D重建领域的研究人员
- 对 NeRF/3DGS 感兴趣的开发者
- 寻找深度估计解决方案的工程师

---

## 2. 页面结构

### 2.1 整体布局（单页滚动）

```
┌────────────────────────────────────────────────────────────┐
│                        NAVIGATION                           │
│  [Logo] DUSt3R → 3DGS    Features | Demo | Docs | [GitHub] │
├────────────────────────────────────────────────────────────┤
│                                                            │
│                        HERO SECTION                        │
│                                                            │
│              "Transform Images to 3D Gaussian              │
│                    Splatting in Minutes"                    │
│                                                            │
│            [Get Started] [View on GitHub]                  │
│                                                            │
│              [动态背景：粒子效果/3D模型旋转]                  │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│                     FEATURES SECTION                       │
│                                                            │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│   │ 📷      │  │ 🎯      │  │ 📦      │  │ ⚡      │      │
│   │ 深度估计 │  │ 相机位姿 │  │ COLMAP  │  │ 3DGS训练│      │
│   │         │  │         │  │         │  │         │      │
│   │ 自动从  │  │ 多视图  │  │ 标准格式 │  │ 实时渲染│      │
│   │ 图像估计 │  │ 一致对齐 │  │ 一键导出 │  │ 带剪枝  │      │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│                    WORKFLOW SECTION                        │
│                                                            │
│        ┌───────┐      ┌───────┐      ┌───────┐            │
│        │ 输入  │ ───▶ │DUSt3R │ ───▶ │COLMAP │            │
│        │ 图像  │      │ 推理  │      │ 导出  │            │
│        └───────┘      └───────┘      └───────┘            │
│                                         │                  │
│                                         ▼                  │
│                                    ┌───────┐              │
│                                    │ 3DGS  │              │
│                                    │ 训练  │              │
│                                    └───────┘              │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│                      DEMO SECTION                          │
│                                                            │
│   [Tab: 输入图像] [Tab: 点云] [Tab: 渲染结果]               │
│   ┌─────────────────────────────────────────────────────┐ │
│   │                                                     │ │
│   │                    图片/视频展示区                    │ │
│   │                                                     │ │
│   │              (支持轮播或切换查看)                     │ │
│   │                                                     │ │
│   └─────────────────────────────────────────────────────┘ │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│                    TECH STACK SECTION                      │
│                                                            │
│      [PyTorch]  [CUDA]  [gsplat]  [Open3D]  [DUSt3R]      │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│                  QUICK START SECTION                       │
│                                                            │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ # 安装依赖                                          │ │
│   │ pip install -r requirements.txt                     │ │
│   │                                                     │ │
│   │ # 运行 DUSt3R 并导出                                │ │
│   │ python dust3r_to_3dgs_verified_fixed.py \           │ │
│   │     --input ./data/input/ \                         │ │
│   │     --output ./data/raw/ \                          │ │
│   │     --coord_system original                         │ │
│   └─────────────────────────────────────────────────────┘ │
│                                      [📋 复制]              │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│                     FOOTER                                 │
│                                                            │
│   [GitHub] [文档] [问题反馈]                               │
│                                                            │
│   Made with ❤️ by xixi-box                                 │
│   © 2024 MIT License                                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 功能需求

### 3.1 Hero Section
| 元素 | 描述 |
|------|------|
| 标题 | "Transform Images to 3D Gaussian Splatting in Minutes" |
| 副标题 | "DUSt3R → COLMAP → 3DGS: A complete pipeline for 3D reconstruction" |
| CTA按钮 | 主按钮 "Get Started"（锚点到 Quick Start）、次按钮 "View on GitHub" |
| 背景 | 渐变色 + 动态粒子效果或 3D 模型旋转动画 |
| 动画 | 文字逐字淡入，按钮弹入 |

### 3.2 Features Section
展示 4 个核心功能卡片：

| 图标 | 标题 | 描述 |
|------|------|------|
| 📷 | Depth Estimation | 自动从图像估计深度图，无需额外标注 |
| 🎯 | Camera Pose | 多视图一致性优化，精确恢复相机位姿 |
| 📦 | COLMAP Export | 标准格式导出，兼容主流 3DGS 训练框架 |
| ⚡ | Real-time Rendering | 基于 gsplat 的实时渲染，支持剪枝优化 |

**交互**：
- 卡片 hover 时轻微上浮 + 阴影增强
- 滚动进入视口时依次渐入

### 3.3 Workflow Section
展示完整工作流程：

```
输入图像 → DUSt3R推理 → 全局对齐 → COLMAP导出 → 3DGS训练 → 实时渲染
```

**交互**：
- 流程节点依次高亮动画
- 连接线流动效果
- 点击节点显示详细说明

### 3.4 Demo Section
展示项目效果：

| Tab | 内容 |
|-----|------|
| 输入图像 | 24张输入图像网格展示 |
| 点云重建 | points3D.ply 可旋转 3D 预览 |
| 渲染结果 | 训练后的渲染视频/mp4 |

**交互**：
- Tab 切换动画
- 图片支持放大查看
- 3D 点云支持鼠标旋转/缩放

### 3.5 Tech Stack Section
技术栈图标展示：

| 技术 | 图标来源 |
|------|----------|
| PyTorch | 官方 logo |
| CUDA | NVIDIA logo |
| gsplat | nerfstudio logo |
| Open3D | 官方 logo |
| DUSt3R | NAVER logo 或自定义 |

### 3.6 Quick Start Section
代码块展示安装和运行命令：

```bash
# 1. 克隆仓库
git clone https://github.com/xixi-box/grad.git
cd grad

# 2. 安装依赖
pip install -r requirements.txt

# 3. 构建 DUSt3R
cd dust3r/croco/models/curope && python setup.py build_ext --inplace

# 4. 运行推理
python dust3r_to_3dgs_verified_fixed.py \
    --input ./data/input/ \
    --output ./data/raw/ \
    --coord_system original \
    --save_depth \
    --evaluate
```

**交互**：
- 代码高亮（bash 语法）
- 一键复制按钮
- 复制成功提示

### 3.7 Footer
- GitHub 仓库链接
- 文档链接（README.md）
- Issue 反馈链接
- 版权信息

---

## 4. 技术架构

### 4.1 技术选型

| 类型 | 技术 | 理由 |
|------|------|------|
| 框架 | React 18 | 组件化、生态丰富 |
| 构建 | Vite | 快速、现代 |
| 样式 | Tailwind CSS | 原子化、开发快 |
| 动画 | Framer Motion | React 动画库首选 |
| 图标 | Lucide React | 现代、轻量 |
| 代码高亮 | Prism.js / Highlight.js | 语法高亮 |
| 部署 | GitHub Pages | 免费、与仓库集成 |

### 4.2 项目结构

```
web/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── index.css
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── Hero.jsx
│   │   ├── Features.jsx
│   │   ├── Workflow.jsx
│   │   ├── Demo.jsx
│   │   ├── TechStack.jsx
│   │   ├── QuickStart.jsx
│   │   ├── Footer.jsx
│   │   └── ui/
│   │       ├── Button.jsx
│   │       ├── Card.jsx
│   │       └── CodeBlock.jsx
│   ├── assets/
│   │   ├── images/
│   │   │   ├── input/          # 输入图像示例
│   │   │   ├── output/         # 输出效果
│   │   │   └── logos/          # 技术栈 logo
│   │   └── videos/
│   │       └── demo.mp4        # 演示视频
│   └── hooks/
│       └── useScrollAnimation.js
└── public/
    └── favicon.ico
```

### 4.3 依赖列表

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.300.0",
    "prismjs": "^1.29.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "vite": "^5.0.0"
  }
}
```

---

## 5. 设计规范

### 5.1 色彩方案（深色主题）

| 用途 | 颜色 | Tailwind 类 |
|------|------|-------------|
| 背景（主） | #0a0a0f | bg-gray-950 |
| 背景（次） | #1a1a2e | bg-slate-900 |
| 强调色 | #8b5cf6 | text-violet-500 |
| 强调色（亮） | #a78bfa | text-violet-400 |
| 文字（主） | #f8fafc | text-slate-50 |
| 文字（次） | #94a3b8 | text-slate-400 |
| 边框 | #334155 | border-slate-700 |
| 成功 | #22c55e | text-green-500 |
| 渐变起点 | #7c3aed | from-violet-600 |
| 渐变终点 | #2563eb | to-blue-600 |

### 5.2 字体

| 类型 | 字体 | 备选 |
|------|------|------|
| 标题 | Inter | system-ui, sans-serif |
| 正文 | Inter | system-ui, sans-serif |
| 代码 | JetBrains Mono | Fira Code, monospace |

### 5.3 间距规范

| 元素 | 间距 |
|------|------|
| Section 垂直间距 | py-24 (96px) |
| 容器最大宽度 | max-w-7xl (1280px) |
| 卡片内边距 | p-6 (24px) |
| 元素间距 | gap-6 / gap-8 |

### 5.4 动画规范

| 场景 | 动画 | 时长 |
|------|------|------|
| 元素进入 | fadeUp | 0.6s |
| 按钮悬停 | scale(1.05) | 0.2s |
| 卡片悬停 | translateY(-4px) | 0.3s |
| Tab 切换 | opacity fade | 0.3s |
| 滚动触发 | stagger children | 每次 0.1s delay |

---

## 6. 响应式设计

### 6.1 断点

| 断点 | 宽度 | Tailwind 前缀 |
|------|------|---------------|
| 手机 | < 640px | 默认 |
| 平板 | ≥ 640px | sm: |
| 桌面 | ≥ 768px | md: |
| 大屏 | ≥ 1024px | lg: |
| 超大 | ≥ 1280px | xl: |

### 6.2 响应式适配

| 元素 | 手机 | 平板 | 桌面 |
|------|------|------|------|
| Hero 标题 | text-3xl | text-4xl | text-6xl |
| Features 卡片 | 1列 | 2列 | 4列 |
| 代码块 | 横向滚动 | 正常显示 | 正常显示 |
| 导航 | 汉堡菜单 | 汉堡菜单 | 完整导航 |

---

## 7. 交互设计

### 7.1 导航栏
- 固定在顶部
- 滚动时添加背景模糊效果
- 移动端汉堡菜单
- 平滑滚动到各 Section

### 7.2 滚动动画
- 使用 Intersection Observer 检测元素进入视口
- 元素进入时触发淡入动画
- 支持子元素依次进入（stagger）

### 7.3 代码块
- 语法高亮（bash）
- 行号显示
- 一键复制按钮
- 复制成功 toast 提示

### 7.4 Demo 展示
- Tab 切换（输入/输出/渲染）
- 图片懒加载
- 3D 点云：使用 three.js 或静态图片替代
- 视频自动播放（可选）

---

## 8. 内容规划

### 8.1 需要准备的素材

| 类型 | 内容 | 数量 |
|------|------|------|
| 输入图像 | data/input/ 中选取代表性图片 | 6-12张 |
| 点云渲染 | 从 points3D.ply 渲染的预览图 | 1-3张 |
| 渲染视频 | results/ 中的 traj_*.mp4 | 1个 |
| 技术栈 logo | 各技术的官方 logo | 5-6个 |

### 8.2 文案

**Hero 标题**:
> Transform Images to 3D Gaussian Splatting in Minutes

**Hero 副标题**:
> A complete pipeline from DUSt3R depth estimation to real-time 3D rendering

**CTA 文案**:
- 主按钮: Get Started
- 次按钮: View on GitHub

**Features 描述**:
- 深度估计: Automatically estimate depth maps from images with state-of-the-art DUSt3R model
- 相机位姿: Multi-view consistent camera pose estimation through global alignment
- COLMAP导出: Standard format export compatible with all major 3DGS training frameworks
- 实时渲染: Real-time rendering powered by gsplat with built-in pruning optimization

---

## 9. 部署方案

### 9.1 GitHub Pages 部署

**方式一：手动部署**
```bash
npm run build
# 将 dist/ 目录推送到 gh-pages 分支
```

**方式二：GitHub Actions 自动部署**
```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

### 9.2 域名配置

- GitHub Pages 默认域名: `https://xixi-box.github.io/grad/`
- 可配置自定义域名（可选）

---

## 10. SEO 优化

### 10.1 Meta 标签

```html
<title>DUSt3R to 3DGS Pipeline - 3D Gaussian Splatting Made Easy</title>
<meta name="description" content="Transform images to 3D Gaussian Splatting in minutes. Complete pipeline from DUSt3R depth estimation to real-time 3D rendering.">
<meta name="keywords" content="3DGS, Gaussian Splatting, DUSt3R, 3D reconstruction, NeRF, depth estimation">
```

### 10.2 Open Graph

```html
<meta property="og:title" content="DUSt3R to 3DGS Pipeline">
<meta property="og:description" content="Transform images to 3D Gaussian Splatting in minutes">
<meta property="og:image" content="/og-image.png">
<meta property="og:url" content="https://xixi-box.github.io/grad/">
<meta property="og:type" content="website">
```

---

## 11. 性能优化

| 优化项 | 方案 |
|--------|------|
| 图片 | WebP 格式，懒加载 |
| 代码 | 分包加载，Tree Shaking |
| 字体 | 子集化，预加载 |
| 动画 | 使用 transform，避免重绘 |

---

## 12. 验收标准

- [ ] 所有 Section 按设计实现
- [ ] 响应式布局在手机/平板/桌面正常
- [ ] 动画流畅，无卡顿
- [ ] 代码块可复制
- [ ] 所有链接正常跳转
- [ ] Lighthouse 性能评分 > 90
- [ ] GitHub Pages 部署成功
- [ ] README.md 添加网站链接

---

## 13. 后续迭代

### Phase 2（可选）
- 添加在线演示（WebGL 3D 查看器）
- 多语言支持（中/英）
- 添加更多示例场景
- 集成文档站点

### Phase 3（可选）
- 交互式参数调节演示
- 用户上传图片预览
- 与后端集成的完整演示

---

**文档版本**: v1.0
**创建日期**: 2024-04-06
**作者**: Claude
**状态**: 待开发