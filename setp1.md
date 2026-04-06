## 完整实验方案

------

### A 组：模块消融（证明每个模块有贡献）

**A1 — Baseline（无深度约束，无剪枝）**

```bash
python simple_trainer_prune_v2.py default --data-dir ./data/raw --result-dir ./results/A1_baseline --data-factor 4 --opacity-reg 0.01 --strategy.refine-stop-iter 7000 --max-steps 25000 --eval-steps 25000 --save-steps 25000 --save-eval-images --disable-viewer --disable-video --disable-tb
```

**A2 — 有深度约束，无剪枝**

```bash
python simple_trainer_prune_v2.py default --data-dir ./data/raw --dense-depth-dir ./data/raw/depths --result-dir ./results/A2_depth_only --data-factor 4 --opacity-reg 0.01 --strategy.refine-stop-iter 7000 --depth-loss --depth-lambda 0.01 --max-steps 25000 --eval-steps 25000 --save-steps 25000 --save-eval-images --disable-viewer --disable-video --disable-tb
```

**A3 — 完整方法（深度约束 + 剪枝 tau=0.60）**

```bash
python simple_trainer_prune_v2.py default --data-dir ./data/raw --dense-depth-dir ./data/raw/depths --result-dir ./results/A3_full_tau060 --data-factor 4 --opacity-reg 0.01 --strategy.refine-stop-iter 7000 --enable-prune --prune-step-ratio 0.7 --prune-tau 0.60 --prune-lambda-d 0.5 --prune-lambda-v 0.3 --prune-lambda-n 0.2 --max-steps 25000 --eval-steps 25000 --save-steps 25000 --save-eval-images --disable-viewer --disable-video --disable-tb
```

**填表：**

| 实验            | 高斯数量 | PSNR | SSIM | LPIPS |
| --------------- | -------- | ---- | ---- | ----- |
| A1 无深度无剪枝 |          |      |      |       |
| A2 有深度无剪枝 |          |      |      |       |
| A3 完整方法     |          |      |      |       |

------

### B 组：tau 消融（证明剪枝可控）

**B1 — tau=0.30**

```bash
python simple_trainer_prune_v2.py default --data-dir ./data/raw --dense-depth-dir ./data/raw/depths --result-dir ./results/B1_tau030 --data-factor 4 --opacity-reg 0.01 --strategy.refine-stop-iter 7000 --enable-prune --prune-step-ratio 0.7 --prune-tau 0.30 --prune-lambda-d 0.5 --prune-lambda-v 0.3 --prune-lambda-n 0.2 --max-steps 25000 --eval-steps 25000 --save-steps 25000 --disable-viewer --disable-video --disable-tb
```

**B2 — tau=0.60（复用 A3 结果，不重跑）**

**B3 — tau=0.75**

```bash
python simple_trainer_prune_v2.py default --data-dir ./data/raw --dense-depth-dir ./data/raw/depths --result-dir ./results/B3_tau075 --data-factor 4 --opacity-reg 0.01 --strategy.refine-stop-iter 7000 --enable-prune --prune-step-ratio 0.7 --prune-tau 0.75 --prune-lambda-d 0.5 --prune-lambda-v 0.3 --prune-lambda-n 0.2 --max-steps 25000 --eval-steps 25000 --save-steps 25000 --disable-viewer --disable-video --disable-tb
```

**填表：**

| 实验        | 高斯数量 | 压缩比 | PSNR | SSIM | LPIPS |
| ----------- | -------- | ------ | ---- | ---- | ----- |
| A1 Baseline |          | —      |      |      |       |
| B1 tau=0.30 |          |        |      |      |       |
| A3 tau=0.60 |          |        |      |      |       |
| B3 tau=0.75 |          |        |      |      |       |

------

### C 组：评分项消融（证明三项都有贡献）

**C1 — 仅 D_i**

```bash
python simple_trainer_prune_v2.py default --data-dir ./data/raw --dense-depth-dir ./data/raw/depths --result-dir ./results/C1_only_D --data-factor 4 --opacity-reg 0.01 --strategy.refine-stop-iter 7000 --enable-prune --prune-step-ratio 0.7 --prune-tau 0.60 --prune-lambda-d 1.0 --prune-lambda-v 0.0 --prune-lambda-n 0.0 --max-steps 25000 --eval-steps 25000 --save-steps 25000 --disable-viewer --disable-video --disable-tb
```

**C2 — D_i + V_i**

```bash
python simple_trainer_prune_v2.py default --data-dir ./data/raw --dense-depth-dir ./data/raw/depths --result-dir ./results/C2_DV --data-factor 4 --opacity-reg 0.01 --strategy.refine-stop-iter 7000 --enable-prune --prune-step-ratio 0.7 --prune-tau 0.60 --prune-lambda-d 0.5 --prune-lambda-v 0.5 --prune-lambda-n 0.0 --max-steps 25000 --eval-steps 25000 --save-steps 25000 --disable-viewer --disable-video --disable-tb
```

**C3 — 完整方法（复用 A3 结果，不重跑）**

**填表：**

| 实验        | 高斯数量 | 压缩比 | PSNR | SSIM | LPIPS |
| ----------- | -------- | ------ | ---- | ---- | ----- |
| C1 仅 D_i   |          |        |      |      |       |
| C2 D_i+V_i  |          |        |      |      |       |
| A3 完整方法 |          |        |      |      |       |

------

### 总结

共需跑 **7 条命令**（A1、A2、A3、B1、B3、C1、C2），A3 结果复用三次。

**跑的顺序：** A1 → A3 → A2 → B1 → B3 → C1 → C2

A1 和 A3 最重要，跑完就有核心对比，其余是补充证据。





| 参数                          | 本地（验证用） | 云端（正式结果） |
| ----------------------------- | -------------- | ---------------- |
| `--data-factor`               | 4              | 2                |
| `--strategy.refine-stop-iter` | 2500           | 7000             |
| `--max-steps`                 | 5000           | 25000            |
| `--eval-steps`                | 5000           | 25000            |
| `--save-steps`                | 5000           | 25000            |

本地 factor 4 图更小，5000 步跑完快，主要目的是验证代码不报错、剪枝逻辑正常。云端 factor 2 分辨率更高、步数更多，出正式论文数据。