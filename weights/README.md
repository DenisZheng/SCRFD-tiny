# 训练产物与评估记录

本目录保存训练得到的权重与其评估数据，便于直接复现与交付。

## 目录

| 路径 | 内容 |
| --- | --- |
| `best.pt` / `last.pt` / `step.csv` / `*.png` | 仓库自带的 **640 基线**权重（上游发布，未改动） |
| `ft320-20261015/` | 2026-10-15 在 GTX 1060 上 320 微调 200 epoch 的产物 |
| `baseline320/` | 上面的基线权重在 **320 输入**下重新评估得到的曲线（同口径对照） |

## ft320-20261015

| 文件 | 说明 |
| --- | --- |
| `best.pt` | 正式交付权重，取自 epoch 156（配置内 mAP 最佳） |
| `last.pt` | epoch 200 权重，**与 best 不同**，交付以 best 为准 |
| `step.csv` | 逐 epoch 训练指标（200 行） |
| `eval-report.json` | 机器可读的评估结果与校验值 |

### 校验

```
11a1e9aaf12e4aa03ae296ff932e9fdaa639999f7af46020d60d59da8f346903  best.pt
f25f77ffca19d9193d2fc1bd4bea10980bfb32a37ed6682dcb6a796408ca207c  last.pt
```

### 评估结果（WIDER val，320 输入，IoU=0.5）

| 子集 | 基线 `../best.pt` | 微调 `best.pt` | 变化 |
| --- | --- | --- | --- |
| Easy | 85.30 | **86.36** | +1.06 |
| Medium | 75.17 | **78.90** | +3.73 |
| Hard | 36.20 | **41.77** | +5.56 |

详细报告见 [`docs/evaluation/320微调与WIDER标准评估-20261016.md`](../docs/evaluation/320微调与WIDER标准评估-20261016.md)。

### 复现加载

```python
import torch
ckpt = torch.load('weights/ft320-20261015/best.pt', map_location='cpu', weights_only=False)
model = ckpt['model'].float()      # checkpoint 保存的是整个 module（EMA 权重）
print('epoch:', ckpt['epoch'])     # 156
```

注意：该 checkpoint pickle 了整个 module，PyTorch 2.6+ 默认 `weights_only=True`
会拒绝加载，需显式传 `weights_only=False`；只应加载自己产出的可信文件。

## 使用边界

- **不要用 `mAP` 与 README 表格的 89.23 / 87.60 / 67.15 直接比较**：
  那是 640 输入 + 1200 epoch 从头训练，本产物是 320 输入 + 200 epoch 微调。
- **不要把 `step.csv` 的 `mAP`（IoU 0.5:0.95）与本目录的 AP（IoU 0.5）混用。**
- **关键点精度没有评估**：WIDER val 只有检测框标注，`kpt` loss 下降不等于关键点准确。
