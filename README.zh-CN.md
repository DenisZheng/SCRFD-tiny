基于 PyTorch 的 [SCRFD](https://arxiv.org/abs/2105.04714) 微型版实现：人脸与 5 点关键点检测

[English](README.md) | **简体中文**

### 安装

```
conda create -n PyTorch python=3.11.11
conda activate PyTorch
conda install python=3.11.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install PyYAML
pip install tqdm
pip install thop
```

### Docker 训练

完整的 Docker 训练流程（镜像构建与验收、RetinaFace 版 WIDER 数据准备、320 微调、评估、面向 ESP-PPQ 的固定形状导出）见 [Docker 训练手册](SCRFD-tiny-Docker%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md)，镜像定义见 [Dockerfile.scrfd-tiny](Dockerfile.scrfd-tiny)。

候选镜像 `scrfd-tiny:torch2.5.1-cu124` 已于 2026-10-15 在 GTX 1060 实机验收通过（CUDA 矩阵运算、cuDNN 卷积前向/反向、228/228 权重加载、320 模型前向均 PASS）；实测记录见 Docker 训练手册第 19 节。

### 训练

* 数据集根目录用 `--data-dir` 指定（默认 `../Dataset/WIDERFace`）；checkpoint 与 `step.csv` 写入 `--save-dir`（默认 `./weights`）——指向其他目录可避免覆盖仓库自带基线。用 `--weights <ckpt.pt>` 从 checkpoint 微调。注意默认参数是从头训练的配方（`--epochs 1200`、`utils/args.yaml` 中 `max_lr 0.01`）：微调时应大幅减少 epochs，并考虑调低 `max_lr`
* 运行 `bash main.sh $ --train` 训练，`$` 为 GPU 数量

### 测试

* 运行 `python main.py --test` 测试（从 `--save-dir` 加载 `best.pt`）
* 运行 `python main.py --export` 把 `--save-dir` 中的 `best.pt` 导出为固定形状 `[1, 3, S, S]`、无动态轴的 TorchScript + ONNX，供 ESP-PPQ 使用（S = `--input-size`）

### 结果

#### 上游参考（640 输入）

下表是上游在 640 输入下的数字。本 fork 默认为 320（`--input-size`，anchor 尺寸自动缩放）；
320 下 AP 会更低，小脸尤其明显——**不要拿 320 的结果与下表对比**。

| Model              | AP-easy | AP-medium | AP-hard | Params (M) | Params Ratio | FLOPS (M) | CPU ONNX Latency (ms) | 
|--------------------|---------|-----------|---------|------------|--------------|-----------|-----------------------|
| SCRFD0.5(ICLR2022) | 90.57   | 88.12     | 68.51   | 0.57       | 1.00x        | 508       | 5.5                   | 
| SCRFD-tiny(ours)   | 89.23   | 87.60     | 67.15   | 0.08       | 7.12x        | 451       | 4.4                   | 

#### 本 fork 实测（320 输入）

WIDER Face val，IoU=0.5，标准评估器（`tools/widerface_evaluation/`）。
两行同为 320 输入、同一 val 集，可直接对比。

| 模型 | 初始化 | Epochs | AP-easy | AP-medium | AP-hard | 权重 |
|---|---|---|---|---|---|---|
| SCRFD-tiny @320（基线） | 上游 640 ckpt | — | 85.30 | 75.17 | 36.20 | `weights/best.pt` |
| **SCRFD-tiny @320（微调后）** | `weights/best.pt` | 200 | **86.36** | **78.90** | **41.77** | [`weights/ft320-20261015/best.pt`](weights/ft320-20261015/best.pt) |
| 变化 | | | +1.06 | +3.73 | +5.56 | |

微调条件：GTX 1060 6GB，200 epoch，batch 32，`max_lr` 0.001，320 输入，约 10.7 小时。
提升集中在 Medium/Hard，符合 320 输入下小脸为主要瓶颈的预期。
完整报告：[`docs/evaluation/320微调与WIDER标准评估-20261016.md`](docs/evaluation/320微调与WIDER标准评估-20261016.md)。

> 两个不同指标：上表是 **IoU=0.5 单点 AP**（分 Easy/Medium/Hard）；
> `step.csv` 里记录的是 **IoU 0.5:0.95 平均值 mAP**，同一 checkpoint 约 0.167。
> 两者不可互换。关键点精度**未评估**——WIDER val 只有检测框标注。

### 数据集

数据集不入库。下载地址、校验值与预期转换结果见
[`docs/数据集来源与校验.md`](docs/数据集来源与校验.md)。
需要两部分：RetinaFace 5 点标注包（`retinaface_gt_v1.1`）与 WIDER Face 图像。

### 数据集结构

    ├── WIDERFace 
        ├── images
            ├── train
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train
                ├── 1111.txt
                ├── 2222.txt
            ├── val
                ├── 1111.txt
                ├── 2222.txt

#### 参考

* https://github.com/jahongir7174/YUNet-eval
* https://github.com/ShiqiYu/libfacedetection.train
* https://github.com/deepinsight/insightface/tree/master/detection/scrfd

### 许可证

MIT（见 [LICENSE](LICENSE)）。说明：上游仓库（jahongir7174/SCRFD-tiny）本身没有协议文件；本协议覆盖的是本 fork 及其修改。
