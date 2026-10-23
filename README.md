Tiny version of [SCRFD](https://arxiv.org/abs/2105.04714) Face and 5 Point Landmark Detection implementation using PyTorch

**English** | [简体中文](README.zh-CN.md)

### Installation

Current workflow: use the [Docker training manual](SCRFD-tiny-Docker%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md) and `Dockerfile.scrfd-tiny` (PyTorch 2.5.1 / CUDA 12.4). The Conda commands below are retained as upstream history and are not the current training setup.

The owner has two registered training hosts: GTX 1060 6 GB (Pascal, sm_61) and RTX 4080 16 GB. The 2026-10-21 task selects the GTX 1060; the RTX 4080 remains available. The candidate image `scrfd-tiny:torch2.5.1-cu124` was validated on the GTX 1060 on 2026-10-15 (CUDA matmul, cuDNN conv forward/backward, 228/228 weight load, 320 model forward all PASS); see section 19 of the Docker training manual for the measured record.

```
conda create -n PyTorch python=3.11.11
conda activate PyTorch
conda install python=3.11.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install PyYAML
pip install tqdm
pip install thop
```

### Docker training

A full Docker-based workflow (image build and acceptance, RetinaFace-WIDER data preparation, 320 finetuning, evaluation, fixed-shape export for ESP-PPQ) is documented in the [Docker training manual](SCRFD-tiny-Docker%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md) (Chinese), with the image defined by [Dockerfile.scrfd-tiny](Dockerfile.scrfd-tiny).

The external `scrfd-tiny-esp` adapter used for data conversion and raw-head ESP export is not included in this checkout; its source/version must be recorded before those steps can be reproduced. This face model is independent of the ESPDet-Pico hand-gesture project.

### Train

* Dataset root is set with `--data-dir` (default `../Dataset/WIDERFace`); checkpoints and `step.csv` go to `--save-dir` (default `./weights`) - point it elsewhere to keep the bundled baseline untouched. Finetune from a checkpoint with `--weights <ckpt.pt>`. Note the defaults are a from-scratch recipe (`--epochs 1200`, `max_lr 0.01` in `utils/args.yaml`): for finetuning use far fewer epochs and consider lowering `max_lr`
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Run `python main.py --test` for testing (loads `best.pt` from `--save-dir`)
* Run `python main.py --export` to export `best.pt` from `--save-dir` as fixed-shape `[1, 3, S, S]` TorchScript + ONNX with no dynamic axes, for ESP-PPQ (S = `--input-size`)

### Results

#### Upstream reference (640 input)

Upstream numbers at 640 input. This fork defaults to 320 (`--input-size`, anchor sizes auto-scale);
expect lower AP at 320, especially on small faces. **Do not compare 320 runs against this table.**

| Model              | AP-easy | AP-medium | AP-hard | Params (M) | Params Ratio | FLOPS (M) | CPU ONNX Latency (ms) | 
|--------------------|---------|-----------|---------|------------|--------------|-----------|-----------------------|
| SCRFD0.5(ICLR2022) | 90.57   | 88.12     | 68.51   | 0.57       | 1.00x        | 508       | 5.5                   | 
| SCRFD-tiny(ours)   | 89.23   | 87.60     | 67.15   | 0.08       | 7.12x        | 451       | 4.4                   | 

#### This fork, measured (320 input, WIDER val)

WIDER Face val, IoU=0.5, standard evaluator (`tools/widerface_evaluation/`).
Same 320 input and same val set for both rows, so the comparison is like-for-like.

| Model | Init | Epochs | AP-easy | AP-medium | AP-hard | Weights |
|-------|------|--------|---------|-----------|---------|---------|
| SCRFD-tiny @320 (baseline) | upstream 640 ckpt | — | 85.30 | 75.17 | 36.20 | `weights/best.pt` |
| **SCRFD-tiny @320 (finetuned)** | `weights/best.pt` | 200 | **86.36** | **78.90** | **41.77** | [`weights/ft320-20261015/best.pt`](weights/ft320-20261015/best.pt) |
| Δ | | | +1.06 | +3.73 | +5.56 | |

Finetune: GTX 1060 6GB, 200 epochs, batch 32, `max_lr` 0.001, 320 input, ~10.7 h.
Gains concentrate on Medium/Hard, as expected since small faces are the main bottleneck at 320.
Full report: [`docs/evaluation/320微调与WIDER标准评估-20261016.md`](docs/evaluation/320微调与WIDER标准评估-20261016.md).

#### Comparison with ESP-DL MSR+MNP (scale-aligned)

WIDER val contains mostly tiny faces (69% of them fall below one stride-8 cell at 320 input),
so its absolute mAP is far below what a business validation set yields. To make a
meaningful comparison against ESP-DL's `MSR+MNP` (reported **0.367 mAP50-95** on an
undisclosed custom val set), the val set is filtered to faces MSR+MNP can actually detect
(≥102.4 px wide, derived from its 160-wide input and 16 px minimum anchor):

| Evaluation set | Model | mAP50-95 | AP50 |
|---|---|---|---|
| Scale-aligned subset (957 img / 1253 faces) | **finetuned @320** | **0.666** | **0.960** |
| Scale-aligned subset | baseline @320 | 0.653 | 0.954 |
| Full WIDER val (3226 img / 39697 faces) | finetuned @320 | 0.165 | 0.330 |
| ESP-DL MSR+MNP (custom val, undisclosed) | — | 0.367 | — |

The same checkpoint scores 0.165 on full WIDER val and 0.666 on the scale-aligned subset —
a 4.0x difference caused purely by face-size distribution. **At deployment scale the
finetuned model is roughly 1.8x the reported MSR+MNP number.**

Caveats: the subset is small (1253 faces), ESP-DL's custom val set is unknown, and our
number is fp32 PyTorch while MSR+MNP is int8. Treat this as an order-of-magnitude
sanity check, not a strict same-set benchmark. Full analysis:
[`docs/evaluation/与ESP-DL-MSRMNP尺度对齐对比-20261016.md`](docs/evaluation/与ESP-DL-MSRMNP尺度对齐对比-20261016.md).

> Two different metrics are in play: the tables above are **IoU=0.5 single-point AP** for
> Easy/Medium/Hard, while `mAP50-95` is the COCO-style average. `step.csv` logs the latter
> (~0.167 for the same checkpoint). They are not interchangeable. Keypoint accuracy is
> **not** evaluated here - WIDER val has box labels only.

### Dataset

The dataset is not bundled. Download links, checksums and expected conversion
counts are documented in [`docs/数据集来源与校验.md`](docs/数据集来源与校验.md).
You need two parts: the RetinaFace 5-point annotation pack (`retinaface_gt_v1.1`)
and the WIDER Face images.

### Dataset structure

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

#### Reference

* https://github.com/jahongir7174/YUNet-eval
* https://github.com/ShiqiYu/libfacedetection.train
* https://github.com/deepinsight/insightface/tree/master/detection/scrfd

### License

MIT (see [LICENSE](LICENSE)). Note: the upstream repository (jahongir7174/SCRFD-tiny) ships no license file; this license covers this fork and its modifications.
