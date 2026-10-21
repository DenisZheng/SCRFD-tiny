# SCRFD-tiny（DenisZheng fork）Docker 训练、评估与导出手册

**适用模型：** `DenisZheng/SCRFD-tiny`（fork 自 `jahongir7174/SCRFD-tiny`），单阶段人脸检测 + 5 点关键点，目标部署 ESP-DL（ESP32-S3/P4）
**训练镜像：** `scrfd-tiny:torch2.5.1-cu124`（见随附 `Dockerfile.scrfd-tiny`）
**正式工作目录：** 宿主机 `$HOME/scrfd-tiny-lab/runs/<run-name>`（容器内 `/runs/<run-name>`，经 `--save-dir` 指定）
**文档状态：** fork 改造已合并主线；CPU 侧基线已实测（见第 2 节）；**GPU 训练、正式评估与量化精度尚未实测**，文中凡属建议起点均已标注，不得当作实测结论引用
**最后更新：** 2026-10-21

本文只描述这条 Docker 主线。仓库 README 里的 Conda 流程（`conda create -n PyTorch python=3.11.11` 等）不再使用；不要在训练容器里临时改依赖后继续把它当作原始训练环境。ESP-PPQ 量化与上板对版属于部署阶段，见第 13–15 节的衔接说明，不在训练容器内完成最终验收。

## 0. 宿主机准备：Docker 与 NVIDIA Container Toolkit

训练机应先安装可正常工作的 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit。安装命令会随 Ubuntu/Debian 版本变化，不要照搬过期的软件源配置；按当前官方文档安装：

- Docker Engine：<https://docs.docker.com/engine/install/>
- NVIDIA Container Toolkit：<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

先确认宿主机驱动和 Docker 正常：

```bash
nvidia-smi
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

安装 NVIDIA Container Toolkit 后，为 Docker 配置 NVIDIA runtime 并重启服务：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker run --rm --gpus all ubuntu nvidia-smi
```

最后一条命令必须能在容器内显示 GPU、驱动版本和显存信息，之后才能构建训练镜像。本手册的基础镜像含 CUDA 12.4 runtime，**宿主机驱动需 ≥ 550**；驱动更旧时改用 CUDA 12.1 基础镜像（`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`）并同步修改 Dockerfile 与镜像名，不要混用。日常命令若不使用 `sudo`，可按 Docker 官方 post-install 文档配置 `docker` 用户组；该用户组具有接近 root 的宿主机权限，应限制成员范围并在重新登录后验证 `docker info`。

## 1. 已确认基线

### 1.1 主机与软件栈

| 项目 | 当前值 |
| --- | --- |
| GPU | NVIDIA RTX 4080 16 GB，Compute Capability 8.9 |
| CPU / 内存 | Intel i5-13600K / 16 GB |
| 基础镜像 | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| Python | `3.11`（随基础镜像） |
| PyTorch | `2.5.1+cu124` |
| torchvision | `0.20.1+cu124`（随基础镜像） |
| CUDA runtime | `12.4` |
| cuDNN | `9` |
| 追加依赖 | `opencv-python-headless`、`PyYAML`、`tqdm`、`thop`、`matplotlib`、`onnx`、`onnxruntime`（版本在构建时解析，构建后按第 16 节用 `pip freeze` 固化记录） |

代码侧已针对现代 PyTorch / NumPy 做过兼容修复并合并主线（`weights_only` 显式声明、ONNX 导出固定旧导出器、NumPy 2.0 的 `trapezoid` 兼容等），因此不要求把环境降回旧版本。**不要在复现时顺手升级基础镜像的大版本**；确需升级时先重跑第 5 节镜像验收与第 7 节探针。

### 1.2 训练参数

本 fork 有两条训练轨，参数口径不同，不要混用：

| 参数 | 微调轨（主线） | 从头训练轨 |
| --- | --- | --- |
| 初始化 | `--weights weights/best.pt`（仓库自带 640 基线） | 不传 `--weights` |
| 输入尺寸 | `--input-size 320`（默认） | 320 或 640（`--input-size 640`） |
| anchor sizes | `args.yaml` 640 基准 `[16, 64, 256]`，代码按 `input_size / 640` 自动缩放（320 时实际为 `[8, 32, 128]`），**不要手改 YAML 的 sizes** | 同左 |
| 单卡 batch | `--batch-size 32`（默认） | `--batch-size 32`（默认） |
| DataLoader worker | 训练 8、评估 4（代码写死，无 CLI 参数） | 同左 |
| 总周期 | `--epochs` 建议起点 **200（未实测）**，按验证 mAP 平台期调整 | `--epochs 1200`（默认，上游配方） |
| 学习率 | 建议把 `utils/args.yaml` 的 `max_lr` 由 `0.01` 改为 `0.001`、**属未实测建议起点**；改动必须留 `git diff` 记录（见 3.2） | `max_lr 0.01`、`min_lr 0.0001`、warmup 5 epoch（`args.yaml` 默认） |
| 保存目录 | `--save-dir /runs/ft320-<日期>`，不得使用默认 `./weights`（会覆盖自带基线） | `--save-dir /runs/scratch-<日期>` |

注意 `main.py` 中 weight decay 会随 `--batch-size` 按公式缩放（`params['weight_decay'] *= batch_size * world_size * accumulate / 32`），改 batch 不只是显存问题，会改变正则强度；改动后要在运行记录中注明。

### 1.3 路径与挂载

| 宿主机 | 容器内 | 权限/用途 |
| --- | --- | --- |
| `$HOME/scrfd-tiny-lab/SCRFD-tiny` | `/workspace/scrfd-tiny` | fork 源码（克隆） |
| `$HOME/scrfd-tiny-lab/datasets` | `/datasets` | 数据：`WIDERFace/`（转换后）、`wider_raw/`（原始下载） |
| `$HOME/scrfd-tiny-lab/runs` | `/runs` | 每次运行的 `--save-dir`（checkpoint、step.csv、曲线图、导出产物） |
| `$HOME/scrfd-tiny-lab/scrfd-tiny-esp` | `/esp` | ESP 适配包（`prepare_wider.py`、`export_onnx_raw.py`、`quantize_espdl.py`、`decode_ref.py`），只读 |

正式 checkpoint 在宿主机 `runs` 中持续保存，不依赖容器可写层。容器使用 `--rm` 退出不会删除宿主机产物。

## 2. 已实测基线（CPU 侧，合并主线前完成）

以下均为已执行并记录在案的结果，不是预期值；它们证明的是代码与权重链路，不证明 GPU 训练精度：

| 项目 | 实测结果 |
| --- | --- |
| fork 状态 | PR #1 已合并 master，merge commit `d561bdf`；改造含 `--weights` / `--data-dir` / `--save-dir` / `--export`、anchor 自动缩放、`load_weight` 计数校验 |
| 模型规模 | 参数量 `82,554`；320 输入约 113 MMACs |
| 基线权重加载 | `load_weight` 加载自带 `weights/best.pt`：`loaded 228/228 tensors (0 skipped)`，与 checkpoint 逐位一致 |
| 解码对版 | 官方测试图（esp-dl `test/human_face.jpg`）320 输入：得分 `0.749`，框 `[98, 59, 193, 189]`，5 点与官方 MSR+MNP 公布输出逐项吻合（数像素差） |
| ESP 量化冒烟 | raw-head ONNX 经 esp-ppq 量化为 `.espdl`（int8 / esp32s3）成功，140 KB，逐层误差最差 3.17%；**校准集为单图增强的冒烟校准，不代表精度** |

GPU 侧的训练 loss、验证 mAP、单 epoch 耗时目前均无实测值，首次正式运行后按第 10、11 节回填本文。

## 3. 准备源码、数据和输出目录

以下命令在训练宿主机执行。

### 3.1 源码

```bash
mkdir -p "$HOME/scrfd-tiny-lab"/{datasets,runs}
cd "$HOME/scrfd-tiny-lab"
git clone https://github.com/DenisZheng/SCRFD-tiny.git
cd SCRFD-tiny
git rev-parse HEAD        # 应为 d561bdf 或其后的主线提交
test -f main.py && test -f utils/args.yaml && test -f weights/best.pt
```

把 ESP 适配包（`scrfd-tiny-esp`，含转换与量化脚本）放到 `$HOME/scrfd-tiny-lab/scrfd-tiny-esp`。

训练启动前记录源码状态，避免以后无法解释模型来自哪份代码：

```bash
git status --short
git rev-parse HEAD
```

### 3.2 微调轨的 args.yaml 改动（仅微调轨，从头训练轨跳过）

微调轨建议降低 `max_lr`。该值在 `utils/args.yaml` 里、没有 CLI 参数，只能改文件；改动必须可审计：

```bash
cd "$HOME/scrfd-tiny-lab/SCRFD-tiny"
sed -i 's/^max_lr: .*/max_lr: 0.001000000000            # maximum learning rate (finetune)/' utils/args.yaml
git diff utils/args.yaml   # 输出保存到本次运行记录中
```

不要改 `face_anchors.sizes`（运行时自动缩放，手改会被缩放逻辑二次处理）。

### 3.3 数据：RetinaFace 版 WIDER Face（带 5 点标注）

本仓库训练标签要求每脸 20 个值（框 + 5 点），必须使用带关键点的标注版（RetinaFace 作者标注的 `label.txt`，即 YuNet/libfacedetection.train 同源数据），原版 WIDER 官方标注没有关键点、不能直接用。

下载（任选其一，链接为 2026-10-21 核实；失效时从 RetinaFace 仓库 README 更新）：

- 标注包（小）：Dropbox <https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0>，或百度网盘 <https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA>（提取码 fstq）；图像本体另从 WIDER Face 官网 <http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html> 下载 WIDER_train / WIDER_val
- 整理整包（图像+标注已分好 train/val）：Google Drive <https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS>，或百度网盘 <https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ>（提取码 ruck）

解压后摆成 `prepare_wider.py` 接受的任一布局：

```text
wider_raw/
  WIDER_train/images/...  WIDER_train/label.txt
  WIDER_val/images/...    WIDER_val/label.txt
# 或整包布局：
wider_raw/
  train/{images, label.txt}
  val/{images, label.txt}
```

转换成本仓库的布局（`images/{train,val}` + `labels/{train,val}`，每图一个同名 txt）：

```bash
cd "$HOME/scrfd-tiny-lab"
python3 scrfd-tiny-esp/prepare_wider.py \
  --wider-root datasets/wider_raw \
  --out datasets/WIDERFace \
  --splits train val

ls datasets/WIDERFace/images/train | wc -l    # 应为 12880
ls datasets/WIDERFace/images/val   | wc -l    # 应为 3226
ls datasets/WIDERFace/labels/train | wc -l    # 应与 train 图片数一致
```

把 `prepare_wider.py` 打印的人脸总数与跳过数记入运行记录；数量明显不符时先停下查数据，不要带病训练。磁盘预留：原始包 + 转换后数据 + 每次运行产物，建议 ≥ 30 GB 可用空间（`df -h "$HOME/scrfd-tiny-lab"` 确认）。

## 4. 构建训练镜像

将随附的 `Dockerfile.scrfd-tiny` 放到宿主机任意目录（构建上下文只需该文件本身）：

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir \
      opencv-python-headless PyYAML tqdm thop matplotlib onnx onnxruntime

WORKDIR /workspace/scrfd-tiny
```

```bash
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

docker build \
  --progress=plain \
  --file Dockerfile.scrfd-tiny \
  --tag scrfd-tiny:torch2.5.1-cu124 \
  .
```

检查镜像存在，并固化依赖清单：

```bash
docker image inspect scrfd-tiny:torch2.5.1-cu124 --format '{{.Id}} {{.Created}}'
docker run --rm scrfd-tiny:torch2.5.1-cu124 pip freeze > "$HOME/scrfd-tiny-lab/runs/image-pip-freeze.txt"
```

说明：使用 `opencv-python-headless` 是有意选择——训练、评估、导出都不需要 GUI；`main.py --demo` 里的 `cv2.imshow` 在容器内不可用，需要可视化时把结果图写到 `/runs` 后在宿主机查看，不要为此改回带 GUI 的 opencv 破坏环境一致性。

## 5. 镜像验收

### 5.1 Python、CUDA 与依赖导入

```bash
docker run --rm --gpus '"device=0"' \
  --ipc=host \
  scrfd-tiny:torch2.5.1-cu124 \
  python3 - <<'PY'
import torch, torchvision, cv2, yaml, tqdm, thop, matplotlib, onnx, onnxruntime
print('torch:', torch.__version__)
print('torch cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))
print('numpy:', __import__('numpy').__version__)

assert torch.__version__.startswith('2.5.1')
assert torch.cuda.is_available()
PY
```

### 5.2 CUDA 实际运算与仓库冒烟

只成功 import 不足以证明 CUDA 算子可用，再执行一次实际 GPU 运算，并跑仓库自身的加载冒烟（不需要数据集）：

```bash
docker run --rm --gpus '"device=0"' \
  --ipc=host \
  -v "$HOME/scrfd-tiny-lab/SCRFD-tiny:/workspace/scrfd-tiny" \
  scrfd-tiny:torch2.5.1-cu124 \
  python3 - <<'PY'
import sys, torch
sys.path.insert(0, '/workspace/scrfd-tiny')
from nets import nn
from utils import util
import yaml

x = torch.randn(256, 256, device='cuda')
y = x @ x
assert y.is_cuda and torch.isfinite(y).all()
print('cuda matmul ok')

params = yaml.safe_load(open('/workspace/scrfd-tiny/utils/args.yaml'))
params['face_anchors']['sizes'] = [s * 320 / 640 for s in params['face_anchors']['sizes']]
model = nn.version_n(params)
util.load_weight(model, '/workspace/scrfd-tiny/weights/best.pt')   # 必须打印 loaded 228/228 tensors (0 skipped)
model = model.cuda().eval()
with torch.no_grad():
    out = model(torch.zeros(1, 3, 320, 320, device='cuda'))
print('forward @320 on GPU ok')
PY
```

`load_weight` 若打印的不是 228/228 或直接抛 `no tensor ... matches`，说明权重文件或代码版本不对，先停下，不要开始训练。两项都通过后，才进入数据与训练验证。

## 6. 创建正式容器

建议先在宿主机创建 tmux 会话，避免 SSH 断开终止前台容器：

```bash
tmux new -s scrfd-tiny-training
```

在 tmux 中创建容器：

```bash
docker run --rm -it \
  --name scrfd-tiny-train \
  --gpus '"device=0"' \
  --ipc=host \
  -e PYTHONFAULTHANDLER=1 \
  -e TORCH_SHOW_CPP_STACKTRACES=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$HOME/scrfd-tiny-lab/SCRFD-tiny:/workspace/scrfd-tiny" \
  -v "$HOME/scrfd-tiny-lab/datasets:/datasets:ro" \
  -v "$HOME/scrfd-tiny-lab/runs:/runs" \
  -v "$HOME/scrfd-tiny-lab/scrfd-tiny-esp:/esp:ro" \
  scrfd-tiny:torch2.5.1-cu124 \
  bash
```

`--ipc=host` 必须保留：Docker 默认共享内存偏小，训练 DataLoader 开 8 个 worker 时可能触发 `Bus error`（宿主机内存只有 16 GB，同时留意宿主机内存占用，见第 9 节）。数据集以只读挂载，防止训练进程误写原始数据。

容器内执行：

```bash
cd /workspace/scrfd-tiny
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h /runs
ls /datasets/WIDERFace
```

需要离开但保持任务运行时按 `Ctrl+B`，再按 `D` 脱离 tmux。重新进入：

```bash
tmux attach -t scrfd-tiny-training
```

## 7. 配置、数据和真实 batch 探针

### 7.1 检查实际配置与数据

在容器内执行（不需要 GPU 参与数据构建）：

```bash
cd /workspace/scrfd-tiny
python3 - <<'PY'
import os, yaml
from utils.dataset import Dataset

params = yaml.safe_load(open('utils/args.yaml'))
sizes = [s * 320 / 640 for s in params['face_anchors']['sizes']]
print('yaml baseline sizes:', params['face_anchors']['sizes'])
print('effective sizes @320:', sizes)
assert sizes == [8.0, 32.0, 128.0]

root = '/datasets/WIDERFace'
for split in ('train', 'val'):
    imgs = sorted(os.listdir(f'{root}/images/{split}'))
    lbls = sorted(os.listdir(f'{root}/labels/{split}'))
    assert len(imgs) == len(lbls), (split, len(imgs), len(lbls))
    print(split, 'images:', len(imgs))

filenames = [f'{root}/images/train/' + f for f in sorted(os.listdir(f'{root}/images/train'))]
ds = Dataset(filenames, 320, params, augment=True)
print('train dataset length:', len(ds))
img, target = ds[0]
print('sample image tensor:', tuple(img.shape), 'targets:', tuple(target.shape))
PY
```

类名与导入路径以当前仓库为准（`utils/dataset.py` 的 `Dataset`，`main.py` 训练与评估均用它构建）；若仓库结构变化，先以 `main.py` 的实际 import 为准修正本节，不要新写一套数据读取逻辑。这一步能发现数据布局、标注格式问题，但不能替代真实训练步。

### 7.2 真实训练步探针（正式训练前必须做）

用正式命令的缩小版跑 1 个 epoch 到独立探针目录，确认数据加载、前向、loss、反向和存盘整条链路都通：

```bash
cd /workspace/scrfd-tiny
mkdir -p /runs/probe-ft320
set -o pipefail
python3 -u main.py --train \
  --data-dir /datasets/WIDERFace \
  --save-dir /runs/probe-ft320 \
  --weights weights/best.pt \
  --input-size 320 \
  --epochs 1 \
  2>&1 | tee /runs/probe-ft320/console.log
```

通过标准（缺一不可）：

- 日志出现 `load_weight: loaded 228/228 tensors from weights/best.pt (0 skipped)`（从头训练轨无此行，属正常）
- `step.csv` 已生成且 loss 列（box/cls/kpt）为有限数值、无 NaN
- 探针目录生成 `best.pt` / `last.pt`
- 无 traceback、CUDA OOM、Bus error

探针的 checkpoint 只证明链路可用，不是正式训练的初始化权重之外的任何东西；正式微调仍从 `weights/best.pt` 开始。探针目录可整目录删除。

## 8. 正式训练

容器内执行。微调轨（主线）：

```bash
cd /workspace/scrfd-tiny
run_dir=/runs/ft320-YYYYMMDD
mkdir -p "$run_dir"

set -o pipefail
python3 -u main.py --train \
  --data-dir /datasets/WIDERFace \
  --save-dir "$run_dir" \
  --weights weights/best.pt \
  --input-size 320 \
  --batch-size 32 \
  --epochs 200 \
  2>&1 | tee -a "$run_dir/train-console.log"

train_status=${PIPESTATUS[0]}
echo "TRAIN_EXIT_CODE=$train_status" | tee -a "$run_dir/train-console.log"
exit "$train_status"
```

从头训练轨：去掉 `--weights`、使用默认 `--epochs 1200` 与原始 `max_lr`（若 3.2 改过 args.yaml，先 `git checkout utils/args.yaml` 恢复），`run_dir` 换新目录。

关键边界：

- `--save-dir` 必须指向 `/runs/...` 下的新目录。默认 `./weights` 会覆盖仓库自带的基线 `best.pt`、`last.pt`、`step.csv` 与曲线图，不要使用。
- 训练结束时代码会自动对 best/last 执行 `disable_grad`（剥离梯度标记），这是正常收尾，不是报错。
- 正式训练不设置 `CUDA_LAUNCH_BLOCKING=1`，不使用调试器包住整个进程。
- 单卡直接 `python3 main.py` 即可（代码按 `WORLD_SIZE` 环境变量判断分布式，未设置时为单进程）；`main.sh` 的 `torch.distributed.launch` 是旧式多卡入口，本手册单卡流程不使用。

### 8.1 中断后继续：本仓库没有真正的 resume

必须明确：本仓库的 checkpoint 只保存 `{'epoch', 'model'}`（EMA 模型），**不保存 optimizer 与调度器状态**，没有 `--resume-from` 语义。中断后继续训练的唯一方式是：

```bash
python3 -u main.py --train \
  --data-dir /datasets/WIDERFace \
  --save-dir /runs/ft320-YYYYMMDD-part2 \
  --weights /runs/ft320-YYYYMMDD/last.pt \
  --input-size 320 \
  --batch-size 32 \
  --epochs <剩余预算> \
  2>&1 | tee -a /runs/ft320-YYYYMMDD-part2/train-console.log
```

这等价于「以 last.pt 为初始化再微调一段」：学习率调度从头开始（warmup 也会重跑），`step.csv` 是新文件，epoch 计数从 0 起。运行记录中必须注明这是接续段及其初始化权重路径，不要把它写成无缝 resume。

## 8.2 宿主机一条命令启动容器并开始训练

以下命令在宿主机执行。命令会创建后台 `tmux` 会话，由 `tmux` 启动 Docker 并直接执行训练；SSH 断开不会影响训练。日期替换为实际日期：

```bash
tmux new-session -d -s scrfd-tiny-ft320-YYYYMMDD "docker run --rm --name scrfd-tiny-train --gpus '\"device=0\"' --ipc=host -e PYTHONFAULTHANDLER=1 -e TORCH_SHOW_CPP_STACKTRACES=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -v \"$HOME/scrfd-tiny-lab/SCRFD-tiny:/workspace/scrfd-tiny\" -v \"$HOME/scrfd-tiny-lab/datasets:/datasets:ro\" -v \"$HOME/scrfd-tiny-lab/runs:/runs\" -v \"$HOME/scrfd-tiny-lab/scrfd-tiny-esp:/esp:ro\" scrfd-tiny:torch2.5.1-cu124 bash -lc 'cd /workspace/scrfd-tiny; run_dir=/runs/ft320-YYYYMMDD; mkdir -p \"\$run_dir\"; set -o pipefail; python3 -u main.py --train --data-dir /datasets/WIDERFace --save-dir \"\$run_dir\" --weights weights/best.pt --input-size 320 --batch-size 32 --epochs 200 2>&1 | tee -a \"\$run_dir/train-console.log\"; train_status=\${PIPESTATUS[0]}; echo TRAIN_EXIT_CODE=\$train_status | tee -a \"\$run_dir/train-console.log\"; exit \$train_status'"
```

参数替换完成后，可用以下命令查看会话和重新连接：

```bash
tmux ls
tmux attach -t scrfd-tiny-ft320-YYYYMMDD
```

退出查看但保持训练运行时按 `Ctrl+B`，再按 `D`。训练结束或异常退出后，使用第 9、10 节检查容器状态、日志中的 `TRAIN_EXIT_CODE` 和最新 checkpoint；`tmux` 会话存在本身不代表训练成功。

## 9. 训练状态监控

以下命令均为只读检查。

宿主机检查容器和 GPU：

```bash
docker ps --filter name=scrfd-tiny-train
nvidia-smi
free -h
```

容器内检查训练进程与进度（`step.csv` 每 epoch 一行，列为 epoch、box、cls、kpt、Recall、Precision、mAP@50、F1、mAP）：

```bash
docker exec scrfd-tiny-train sh -lc "ps -ef | grep '[m]ain.py'"
docker exec scrfd-tiny-train sh -lc "tail -n 5 /runs/ft320-YYYYMMDD/step.csv"
docker exec scrfd-tiny-train sh -lc "tail -n 80 /runs/ft320-YYYYMMDD/train-console.log"
ls -lh "$HOME/scrfd-tiny-lab/runs/ft320-YYYYMMDD"/*.pt
```

扫描常见致命错误：

```bash
grep -Ein \
  'traceback|segmentation fault|bus error|out of memory|cuda error|killed|nan' \
  "$HOME/scrfd-tiny-lab/runs/ft320-YYYYMMDD/train-console.log"
```

`grep` 没有输出只表示这些模式没有匹配，不能单独证明训练成功。宿主机内存只有 16 GB：训练时若 `free -h` 显示 swap 大量使用且 step 变慢，先考虑宿主机内存压力（8 个 DataLoader worker + 数据缓存），不要先改 batch 掩盖问题。

## 10. 完成判定与 checkpoint 固化

必须同时检查日志结尾、最终 checkpoint 和进程退出状态：

```bash
run_dir="$HOME/scrfd-tiny-lab/runs/ft320-YYYYMMDD"

test -f "$run_dir/best.pt"
stat -c '%n %s bytes %y' "$run_dir/best.pt" "$run_dir/last.pt"
tail -n 3 "$run_dir/step.csv"
grep 'TRAIN_EXIT_CODE' "$run_dir/train-console.log"
sha256sum "$run_dir/best.pt"
```

完成证据应包括：

1. `TRAIN_EXIT_CODE=0`，日志无新的 traceback / OOM 证据。
2. `step.csv` 行数与 `--epochs` 一致，最后一行 mAP 已记录。
3. `best.pt` 存在、可被 `load_weight` 以 228/228 加载（第 12 节评估会自然验证）。
4. `best.pt` 的 SHA-256、源码 commit（含 3.2 的 args.yaml diff）、镜像 ID、完整训练命令一并记入运行记录。

交付时以明确的 `best.pt` 及其 SHA-256 为准；`last.pt` 是最后一个 epoch 的模型，与 best 不是同一个东西，不要混用。

## 11. 单轮性能记录

本模型在 320 输入下计算量很小（约 113 MMACs），预期瓶颈在数据加载与小 kernel 启动开销而非算力；具体数字**必须实测后回填**，不要引用 640 官方 SCRFD 的耗时类比。

首次正式运行后，从 `step.csv` 与控制台日志统计：

- 每个 epoch 的墙钟时间（首个 epoch 含初始化与缓存，不计入稳定值）
- 峰值 GPU 显存（`nvidia-smi` 采样）
- 宿主机内存与 swap 峰值
- DataLoader 是否成为瓶颈（GPU 利用率长时间不满）

不要通过把 `--epochs` 改为 1 来定义性能基线；探针运行（7.2）的数据只用于链路验收。

## 12. 独立评估（仓库评估器）

训练完成后，用仓库自带评估在 val 集上出数（曲线图会写入 `--save-dir`，不要指向仓库 `./weights`）：

```bash
cd /workspace/scrfd-tiny
python3 -u main.py --test \
  --data-dir /datasets/WIDERFace \
  --save-dir /runs/ft320-YYYYMMDD \
  --input-size 320 \
  2>&1 | tee -a /runs/ft320-YYYYMMDD/test-console.log
```

评估完成后应保存：

- `best.pt` 的 SHA-256 与源码 commit
- 评估命令完整行与控制台输出（mAP@50、mAP、F1、Recall、Precision）
- `save-dir` 中的 `PR_curve.png`、`F1_curve.png`、`P_curve.png`、`R_curve.png`

口径说明（重要）：

- 这里的 mAP 是仓库 `compute_ap` 在 val 集上的 IoU 口径结果，与 README 结果表（640 输入、上游口径）不可直接比较；与官方 WIDER Face 评估器（Easy/Medium/Hard）也不是同一套工具，需要标准成绩时另行接官方评估器。
- 320 输入下小脸会小于最小 anchor（8 px），Hard 子集类场景预期明显弱于 640；判断微调是否有效，只和**同为 320 的基线**比：先用自带 `weights/best.pt` 按同一命令跑一次 `--test`（`--save-dir` 指向一个临时目录），得到基线数，再与微调后对比。

## 13. ONNX 导出（仓库通道）

与 InsightFace 流程不同，本仓库的训练镜像已包含 `onnx` / `onnxruntime`（第 4 节），导出不需要另建转换镜像，也不要在训练容器里临时装包改变环境。导出走 fork 已合并的 `--export` 入口（固定形状、无动态轴）：

```bash
cd /workspace/scrfd-tiny
python3 main.py --export \
  --save-dir /runs/ft320-YYYYMMDD \
  --input-size 320
ls -lh /runs/ft320-YYYYMMDD/last.onnx /runs/ft320-YYYYMMDD/best.model
sha256sum /runs/ft320-YYYYMMDD/last.onnx
```

产物说明：

- `last.onnx`：opset 11，固定输入 `[1, 3, 320, 320]`，9 个输出（score/bbox/kps × 3 个 stride），含模型内的 sigmoid 等后处理头，是仓库格式的通用 ONNX。
- `best.model`：同模型的 TorchScript 版。
- **这份 ONNX 不是 ESP-PPQ 的输入**：ESP 量化要用第 14 节的 raw-head 导出（无 sigmoid/flatten 的原始头），两者不要混淆。

## 14. ESP-PPQ 量化衔接（部署阶段概要）

在宿主机或同一镜像容器中，用 ESP 适配包的脚本完成（量化环境 esp-ppq 的安装不在本手册范围内，按适配包 README 执行）：

```bash
# 1) raw-head ONNX（以正式 best.pt 为输入，不要再用自带基线权重）
python3 /esp/export_onnx_raw.py \
  --ckpt /runs/ft320-YYYYMMDD/best.pt \
  --size 320

# 2) 备校准集：从 val 集挑数百张含人脸图片放进 /runs/calib320/（脚本只读图片文件）
# 3) 量化（--calib-dir 指向上一步的正式校准集，不要省略——省略是单图冒烟模式，精度不作数）
python3 /esp/quantize_espdl.py \
  --onnx /esp/models/scrfd_tiny_320_raw.onnx \
  --target esp32s3 \
  --calib-dir /runs/calib320 \
  --calib-steps 200
```

冻结口径（改任一项必须全链路同步，包括板端 C++ 后处理器）：输入 320×320、letterbox 黑边填充、RGB、像素值 0–255 不归一化；anchor 中心在格点上（offset=0）；score 阈值 0.5；NMS IoU 0.45。板端后处理与 C++ 草稿在适配包 `cpp/` 中，上板前须在 ESP-IDF 工程里实际编译并与 `decode_ref.py` 逐项对版（该草稿尚未上板验证）。

## 15. ONNX 结构与数值验收

### 15.1 ONNX checker 与形状

```bash
python3 - <<'PY'
import onnx

path = '/runs/ft320-YYYYMMDD/last.onnx'
model = onnx.load(path)
onnx.checker.check_model(model)
inp = model.graph.input[0]
dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
print('input dims:', dims)
print('outputs:', [o.name for o in model.graph.output])
assert dims == [1, 3, 320, 320]
assert len(model.graph.output) == 9
PY
```

### 15.2 ONNX Runtime CPU 冒烟测试

```bash
python3 - <<'PY'
import numpy as np
import onnxruntime as ort

path = '/runs/ft320-YYYYMMDD/last.onnx'
session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
model_input = session.get_inputs()[0]
x = np.random.randn(1, 3, 320, 320).astype(np.float32)
outputs = session.run(None, {model_input.name: x})

for value in outputs:
    print(value.shape, value.dtype, np.isfinite(value).all())
assert len(outputs) == 9
assert all(np.isfinite(v).all() for v in outputs)
PY
```

随机输入测试只能证明图可加载、可执行且输出有限，不能证明预处理、后处理或检测结果正确。

### 15.3 PyTorch 与 ONNX 数值对照

必须做数值对照，并满足以下条件：

1. 使用同一张真实图片和完全相同的预处理（本节指仓库通道：与 `demo` 一致的读图与缩放）。
2. PyTorch 侧用 `best.pt` 的模型、ONNX 侧用同一输入，按输出名称逐个对齐 9 个输出。
3. 为每个输出记录 shape、dtype、最大绝对误差与平均绝对误差；阈值按部署精度要求与实测分布确定，不能只检查「程序没报错」。

### 15.4 解码级验收（基线复现，量化前后各做一次）

用适配包的 `decode_ref.py` 对官方测试图（esp-dl `test/human_face.jpg`）跑 raw-head ONNX 解码。自带基线权重在 320 下的已实测基准是：

| 项目 | 基准值（自带 best.pt，320 输入） |
| --- | --- |
| 最高得分 | `0.749` |
| 框 | `[98, 59, 193, 189]` |
| 5 点 | 左眼 `(114, 115)`、右眼 `(158, 114)`、鼻 `(130, 142)`、左嘴角 `(120, 159)`、右嘴角 `(154, 159)` |

- 换成微调后的 `best.pt` 时数值会变，这是正常的：验收标准是**同一权重的 PyTorch 解码与 ONNX 解码一致**（框/点逐项误差在像素级内），而不是等于上表。
- int8 量化后用同一流程对 `.espdl` 的等效输出复测，记录框/点漂移量；漂移明显时先查校准集质量，不要先动阈值。

## 16. 交付物

一次可复现交付至少包含：

```text
runs/ft320-YYYYMMDD/best.pt            # 正式 checkpoint（附 SHA-256）
runs/ft320-YYYYMMDD/step.csv
runs/ft320-YYYYMMDD/train-console.log
runs/ft320-YYYYMMDD/test-console.log    # 第 12 节评估后
runs/ft320-YYYYMMDD/{PR,F1,P,R}_curve.png
runs/ft320-YYYYMMDD/last.onnx           # 第 13 节导出后
runs/image-pip-freeze.txt
```

同时记录：

- 源码 commit 与 `git status`（含 3.2 的 args.yaml 改动 diff）
- 训练镜像 ID（`docker image inspect` 输出）
- 完整训练命令、起止时间、`TRAIN_EXIT_CODE`
- 数据来源与 `prepare_wider.py` 的转换统计（图片数/人脸数）
- 320 基线与微调后的评估数字（同口径对比）
- ONNX checker、ORT 冒烟、数值对照与解码验收结果
- 量化产物的校准集构成与 `.espdl` 大小（第 14 节执行后）

## 17. 常见故障速查

| 现象 | 先查什么 | 处理 |
| --- | --- | --- |
| 启动即 `load_weight: no tensor ... matches` 或计数不是 228/228 | `--weights` 路径指错文件，或代码版本与权重不匹配 | 停训核对路径与 commit；不要忽略计数硬跑（会变成从头训练而不自知） |
| DataLoader `Bus error` / worker 被杀 | 是否漏了 `--ipc=host`；宿主机内存（16 GB）是否被占满 | 补 `--ipc=host`；关掉宿主机其他大内存进程；仍不行再考虑减少并发，不要先改 batch |
| CUDA 报错 `driver version is insufficient` | 宿主机驱动 < 550 与 CUDA 12.4 镜像不匹配 | 升级驱动，或改用 cuda12.1 基础镜像并重建（第 0 节） |
| 显存不足 OOM | 本模型 320 输入下不应发生；先确认没有其他进程占卡 | `nvidia-smi` 查占用；确认 `--input-size` 没误传大尺寸 |
| 训练 loss 出现 NaN | 微调时 `max_lr` 是否忘了按 3.2 调低；数据标注是否转换错误 | 查 args.yaml diff 与 `prepare_wider.py` 统计；用 7.1 探针抽查标签图叠加 |
| `--test` 曲线图没出现在运行目录 | `--save-dir` 是否与训练一致 | 曲线图写 `--save-dir`；仓库 `./weights` 里的同名文件是自带基线，不要拿来当本次结果 |
| `main.py --demo` 显示窗口失败 | 容器使用 headless opencv，无 GUI | 预期行为；把标注图写到 `/runs` 在宿主机查看（第 4 节说明） |
| 导出 ONNX 输入维度不是 `[1, 3, 320, 320]` | 是否误用了 `util.export(dynamic=True)` 的旧路径 | 用第 13 节的 `--export` 入口重导 |

出现 native 崩溃（SIGSEGV 等）时，先保存报错时间、退出码、完整日志和最近 checkpoint，不要覆盖正式 run 目录，也不要先改 batch/worker 掩盖问题；在独立探针目录用相同参数复现，再缩小范围定位。

## 18. 推荐执行顺序

1. 宿主机 Docker / 驱动 / Container Toolkit 就绪（第 0 节），构建镜像并通过验收（第 4、5 节）。
2. 准备数据并完成转换，核对图片/标注计数（第 3.3 节）。
3. 跑真实训练步探针（第 7.2 节），通过后再启动正式微调（第 8 节）。
4. 按第 10 节固化 checkpoint 与运行记录，回填第 11 节性能数字。
5. 同口径评估：先测自带基线、再测微调模型（第 12 节），确认微调收益后再继续。
6. 仓库通道导出 ONNX 并完成第 15 节验收。
7. 进入 ESP-PPQ 正式校准量化（第 14 节）与上板对版；量化精度与板端延迟实测后，再下「平替 MSR+MNP」的结论。

当前最直接的下一步是第 2、3 步：把数据下好转好、探针跑通，训练本身反而是最不该抢跑的一步。
