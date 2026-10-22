# WIDER Face 官方评估器（Easy / Medium / Hard）

本目录把 wondervictor 的 WIDER Face 评估代码纳入本仓库，用于 SCRFD-tiny 的标准
AP 评估。原始评估器不在 SCRFD-tiny 上游仓库中，此处为引入并做最小适配。

## 来源与许可

| 项 | 值 |
| --- | --- |
| 上游仓库 | `wondervictor/WiderFace-Evaluation` |
| 代码文件 | `evaluation.py`（原作者 wondervictor，`tianhengcheng@gmail.com`） |
| IoU 实现来源 | Sergey Karayev 的 `box_overlaps.pyx`（Fast R-CNN，MIT） |
| 许可 | 上游未声明独立许可；`box_overlaps.pyx` 部分为 MIT。使用前请自行确认 |
| ground-truth | WIDER Face 官方 `wider_*_val.mat`（随 WIDER 数据集分发） |

## 相对上游的改动

全部改动在 `evaluation.py` 文件头与本说明中标注，摘要：

1. **`bbox_overlaps` 改为纯 numpy 实现**，替代 cython 扩展 `box_overlaps.pyx`。
   免去 cython 编译步骤和 `setup.py build_ext`。数值语义与上游完全一致（含 `+1` 像素约定），
   由 `tests/test_bbox_overlaps.py` 逐值比对验证。
2. **移除 `from IPython import embed`**（上游仅调试用，未实际调用）。
3. **`rstrip('.jpg')` 改为显式切片**。`rstrip` 按字符集剥离，对以 `.`/`j`/`p`/`g` 结尾的
   文件名会误伤。当前 WIDER val 的 3226 张图实测未触发（已逐名核对），属潜在错误修正。
4. **`evaluation()` 增加 `return_aps` 参数**，便于程序化调用。
5. **`tqdm` 改为可选依赖**。
6. **GT 读取使用 `scipy.io.loadmat`**，依赖已加入 `Dockerfile.scrfd-tiny`。

## ground_truth/ 文件校验

```
50243024b7d44a4a31147a2d91571530  wider_easy_val.mat    409036 B
1c2fc4ede1eaff32493e149280f5b6ff  wider_face_val.mat    397768 B
9f1bcb00e3e9f283428fc30922492f79  wider_hard_val.mat    424085 B
7df7bdfb2b4dd2af5c03ad11046910a5  wider_medium_val.mat  412725 B
```

## 用法

评估需要**预测结果目录**，格式为每个事件一个子目录、每张图一个 txt：

```
<pred-dir>/<event>/<image_stem>.txt
   第 1 行: <event>/<image>.jpg
   第 2 行: <检测框数量>
   后续每行: x1 y1 w h score          # 原图坐标系，w/h 是宽高不是右下角坐标
```

用本仓库脚本生成：

```bash
# 1) 导出预测（容器内，需 GPU）
python3 tools/export_wider_preds.py \
  --ckpt /runs/ft320-20261015/best.pt \
  --images-root /datasets/WIDERFace/images/val \
  --out /runs/ft320-20261015/wouts \
  --input-size 320

# 2) 计算 Easy/Medium/Hard AP
python3 tools/widerface_evaluation/evaluation.py \
  -p /runs/ft320-20261015/wouts \
  -g tools/widerface_evaluation/ground_truth
```

## 自测

```bash
python3 tools/widerface_evaluation/tests/test_bbox_overlaps.py
```

## 重要口径说明

- 本评估器输出的是 **IoU=0.5 单点**的 AP，与仓库 `main.py --test` 里的
  `mAP`（IoU 0.5:0.95 平均）**不是同一指标**，两者不能互相替代。
- WIDER val 的 `label.txt` 只有检测框、没有关键点，因此本评估**只评检测**，不评 5 点精度。
- README 的 `SCRFD-tiny(ours) 89.23 / 87.60 / 67.15` 是 **640 输入**的结果；
  本工作流默认 **320 输入**，两者必须在同一输入尺寸下比较才有意义。
