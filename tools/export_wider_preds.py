"""
把 SCRFD-tiny 的 best.pt 在 WIDER val 上的检测结果导出为官方评估器可读的格式。

输出布局（与 wondervictor/WiderFace-Evaluation 一致）：
    <out-dir>/<event>/<image_stem>.txt
    每个 txt:
        第 1 行: <该图相对路径，形如 0--Parade/xxx.jpg>
        第 2 行: <检测框数量>
        之后每行: x1 y1 w h score      # 原图坐标系，x1 y1 为左上角，w h 为宽高

注意：
- 官方评估器的 image_eval() 会把 (x1,y1,x2,y2) 转成 (x1,y1,x2+w,y2+h)，
  因此这里必须写 **w/h 而不是 x2/y2**。
- 框必须映射回 **原图坐标系**（推理在 input_size 空间完成，需要反算 letterbox 缩放与填充）。
- 预处理必须与训练/评估一致：utils.dataset.resize()（等比缩放 + 居中补边）。
"""
import argparse
import os
import sys

import cv2
import numpy
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets import nn
from utils import util
from utils.dataset import resize as letterbox_resize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint，如 /runs/xxx/best.pt')
    ap.add_argument('--images-root', required=True, help='含 <event>/<image>.jpg 的目录')
    ap.add_argument('--out', required=True, help='预测 txt 输出根目录')
    ap.add_argument('--input-size', default=320, type=int)
    ap.add_argument('--conf-threshold', default=0.02, type=float,
                    help='与 NMS 默认一致的低阈值，评估器会自己做阈值扫描')
    ap.add_argument('--limit', default=0, type=int, help='调试：只处理前 N 张')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    params = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'utils', 'args.yaml'), errors='ignore'))
    params['face_anchors']['sizes'] = [s * args.input_size / 640
                                       for s in params['face_anchors']['sizes']]

    # 与 main.py --test 相同的加载方式
    model = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = model['model'].float()
    model.eval()
    model = model.to(args.device)

    nms = util.NMS(params, conf_threshold=args.conf_threshold)

    # 收集图片
    items = []
    for event in sorted(os.listdir(args.images_root)):
        ev_dir = os.path.join(args.images_root, event)
        if not os.path.isdir(ev_dir):
            continue
        for name in sorted(os.listdir(ev_dir)):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append((event, name, os.path.join(ev_dir, name)))
    if args.limit:
        items = items[:args.limit]
    print('images to process:', len(items))

    n_boxes = 0
    n_empty = 0
    for idx, (event, name, path) in enumerate(items):
        image = cv2.imread(path)
        if image is None:
            print('WARN cannot read', path, file=sys.stderr)
            continue
        h0, w0 = image.shape[:2]

        # 与 Dataset(augment=False) 完全一致的预处理
        resized, ratio, pad = letterbox_resize(image, args.input_size)
        sample = resized.transpose((2, 0, 1))[::-1]      # HWC->CHW, BGR->RGB
        sample = numpy.ascontiguousarray(sample, dtype='float32')
        sample = torch.from_numpy(sample).unsqueeze(0).to(args.device)

        with torch.no_grad():
            outputs = model(sample)
        outputs = nms(outputs)

        det = outputs[0][0]        # (N, 5): x1 y1 x2 y2 score
        det = det.float().cpu().numpy()

        # 反算到原图坐标
        if det.shape[0]:
            det[:, [0, 2]] -= pad[0]
            det[:, [1, 3]] -= pad[1]
            det[:, :4] /= ratio
            det[:, 0] = numpy.clip(det[:, 0], 0, w0 - 1)
            det[:, 1] = numpy.clip(det[:, 1], 0, h0 - 1)
            det[:, 2] = numpy.clip(det[:, 2], 0, w0 - 1)
            det[:, 3] = numpy.clip(det[:, 3], 0, h0 - 1)
            keep = (det[:, 2] > det[:, 0]) & (det[:, 3] > det[:, 1])
            det = det[keep]

        out_dir = os.path.join(args.out, event)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(name)[0]
        with open(os.path.join(out_dir, stem + '.txt'), 'w') as f:
            f.write('%s/%s\n' % (event, name))
            f.write('%d\n' % det.shape[0])
            for x1, y1, x2, y2, s in det:
                # 官方评估器期望 w/h
                f.write('%.4f %.4f %.4f %.4f %.6f\n' % (x1, y1, x2 - x1, y2 - y1, s))

        n_boxes += det.shape[0]
        if det.shape[0] == 0:
            n_empty += 1
        if (idx + 1) % 200 == 0:
            print('  processed %d/%d' % (idx + 1, len(items)))

    print('done. images=%d boxes=%d empty=%d' % (len(items), n_boxes, n_empty))


if __name__ == '__main__':
    main()
