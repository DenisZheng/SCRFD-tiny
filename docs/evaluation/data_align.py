"""
统计 WIDER val 的人脸尺寸分布，并推算 MSR+MNP / SCRFD-tiny 的能力边界。

用途：支撑 docs/evaluation/与ESP-DL-MSRMNP尺度对齐对比-20261016.md 中的数据表。

关键事实来源：
- MSR+MNP 的 anchor 配置读自 esp-dl/models/human_face_detect/human_face_detect.cpp:34
      MSRPostprocessor(..., {{8, 8, 9, 9, {{16, 16}, {32, 32}}},
                             {16, 16, 9, 9, {{64, 64}, {128, 128}}}})
  格式 {stride_h, stride_w, ..., {{anchor_h, anchor_w}, ...}}
  => stride 8 配 anchor 16/32 px；stride 16 配 anchor 64/128 px
- MSR 输入固定 120x160（高 x 宽）

用法：
    python3 data_align.py <val_labels_dir> <val_images_dir>
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

MSR_IN_W = 160
MSR_IN_H = 120
MIN_ANCHOR = 16.0


def main():
    val_lbl = sys.argv[1] if len(sys.argv) > 1 else '/root/scrfd-tiny-lab/datasets/WIDERFace/labels/val'
    src_img = sys.argv[2] if len(sys.argv) > 2 else '/root/scrfd-lab/datasets/retinaface/val/images'

    events = sorted([d for d in os.listdir(src_img) if os.path.isdir(os.path.join(src_img, d))],
                    key=len, reverse=True)

    face_w, img_w, per_img = [], [], []
    for fn in sorted(os.listdir(val_lbl)):
        rows = []
        with open(os.path.join(val_lbl, fn)) as f:
            for line in f:
                v = line.split()
                if len(v) >= 5:
                    rows.append(float(v[3]) - float(v[1]))
        stem = fn[:-4]
        found = None
        for ev in events:
            if stem.startswith(ev + '_'):
                cand = os.path.join(src_img, ev, stem[len(ev) + 1:] + '.jpg')
                if os.path.isfile(cand):
                    found = cand
                    break
        if found is None:
            continue
        im = cv2.imread(found, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        img_w.append(im.shape[1])
        per_img.append(len(rows))
        face_w.extend(rows)

    face_w = np.array(face_w)
    img_w = np.array(img_w)

    print('=' * 76)
    print('WIDER val: %d 图 / %d 脸' % (len(img_w), len(face_w)))
    widths = Counter(img_w.tolist())
    print('图宽分布:', dict(widths))
    print('=' * 76)

    sc_msr = float(img_w[0]) / MSR_IN_W
    sc_ours = float(img_w[0]) / 320

    print('\n缩放比: MSR %.2f  |  我们的模型@320 %.2f' % (sc_msr, sc_ours))
    print('MSR 最小可检测脸（原图）= %.1f px' % (MIN_ANCHOR * sc_msr))
    print('我们的最小 anchor=8px（输入空间）=> 原图 %.1f px' % (8 * sc_ours))

    print('\n【人脸尺寸分布】')
    print('  %-12s %-10s %-14s %-14s' % ('原图宽', '占比', 'MSR输入空间', '我们@320输入空间'))
    for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]:
        m = (face_w >= lo) & (face_w < hi)
        if m.sum() == 0:
            continue
        lab = '%d-%d' % (lo, hi) if hi < 1e9 else '%d+' % lo
        a = '%.1f-%.1f' % (lo / sc_msr, hi / sc_msr) if hi < 1e9 else '%.1f+' % (lo / sc_msr)
        b = '%.1f-%.1f' % (lo / sc_ours, hi / sc_ours) if hi < 1e9 else '%.1f+' % (lo / sc_ours)
        print('  %-12s %-10s %-14s %-14s' % (lab, '%.1f%%' % (m.mean() * 100), a, b))

    print('\n【按各自输入空间的 stride-8 分档】')
    fw_msr = face_w / sc_msr
    fw_ours = face_w / sc_ours
    for name, arr in (('MSR+MNP', fw_msr), ('SCRFD-tiny@320', fw_ours)):
        h = (arr < 8).mean() * 100
        r = (arr >= 32).mean() * 100
        print('  %-16s 不足8px: %5.1f%%   可达32px+: %4.1f%%' % (name, h, r))

    print('\n【尺度对齐子集】')
    subset_img = sum(1 for i in range(len(img_w))
                     if per_img[i] > 0 and np.all(
                         np.array(face_w[sum(per_img[:i]):sum(per_img[:i + 1])]) >= MIN_ANCHOR * sc_msr))
    print('  图内所有人脸 >= %.1f px 的图: %d / %d (%.1f%%)'
          % (MIN_ANCHOR * sc_msr, subset_img, len(img_w), subset_img / len(img_w) * 100))


if __name__ == '__main__':
    main()
