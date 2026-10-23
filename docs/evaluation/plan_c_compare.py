"""
在「MSR+MNP 也能检测」的 WIDER val 子集上，对比 SCRFD-tiny 两个 checkpoint。

子集定义（方案 C）：
    只保留「图内所有人脸都 >= 102.4 px 宽」的图片
    102.4 = MSR 输入空间最小 anchor 16px × 缩放比 (1024/160 = 6.4)
    即：MSR+MNP 能可靠检测的尺度范围

评估方式：COCO 式 mAP（IoU 0.50:0.95，10 阈值平均），自实现，不依赖 pycocotools。

用法：
    python3 plan_c_compare.py <val_labels_dir> <preds_finetuned> <preds_baseline> [out_json]

    <preds_*> 为 tools/export_wider_preds.py 的输出目录，
    结构为 <event>/<stem>.txt

说明：本脚本产出的数字是「尺度对齐的参考值」，不是官方 WIDER 成绩。
"""
import json
import os
import sys

import numpy as np

MIN_INPUT_PX = 16.0
MSR_IN_W = 160
IMG_W = 1024
SCALE_MSR = IMG_W / MSR_IN_W
MIN_ORIG_PX = MIN_INPUT_PX * SCALE_MSR      # 102.4


def load_gt(val_lbl):
    items = []
    for fn in sorted(os.listdir(val_lbl)):
        rows = []
        with open(os.path.join(val_lbl, fn)) as f:
            for line in f:
                v = line.split()
                if len(v) >= 5:
                    rows.append([float(v[1]), float(v[2]), float(v[3]), float(v[4])])
        items.append((fn[:-4], rows))
    return items


def load_preds(wouts):
    """预测目录是 <event>/<stem>.txt，label 是 <event>_<stem>.txt，拼 flat_key 对齐。"""
    preds = {}
    for ev in os.listdir(wouts):
        d = os.path.join(wouts, ev)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.txt'):
                continue
            boxes = []
            with open(os.path.join(d, fn)) as f:
                lines = f.readlines()
            for line in lines[2:]:
                p = line.split()
                if len(p) >= 5:
                    x1, y1, w, h, s = map(float, p[:5])
                    boxes.append([x1, y1, x1 + w, y1 + h, s])
            preds[ev + '_' + fn[:-4]] = np.array(boxes) if boxes else np.zeros((0, 5))
    return preds


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    ih = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = iw * ih
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(union > 0, inter / union, 0)


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(items, preds, iou_list, tag):
    aps = []
    for thr in iou_list:
        scores, n_gt = [], 0
        for stem, gt in items:
            n_gt += len(gt)
            pred = preds.get(stem, np.zeros((0, 5)))
            if len(gt) == 0 or len(pred) == 0:
                continue
            ious = iou_matrix(pred[:, :4], np.array(gt))
            matched = np.zeros(len(gt), dtype=bool)
            for pi in np.argsort(-pred[:, 4]):
                row = ious[pi]
                if len(row) == 0:
                    scores.append((pred[pi, 4], 0))
                    continue
                best = int(np.argmax(row))
                if row[best] >= thr and not matched[best]:
                    matched[best] = True
                    scores.append((pred[pi, 4], 1))
                else:
                    scores.append((pred[pi, 4], 0))
        if n_gt == 0:
            aps.append(0.0)
            continue
        scores.sort(key=lambda x: -x[0])
        tp = np.cumsum([s[1] for s in scores])
        fp = np.cumsum([1 - s[1] for s in scores])
        aps.append(compute_ap(tp / n_gt, tp / np.maximum(tp + fp, 1e-9)))
    m, ap50 = float(np.mean(aps)), aps[0]
    print('  %-32s mAP50-95=%.4f  AP50=%.4f  (faces=%d)'
          % (tag, m, ap50, sum(len(g) for _, g in items)))
    return m, ap50


def main():
    val_lbl = sys.argv[1]
    preds_ft_dir = sys.argv[2]
    preds_bs_dir = sys.argv[3]
    out_json = sys.argv[4] if len(sys.argv) > 4 else None

    print('=' * 74)
    print('尺度对齐子集对比（对齐 MSR+MNP 能力边界）')
    print('筛选: 图内所有人脸 >= %.1f px' % MIN_ORIG_PX)
    print('=' * 74)

    all_items = load_gt(val_lbl)
    print('全量 val: %d 图 / %d 脸' % (len(all_items), sum(len(g) for _, g in all_items)))

    subset = [(s, g) for s, g in all_items
              if len(g) > 0 and all((b[2] - b[0]) >= MIN_ORIG_PX for b in g)]
    print('对齐子集: %d 图 / %d 脸\n' % (len(subset), sum(len(g) for _, g in subset)))

    pf = load_preds(preds_ft_dir)
    pb = load_preds(preds_bs_dir)
    hit = sum(1 for s, _ in all_items if s in pf)
    print('预测匹配率: %d/%d (%.1f%%)' % (hit, len(all_items), hit / len(all_items) * 100))
    if hit < len(all_items) * 0.99:
        print('WARN: 预测未完全匹配')

    iou_list = np.arange(0.5, 1.0, 0.05).tolist()
    print('\n【对齐子集】')
    m_ft, a_ft = evaluate(subset, pf, iou_list, '微调 ft320-20261015')
    m_bs, a_bs = evaluate(subset, pb, iou_list, '基线 weights/best.pt')
    print('\n【全量 val（对照）】')
    m_ft_all, a_ft_all = evaluate(all_items, pf, iou_list, '微调 ft320')
    m_bs_all, a_bs_all = evaluate(all_items, pb, iou_list, '基线')

    print('\n' + '=' * 74)
    print('%-30s %10s %10s' % ('', 'mAP50-95', 'AP50'))
    print('%-30s %10.4f %10.4f' % ('微调 @对齐子集', m_ft, a_ft))
    print('%-30s %10.4f %10.4f' % ('基线 @对齐子集', m_bs, a_bs))
    print('%-30s %10.4f %10.4f' % ('微调 @全量 val', m_ft_all, a_ft_all))
    print('%-30s %10.4f %10.4f' % ('基线 @全量 val', m_bs_all, a_bs_all))
    print()
    print('参考: ESP-DL MSR+MNP 报告 mAP50-95 = 0.367 (custom val set, 未公开)')

    if out_json:
        rec = {
            'subset': {'min_face_px': MIN_ORIG_PX, 'images': len(subset),
                       'faces': sum(len(g) for _, g in subset),
                       'derivation': 'MSR input 160w, min anchor 16px, WIDER val width 1024px'},
            'finetuned_subset': {'mAP50_95': m_ft, 'AP50': a_ft},
            'baseline_subset': {'mAP50_95': m_bs, 'AP50': a_bs},
            'finetuned_full': {'mAP50_95': m_ft_all, 'AP50': a_ft_all},
            'baseline_full': {'mAP50_95': m_bs_all, 'AP50': a_bs_all},
            'note': 'scale-aligned subset, NOT an official WIDER score',
        }
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print('报告: %s' % out_json)


if __name__ == '__main__':
    main()
