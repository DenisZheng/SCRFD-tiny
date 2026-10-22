"""验证本地纯 numpy bbox_overlaps 与上游 cython box_overlaps.pyx 数值一致。

上游定义（box_overlaps.pyx，含 +1 像素约定）：
    area(q) = (q[2]-q[0]+1) * (q[3]-q[1]+1)
    iw = min(b[2],q[2]) - max(b[0],q[0]) + 1
    ih = min(b[3],q[3]) - max(b[1],q[1]) + 1
    inter = iw*ih  (iw>0 且 ih>0)
    iou = inter / (area(b) + area(q) - inter)

本测试按上式独立实现一份逐元素参考版本，与 evaluation.bbox_overlaps 对比。
"""
import os
import sys

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from evaluation import bbox_overlaps  # noqa: E402


def reference_overlaps(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    out = numpy.zeros((N, K), dtype=numpy.float64)
    for k in range(K):
        q = query_boxes[k]
        q_area = (q[2] - q[0] + 1.0) * (q[3] - q[1] + 1.0)
        for n in range(N):
            b = boxes[n]
            iw = min(b[2], q[2]) - max(b[0], q[0]) + 1.0
            if iw > 0:
                ih = min(b[3], q[3]) - max(b[1], q[1]) + 1.0
                if ih > 0:
                    b_area = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
                    inter = iw * ih
                    out[n, k] = inter / (b_area + q_area - inter)
    return out


def main():
    rng = numpy.random.RandomState(20261015)
    fails = 0

    # 随机框集合
    for trial in range(20):
        N = rng.randint(1, 40)
        K = rng.randint(1, 20)
        b = rng.uniform(0, 500, size=(N, 2))
        b = numpy.concatenate([b, b + rng.uniform(1, 200, size=(N, 2))], axis=1)
        q = rng.uniform(0, 500, size=(K, 2))
        q = numpy.concatenate([q, q + rng.uniform(1, 200, size=(K, 2))], axis=1)

        got = bbox_overlaps(b, q)
        exp = reference_overlaps(b, q)
        if not numpy.allclose(got, exp, atol=1e-12, rtol=1e-12):
            fails += 1
            print("MISMATCH trial", trial, "maxdiff", numpy.abs(got - exp).max())

    # 边界：完全相同框
    b = numpy.array([[0., 0., 10., 10.]])
    q = numpy.array([[0., 0., 10., 10.]])
    got = bbox_overlaps(b, q)
    assert abs(got[0, 0] - 1.0) < 1e-12, got
    print("identical box IoU =", got[0, 0])

    # 边界：不相交
    b = numpy.array([[0., 0., 10., 10.]])
    q = numpy.array([[100., 100., 200., 200.]])
    got = bbox_overlaps(b, q)
    assert got[0, 0] == 0.0, got
    print("disjoint box IoU =", got[0, 0])

    # 边界：空输入
    assert bbox_overlaps(numpy.zeros((0, 4)), q).shape == (0, 1)
    assert bbox_overlaps(b, numpy.zeros((0, 4))).shape == (1, 0)
    print("empty input shapes OK")

    # 已知数值手算校验
    # b=(0,0,9,9) q=(0,0,4,4): area_b=100, area_q=25
    # iw=5, ih=5 -> inter=25 -> iou=25/(100+25-25)=0.25
    got = bbox_overlaps(numpy.array([[0., 0., 9., 9.]]), numpy.array([[0., 0., 4., 4.]]))
    assert abs(got[0, 0] - 0.25) < 1e-12, got
    print("hand-computed IoU =", got[0, 0], "(expect 0.25)")

    if fails:
        print(f"FAIL: {fails} mismatching trials")
        sys.exit(1)
    print("PASS: numpy bbox_overlaps matches upstream cython semantics")


if __name__ == "__main__":
    main()
