"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor

本文件从 wondervictor/WiderFace-Evaluation 引入，用于 SCRFD-tiny 的
WIDER Face Easy/Medium/Hard 标准评估。

相对上游的改动（均已在下方标注 [FORK]）：
1. `from bbox import bbox_overlaps`（cython 扩展）改为本地纯 numpy 实现 `bbox_overlaps`，
   避免编译 box_overlaps.pyx 和引入 cython 构建依赖。数值定义保持与上游一致（含 +1 像素）。
   该 numpy 实现已与上游 cython 版本做逐值对比验证（见 tools/widerface_evaluation/tests/）。
2. `from IPython import embed` 移除（上游仅调试用，未实际调用）。
3. `imgname.rstrip('.jpg')` 改为 `imgname[:-4]`。`rstrip` 按字符集剥离，
   对以 `.` `j` `p` `g` 结尾的文件名会误伤；当前 WIDER val 3226 张图实测未触发，
   但属潜在错误，此处一并修正。
4. `evaluation()` 增加 `return_aps` 参数与返回值，便于程序化调用。
5. 移除 tqdm 依赖，改为可选（无 tqdm 时退化为 range）。
6. GT 读取使用 scipy.io.loadmat（镜像已包含 scipy，见 Dockerfile.scrfd-tiny），
   不再依赖 MATLAB。
"""

import os
import pickle
import argparse

import numpy

try:
    from tqdm import tqdm
except ImportError:  # [FORK] tqdm 可选
    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# [FORK] 纯 numpy 实现，替代 box_overlaps.pyx
# 与上游语义一致：坐标按 (x1, y1, x2, y2)，面积含 +1 像素
# ---------------------------------------------------------------------------
def bbox_overlaps(boxes, query_boxes):
    """计算 boxes (N,4) 与 query_boxes (K,4) 的 IoU 矩阵 (N,K)。

    与上游 box_overlaps.pyx 保持相同的 +1 像素约定。
    """
    boxes = numpy.asarray(boxes, dtype=numpy.float64)
    query_boxes = numpy.asarray(query_boxes, dtype=numpy.float64)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = numpy.zeros((N, K), dtype=numpy.float64)
    if N == 0 or K == 0:
        return overlaps

    # 逐 query box 计算，保持与上游相同的循环结构与数值行为
    for k in range(K):
        qx1, qy1, qx2, qy2 = query_boxes[k]
        box_area = (qx2 - qx1 + 1.0) * (qy2 - qy1 + 1.0)

        iw = numpy.minimum(boxes[:, 2], qx2) - numpy.maximum(boxes[:, 0], qx1) + 1.0
        ih = numpy.minimum(boxes[:, 3], qy2) - numpy.maximum(boxes[:, 1], qy1) + 1.0

        valid = (iw > 0) & (ih > 0)
        if not numpy.any(valid):
            continue

        area_b = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
        inter = iw[valid] * ih[valid]
        ua = area_b[valid] + box_area - inter
        overlaps[valid, k] = inter / ua

    return overlaps


def load_mat(path):
    """[FORK] 读取 .mat，依赖镜像内的 scipy（见 Dockerfile.scrfd-tiny）。"""
    from scipy.io import loadmat
    return loadmat(path)


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = load_mat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = load_mat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = load_mat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = load_mat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if len(line) < 5 or line[0] == '':
            continue
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = numpy.array(boxes) if boxes else numpy.zeros((0, 5))
    # [FORK] rstrip('.jpg') -> 显式切片，避免按字符集误剥离
    stem = img_file.split('/')[-1]
    if stem.lower().endswith('.jpg'):
        stem = stem[:-4]
    return stem, boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        if not os.path.isdir(event_dir):
            continue
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = numpy.min(v[:, -1])
            _max = numpy.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    if diff <= 0:
        return
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score) / diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = numpy.zeros(_pred.shape[0])
    recall_list = numpy.zeros(_gt.shape[0])
    proposal_list = numpy.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = numpy.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = numpy.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = numpy.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = numpy.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = numpy.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0] if pr_curve[i, 0] else 0.0
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face if count_face else 0.0
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = numpy.concatenate(([0.], rec, [1.]))
    mpre = numpy.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = numpy.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = numpy.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5, return_aps=False, verbose=True):
    pred = get_preds(pred) if not isinstance(pred, dict) else pred
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = numpy.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm(range(event_num))
        for i in pbar:
            if hasattr(pbar, 'set_description'):
                pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = numpy.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    if verbose:
        print("==================== Results ====================")
        print("Easy   Val AP: {}".format(aps[0]))
        print("Medium Val AP: {}".format(aps[1]))
        print("Hard   Val AP: {}".format(aps[2]))
        print("=================================================")
    if return_aps:
        return aps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./widerface_txt/")
    parser.add_argument('-g', '--gt', default='./ground_truth/')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)
