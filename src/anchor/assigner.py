"""
Assigner code from mmdet.v0.6rc0
    - https://github.com/open-mmlab/mmdetection/blob/f2cfa86b4294e2593429adccce64bfd049a27651/mmdet/core/bbox/assigners/max_iou_assigner.py#L87-L146
"""
import torch


def assign_wrt_overlaps(overlaps, pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0.0):
    """Assign w.r.t. the overlaps of bboxes with gts.

    Args:
        overlaps (torcch.Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
        gt_labels (torch.Tensor, optional): Labels of k gt_bboxes, shape (k, ).

    Returns:
        :obj:`AssignResult`: The assign result.
    """
    if overlaps.numel() == 0:
        raise ValueError("No gt or proposals")

    num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

    # 1. assign -1 by default
    assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

    # for each anchor, which gt best overlaps with it
    # for each anchor, the max iou of all gts
    max_overlaps, argmax_overlaps = overlaps.max(dim=0)
    # for each gt, which anchor best overlaps with it
    # for each gt, the max iou of all proposals
    gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

    # 2. assign negative: below
    if isinstance(neg_iou_thr, float):
        assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < neg_iou_thr)] = 0
    elif isinstance(neg_iou_thr, tuple):
        assert len(neg_iou_thr) == 2
        assigned_gt_inds[
            (max_overlaps >= neg_iou_thr[0]) & (max_overlaps < neg_iou_thr[1])
        ] = 0

    # 3. assign positive: above positive IoU threshold
    pos_inds = max_overlaps >= pos_iou_thr
    assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

    # 4. assign fg: for each gt, proposals with highest IoU
    for i in range(num_gts):
        if gt_max_overlaps[i] >= min_pos_iou:
            max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
            assigned_gt_inds[max_iou_inds] = i + 1

    return num_gts, assigned_gt_inds, max_overlaps


def bbox_overlaps(bboxes1, bboxes2):
    """Calculate overlap between two set of bboxes.

    Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (torch.Tensor): shape (m, 4)
        bboxes2 (torch.Tensor): shape (n, 4)

    Returns:
        ious(torch.Tensor): shape (m, n)
    """

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

    wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
    overlap = wh[:, :, 0] * wh[:, :, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)

    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1[:, None] + area2 - overlap)

    return ious
