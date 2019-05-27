"""
Loss code from mmdet.v0.6rc0
    - https://github.com/open-mmlab/mmdetection/blob/f2cfa86b4294e2593429adccce64bfd049a27651/mmdet/core/loss/losses.py#L76-L89
"""
import torch
import torch.nn.functional as F


def smooth_l1_loss(pred, target, beta=1.0):
    """
    Args:
        pred (torch.Tensor): shape (m, 4), m is the number of positive predictions
        target (torch.Tensor): shape (m, 4), m is the number of positive targets
        beta (float): smooth l1 loss parameter, default 1.0

    Returns:
        loss (torch.Tensor): scalar tensor
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta).sum()
    return loss


def binary_cross_entropy(pred, label):
    """
    Args:
        pred (torch.Tensor): shape (n, 4), n is the number of positive and negative predictions
        label (torch.Tensor): shape (n, 4), n is the number of positive and negative predictions

    Returns:
        loss (torch.Tensor): scalar tensor
    """
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction="sum")
    return loss
