"""
Anchor generation code from mmdet.v0.6rc0
    - https://github.com/open-mmlab/mmdetection/blob/v0.6rc0/mmdet/core/anchor/anchor_generator.py
    - https://github.com/open-mmlab/mmdetection/blob/f2cfa86b4294e2593429adccce64bfd049a27651/mmdet/models/anchor_heads/anchor_head.py#L89-L126
"""
import numpy as np
import torch


def meshgrid(x, y):
    """
    Args:
        x (torch.Tensor): number of position to draw an anchor along y-axis
        y (torch.Tensor): number of position to draw an anchor along x-axis
    """
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    return xx, yy


def gen_base_anchors(base_size, ratios, scales):
    """Generate base anchors.

    Args:
        base_size (int): base anchor size
        ratios (torch.Tensor): ratios of anchor
        scales (torch.Tensor): scales of anchor

    Returns:
        base_anchors (torch.Tensor): [len(ratios) * len(scales), 4]
    """
    w = base_size
    h = base_size

    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ],
        dim=-1,
    ).round()
    return base_anchors


def grid_anchors(base_anchors, featmap_size, stride=16, device="cuda"):
    """Generate grid anchors.

    Args:
        base_anchors (torch.Tensor): [len(ratios) * len(scales), 4]
        featmap_size (list): [f_w, f_h]
        stride (int): stride btw feature map and original image
        device (str): 'cuda' or 'cpu'
    """
    base_anchors = base_anchors.to(device)

    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w, device=device) * stride
    shift_y = torch.arange(0, feat_h, device=device) * stride
    shift_xx, shift_yy = meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    shifts = shifts.type_as(base_anchors)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    all_anchors = all_anchors.view(-1, 4)
    return all_anchors, shifts


def valid_flags(featmap_size, valid_size, num_base_anchors, device="cuda"):
    """Generate flags for each anchors.

    Args:
        featmap_size (list): [f_w, f_h]
        valid_size (list): [valid_w, valid_h], normally equal to featmap_size
        num_base_anchors (int): len(scales) * len(ratios)
        device (str): 'cuda' or 'cpu'
    """
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    valid = valid[:, None].expand(valid.size(0), num_base_anchors).contiguous().view(-1)
    return valid


def get_anchors(
    image_shape, featmap_size, base_size, anchor_stride, scales, ratios, device="cuda"
):
    """Generate anchors and flags.

    Args:
        image_shape (list): [w, h, 3]
        featmap_size (list): [f_w, f_h]
        anchor_stride (int): normally w // f_w or h // f_h
        scales (torch.Tensor): scales of anchor
        ratios (torch.Tensor): ratios of anchor
        device (str): 'cuda' or 'cpu'

    Returns:
        anchors (torch.Tensor): [f_w * f_h * len(scales) * len(ratios), 4]
        flags (torch.Tensor): [f_w * f_h * len(scales) * len(ratios)]
    """
    num_base_anchors = len(scales) * len(ratios)
    base_anchors = gen_base_anchors(base_size, ratios, scales)
    anchors, shifts = grid_anchors(base_anchors, featmap_size, anchor_stride, device)

    feat_h, feat_w = featmap_size
    h, w = image_shape[:2]
    valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
    valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
    valid_size = [valid_feat_h, valid_feat_w]
    flags = valid_flags(featmap_size, valid_size, num_base_anchors, device)
    return anchors, flags
