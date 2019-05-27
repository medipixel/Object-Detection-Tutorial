import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch

from transforms import bbox2delta


def prepare_base_figure(grid_size, figsize=(20, 20)):
    """
    Args:
        grid_size (int): grid interval size
        figsize (tuple): figure size, default is (20, 20)

    Return:
        ax (AxesSubplot)
    """
    fig, ax = plt.subplots(figsize=figsize)

    loc = plticker.MultipleLocator(base=grid_size)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    ax.grid(which="major", axis="both", linestyle="--", color="w")
    return ax


def draw_anchor_gt_overlaps(
    overlaps,
    gt_bboxes_list,
    featmap_size,
    anchors_per_grid,
    anchor_stride,
    grid_size=1,
    draw_gt=False,
    figsize=(20, 20),
):
    """Draw anchor overlaps w.r.t. gt bboxes

    Args:
        overlaps (torch.Tensor): shape (m, n), m is the number of gt_bboxes,
                                 n is the number of anchors
        gt_bboxes_list (torch.Tensor): shape (m, 4), m is the number of gt_bboxes
        anchors_per_grid (int): the number of anchors in a grid
        anchor_stride (int): stride of anchor
        grid_size (int): grid interval size
        draw_gt (bool): represent gt or heatmap, default is False
        figsize (tuple): figure size, default is (20, 20)
    """
    max_anchor_overlaps = (
        overlaps.reshape(*featmap_size, anchors_per_grid)
        .cpu()
        .numpy()
        .max(axis=-1)
        .copy()
    )
    positive_overlaps = np.where(max_anchor_overlaps > 0)

    for gt_bbox in gt_bboxes_list:
        assert any(gt_bbox % anchor_stride) is False

    grid_x, grid_y = max_anchor_overlaps.shape[:2]
    ax = prepare_base_figure(grid_size, figsize)
    title = "Overlap(feature map's reg prediction and gt)"
    text_size = 320 / featmap_size[0]

    if draw_gt:
        background_image = np.zeros([*max_anchor_overlaps.shape, 3])
        for gt_bbox in gt_bboxes_list:
            x1, y1, x2, y2 = (gt_bbox // anchor_stride).numpy().astype(int)
            cv2.rectangle(background_image, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 1)
        ax.imshow(background_image, extent=[0, grid_x, grid_y, 0])
        title += ", [gt:blue square]"

        legend_elements = [
            Line2D([0], [0], marker="s", color="b", markersize=20, lw=0, label="gt")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=30)
    else:
        ax.imshow(max_anchor_overlaps, extent=[0, grid_x, grid_y, 0])
        title += ", [overlaps:heatmap]"

    for (x, y) in zip(*positive_overlaps):
        ax.annotate(
            f"{max_anchor_overlaps[x, y]:.2f}",
            xy=(0, 0),
            xytext=(x + 0.13, y + 0.6),
            color="white",
            size=text_size,
        )
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="major", labelsize=text_size)
    plt.margins(0)
    plt.title(title, fontsize=30, pad=20)
    plt.show()


def draw_pos_assigned_bboxes(
    image_shape,
    grid_size,
    gt_bboxes_list,
    pos_bboxes,
    pos_pred_bboxes=None,
    figsize=(20, 20),
):
    """Draw positive, negative bboxes.

    Args:
        image_shape (list): shape of the image
        grid_size (int): grid interval size
        gt_bboxes_list (torch.Tensor): shape (m, 4), m is the number of gt_bboxes
        pos_bboxes (torch.Tensor): shape (n, 4), n is the number of pos_bboxes
        pos_pred_bboxes (torch.Tensor): shape (n, 4), n is the number of pos_pred_bboxes
        figsize (tuple): figure size, default is (20, 20)
    """
    assert len(pos_bboxes) == len(pos_pred_bboxes)
    for i in range(len(pos_bboxes)):
        background_image = np.zeros(image_shape)
        ax = prepare_base_figure(grid_size, figsize)

        for gt_bbox in gt_bboxes_list:
            x1, y1, x2, y2 = gt_bbox
            gt_coord = [(x2 + x1) // 2, (y2 + y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2]
            cv2.rectangle(background_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        x1, y1, x2, y2 = pos_bboxes[i]
        anchor_coord = [(x2 + x1) // 2, (y2 + y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2]
        cv2.rectangle(background_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        x1, y1, x2, y2 = pos_pred_bboxes[i]
        pred_coord = [(x2 + x1) // 2, (y2 + y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2]
        cv2.rectangle(background_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        legend_elements = [
            Line2D([0], [0], color="b", lw=4, label="gt"),
            Line2D([0], [0], color="r", lw=4, label=f"pred(pos[{i}])"),
            Line2D([0], [0], color="g", lw=4, label=f"anchor(pos[{i}])"),
        ]

        image_x, image_y = image_shape[:2]

        ax.imshow(background_image, extent=[0, image_x, image_y, 0])
        ax.legend(handles=legend_elements, loc="upper right", fontsize=30)

        ax.annotate(
            "coordination:", xy=(0, 0), xytext=(30, 150), color="white", size=30
        )
        ax.annotate(
            f"- anchor: {[int(x) for x in anchor_coord]}",
            xy=(0, 0),
            xytext=(40, 160),
            color="green",
            size=30,
        )
        ax.annotate(
            f"- gt: {[int(x) for x in gt_coord]}",
            xy=(0, 0),
            xytext=(40, 170),
            color="blue",
            size=30,
        )
        ax.annotate(
            f"- pred: {[int(x) for x in pred_coord]}",
            xy=(0, 0),
            xytext=(40, 180),
            color="red",
            size=30,
        )

        gt_coord = torch.tensor([gt_coord])
        anchor_coord = torch.tensor([anchor_coord])
        pred_coord = torch.tensor([pred_coord])
        gt_delta = bbox2delta(anchor_coord, gt_coord)
        pred_delta = bbox2delta(anchor_coord, pred_coord)

        ax.annotate("delta:", xy=(0, 0), xytext=(30, 190), color="white", size=30)
        ax.annotate(
            f"- anchor_gt: {[round(float(x), 2) for x in gt_delta[0]]}",
            xy=(0, 0),
            xytext=(40, 200),
            color="white",
            size=30,
        )
        ax.annotate(
            f"- anchor_pred: {[round(float(x), 2) for x in pred_delta[0]]}",
            xy=(0, 0),
            xytext=(40, 210),
            color="white",
            size=30,
        )
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", which="major", labelsize=20)

        plt.title(f"gt, prediction, anchor (pos[{i}])", fontsize=30, pad=20)
        plt.show()


def draw_base_anchor_on_grid(base_anchor, figsize=(20, 20)):
    board = np.zeros((256, 256, 3))

    for anchor in base_anchor:
        x1, y1, x2, y2 = np.array(anchor) + 112  # 왜 112?
        cv2.rectangle(board, (x1, y1), (x2, y2), (0, 255, 0), 1)

    ax = prepare_base_figure(16, figsize)

    ax.annotate(
        "base anchor center at (0, 0)",
        xy=(0, 0),
        xytext=(20, -85),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )

    ax.annotate(
        "-23, -45",
        xy=(-23, -45),
        xytext=(-48, -78),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )
    ax.annotate(
        "-32, -32",
        xy=(-32, -32),
        xytext=(-68, -68),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )
    ax.annotate(
        "-45, -23",
        xy=(-45, -23),
        xytext=(-78, -58),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )

    ax.annotate(
        "23, 45",
        xy=(23, 45),
        xytext=(22, 72),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )
    ax.annotate(
        "32, 32",
        xy=(32, 32),
        xytext=(43, 43),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )
    ax.annotate(
        "45, 23",
        xy=(45, 23),
        xytext=(72, 22),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )
    legend_elements = [Line2D([0], [0], color="g", lw=4, label="base anchor")]
    ax.imshow(board, extent=[-128, 128, 128, -128])
    ax.legend(handles=legend_elements, loc="upper right", fontsize=30)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="major", labelsize=20)
    plt.title("Base Anchor", fontsize=30, pad=20)
    plt.show()


def draw_anchor_samples_on_image(image_shape, base_anchor, all_anchors, shifts):
    board = np.zeros(image_shape)
    fig, ax = plt.subplots(figsize=(20, 20))
    loc = plticker.FixedLocator(range(0, image_shape[0], 16))

    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    ax.grid(which="major", axis="both", linestyle="--", color="w")

    for anchor in base_anchor:
        x1, y1, x2, y2 = np.array(anchor, dtype=np.uint8) + np.array(
            [48, 112, 48, 112], dtype=np.uint8
        )
        cv2.rectangle(board, (x1, y1), (x2, y2), (0, 255, 0), 1)
        x1 = int(x1 + 128)
        x2 = int(x2 + 128)
        cv2.rectangle(board, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for i in range(64, 208, 16):
        ax.scatter(i, 128, s=50, c="r")

    legend_elements = [
        Line2D([0], [0], color="g", lw=4, label="anchor[4, 8]"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="r",
            label="center points of anchors",
            markerfacecolor="r",
            lw=0,
            markersize=12,
        ),
        Line2D([0], [0], color="b", lw=4, label="anchor[12, 8]"),
    ]

    ax.annotate(
        "coords: (4x16, 8x16)\nindex: (4, 8, :3)",
        xy=(63, 127),
        xytext=(30, 70),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )
    ax.annotate(
        "8 shifts along x-axis",
        xy=(80, 128),
        xytext=(64, 145),
        color="white",
        size=25,
    )
    ax.annotate(
        "",
        xy=(188, 133),
        xytext=(64, 133),
        color="white",
        size=30,
        arrowprops=dict(facecolor="white", shrink=5),
    )
    ax.imshow(board, extent=[0, image_shape[0], image_shape[1], 0])
    ax.legend(handles=legend_elements, loc="upper right", fontsize=30)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="major", labelsize=20)
    plt.title("Anchor Samples on Image", fontsize=30, pad=20)
    plt.show()
