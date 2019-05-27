import torch


def predict_anchors(shape, target_deltas, sampled_neg_inds, seed=99):
    """Predict random positive and negative anchor

    Args:
        shape (list): [m, 4], n is the number of anchors
        target_deltas (torch.Tensor): shape (n, 4), n is the number of positive samples
        sampled_neg_inds (torch.Tensor): shape (n', 1), n' is the number of negative samples
        seed (int): seed for randomly generated positive anchor value

    Returns:
        pos_neg_cls_pred (torch.Tensor): shape (n + n', 1)
        pos_delta_pred (torch.Tensor): shape (n, 4)
    """
    # predicted value
    torch.manual_seed(seed)
    pos_delta_pred = target_deltas + torch.rand(target_deltas.shape) / 5
    num_pos_neg_samples = target_deltas.shape[0] + len(sampled_neg_inds)
    pos_neg_cls_pred = torch.rand(num_pos_neg_samples)
    return pos_neg_cls_pred, pos_delta_pred
