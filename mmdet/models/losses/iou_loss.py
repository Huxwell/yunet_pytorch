import math
import warnings
import mmcv
import torch
import torch.nn as nn
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def eiou_loss(pred, target, smooth_point=0.1, eps=1e-07):
    """Implementation of paper 'Extended-IoU Loss: A Systematic IoU-Related
     Method: Beyond Simplified Regression for Better Localization,
     <https://ieeexplore.ieee.org/abstract/document/9429909> '.
    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    px1, py1, px2, py2 = pred[:, (0)], pred[:, (1)], pred[:, (2)], pred[:, (3)]
    tx1, ty1, tx2, ty2 = target[:, (0)], target[:, (1)], target[:, (2)
        ], target[:, (3)]
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)
    intersection = (ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1) - (
        ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (iy1 - ey1)
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1
        ) - intersection + eps
    ious = 1 - intersection / union
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * ious ** 2 / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point)
    return loss


@LOSSES.register_module()
class IoULoss(nn.Module):
    """IoULoss.
    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

@LOSSES.register_module()
class EIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0,
        smooth_point=0.1):
        super(EIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self, pred, target, weight=None, avg_factor=None,
        reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.
            reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * eiou_loss(pred, target, weight,
            smooth_point=self.smooth_point, eps=self.eps, reduction=
            reduction, avg_factor=avg_factor, **kwargs)
        return loss
