import warnings
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss


def binary_cross_entropy(pred, label, weight=None, reduction='mean',
    avg_factor=None, class_weight=None, ignore_index=-100, avg_non_ignore=False
    ):
    """Calculate the binary CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
    Returns:
        torch.Tensor: The calculated loss.
    """
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(label, weight,
            pred.size(-1), ignore_index)
    else:
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    if avg_factor is None and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(),
        pos_weight=class_weight, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor
        =avg_factor)
    return loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean',
        class_weight=None, ignore_index=None, loss_weight=1.0,
        avg_non_ignore=False):
        """CrossEntropyLoss.
        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if (ignore_index is not None and not self.avg_non_ignore and self.
            reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to ignore the certain label and average loss over non-ignore labels, which is the same with PyTorch official cross_entropy, set ``avg_non_ignore=True``.'
                )
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None,
        reduction_override=None, ignore_index=None, **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.
            reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=
                cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label,
            weight, class_weight=class_weight, reduction=reduction,
            avg_factor=avg_factor, ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore, **kwargs)
        return loss_cls
