import torch
from mmdet.core.bbox.transforms import bbox2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class YuNet(SingleStageDetector):

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=
        None, pretrained=None):
        super(YuNet, self).__init__(backbone, neck, bbox_head, train_cfg,
            test_cfg, pretrained)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
        gt_keypointss=None, gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
            gt_labels, gt_keypointss, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if torch.onnx.is_in_onnx_export():
            return outs
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale
            )
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.
            num_classes) for det_bboxes, det_labels in bbox_list]
        return bbox_results
