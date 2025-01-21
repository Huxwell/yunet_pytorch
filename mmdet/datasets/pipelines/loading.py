import os.path as osp
import mmcv
import numpy as np
from ..builder import PIPELINES
try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, to_float32=False, color_type='color', channel_order=
        'bgr', file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['img_info'][
                'filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type,
            channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.
    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles:
    """Load multi-channel images from a list of separate channel files.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.
    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_keypoints (bool): Whether to parse and load the keypoints
            annotation. Default: False.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        denorm_bbox (bool): Whether to convert bbox from relative value to
            absolute value. Only used in OpenImage Dataset.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, with_bbox=True, with_label=True, with_keypoints=
        False, with_mask=False, with_seg=False, poly2mask=True, denorm_bbox
        =False, file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_keypoints = with_keypoints
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.denorm_bbox = denorm_bbox
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()
        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h
        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _load_keypoints(self, results):
        """Private function to load bounding box annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        ann_info = results['ann_info']
        results['gt_keypointss'] = ann_info['keypointss'].copy()
        results['keypoints_fields'] = ['gt_keypointss']
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_keypoints:
            results = self._load_keypoints(results)
        return results


@PIPELINES.register_module()
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.
    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """


@PIPELINES.register_module()
class LoadProposals:
    """Load proposal pipeline.
    Required key is "proposals". Updated keys are "proposals", "bbox_fields".
    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.
    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Default: True
    """
