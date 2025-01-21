import warnings
import mmcv
import numpy as np
from numpy import random
from ..builder import PIPELINES
try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None
try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class Resize:
    """Resize images & bbox & mask.
    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.
    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - ``ratio_range is not None``: randomly sample a ratio from the ratio       range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly       sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly       sample a scale from multiple scales.
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self, img_scale=None, multiscale_mode='range', ratio_range
        =None, keep_ratio=True, bbox_clip_border=True, backend='cv2',
        interpolation='bilinear', override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)
        if ratio_range is not None:
            assert len(self.img_scale) == 1
        else:
            assert multiscale_mode in ['value', 'range', 'square_range']
        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into                 ``results``, which would be used by subsequent pipelines.
        """
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0],
                self.ratio_range)
        elif len(self.img_scale) == 1:
            if self.multiscale_mode == 'square_range':
                scale, scale_idx = self.random_sample_square(self.img_scale)
            else:
                scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(results[key], results[
                    'scale'], return_scale=True, interpolation=self.
                    interpolation, backend=self.backend)
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(results[key], results
                    ['scale'], return_scale=True, interpolation=self.
                    interpolation, backend=self.backend)
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                dtype=np.float32)
            results['img_shape'] = img.shape
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_keypoints(self, results):
        for key in results.get('keypoints_fields', []):
            keypointss = results[key].copy()
            factors = results['scale_factor']
            assert factors[0] == factors[2]
            assert factors[1] == factors[3]
            keypointss[:, :, (0)] *= factors[0]
            keypointss[:, :, (1)] *= factors[1]
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                keypointss[:, :, (0)] = np.clip(keypointss[:, :, (0)], 0,
                    img_shape[1])
                keypointss[:, :, (1)] = np.clip(keypointss[:, :, (1)], 0,
                    img_shape[0])
            results[key] = keypointss

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(results[key], results['scale'],
                    interpolation='nearest', backend=self.backend)
            else:
                gt_seg = mmcv.imresize(results[key], results['scale'],
                    interpolation='nearest', backend=self.backend)
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',                 'keep_ratio' keys are added into result dict.
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple([int(x * scale_factor) for x in
                    img_shape][::-1])
            else:
                self._random_scale(results)
        elif not self.override:
            assert 'scale_factor' not in results, 'scale and scale_factor cannot be both set.'
        else:
            results.pop('scale')
            if 'scale_factor' in results:
                results.pop('scale_factor')
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_keypoints(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results


@PIPELINES.register_module()
class RandomFlip:
    """Flip the image & bbox & mask.
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:
    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.
    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError(
                'flip_ratios must be None, float, or list of float')
        self.flip_ratio = flip_ratio
        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction
        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.
        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[(...), 0::4] = w - bboxes[(...), 2::4]
            flipped[(...), 2::4] = w - bboxes[(...), 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[(...), 1::4] = h - bboxes[(...), 3::4]
            flipped[(...), 3::4] = h - bboxes[(...), 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[(...), 0::4] = w - bboxes[(...), 2::4]
            flipped[(...), 1::4] = h - bboxes[(...), 3::4]
            flipped[(...), 2::4] = w - bboxes[(...), 0::4]
            flipped[(...), 3::4] = h - bboxes[(...), 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def keypoints_flip(self, keypointss, img_shape, direction):
        assert direction == 'horizontal'
        assert keypointss.shape[-1] == 3
        assert keypointss.shape[1] == 5
        assert keypointss.ndim == 3
        flipped = keypointss.copy()
        flip_order = [1, 0, 2, 4, 3]
        for idx, a in enumerate(flip_order):
            flipped[:, (idx), :] = keypointss[:, (a), :]
        w = img_shape[1]
        flipped[..., 0] = w - flipped[..., 0]
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added                 into result dict.
        """
        if 'flip' not in results:
            if isinstance(self.direction, list):
                direction_list = self.direction + [None]
            else:
                direction_list = [self.direction, None]
            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1
                    ) + [non_flip_ratio]
            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(results[key], direction=results[
                    'flip_direction'])
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key], results[
                    'img_shape'], results['flip_direction'])
            for key in results.get('keypoints_fields', []):
                results[key] = self.keypoints_flip(results[key], results[
                    'img_shape'], results['flip_direction'])
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(results[key], direction=results[
                    'flip_direction'])
        return results


@PIPELINES.register_module()
class RandomShift:
    """Shift the image and box given shift pixels and probability.
    Args:
        shift_ratio (float): Probability of shifts. Default 0.5.
        max_shift_px (int): The max pixels for shifting. Default 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Default 1.
    """


@PIPELINES.register_module()
class Pad:
    """Pad the image & masks & segmentation map.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self, size=None, size_divisor=None, pad_to_square=False,
        pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                f'pad_val of float type is deprecated now, please use pad_val=dict(img={pad_val}, masks={pad_val}, seg=255) instead.'
                , DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square
        if pad_to_square:
            assert size is None and size_divisor is None, 'The size and size_divisor must be None when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, 'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = max_size, max_size
            if self.size is not None:
                pad_r = self.size[0] - results[key].shape[1]
                pad_b = self.size[1] - results[key].shape[0]
                padded_img = mmcv.impad(results[key], padding=(0, 0, pad_r,
                    pad_b), pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(results[key], self.
                    size_divisor, pad_val=pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key], shape=results[
                'pad_shape'][:2], pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.
                std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb
            =self.to_rgb)
        return results


@PIPELINES.register_module()
class RandomCrop:
    """Random crop the image & bboxes & masks.
    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.
    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """


@PIPELINES.register_module()
class RandomSquareCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.
    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
    Note:
        The keys for bboxes, labels and masks should be paired. That is,         `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and         `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self, crop_ratio_range=None, crop_choice=None,
        bbox_clip_border=True):
        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = crop_choice
        self.bbox_clip_border = bbox_clip_border
        assert (self.crop_ratio_range is None) ^ (self.crop_choice is None)
        if self.crop_ratio_range is not None:
            self.crop_ratio_min, self.crop_ratio_max = self.crop_ratio_range
        self.bbox2label = {'gt_bboxes': 'gt_labels', 'gt_bboxes_ignore':
            'gt_labels_ignore'}
        self.bbox2mask = {'gt_bboxes': 'gt_masks', 'gt_bboxes_ignore':
            'gt_masks_ignore'}

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images and bounding boxes cropped,                 'img_shape' key is updated.
        """
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'
                ], 'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        assert 'gt_bboxes' in results
        boxes = results['gt_bboxes']
        h, w, c = img.shape
        scale_retry = 0
        if self.crop_ratio_range is not None:
            max_scale = self.crop_ratio_max
        else:
            max_scale = np.amax(self.crop_choice)
        while True:
            scale_retry += 1
            if scale_retry == 1 or max_scale > 1.0:
                if self.crop_ratio_range is not None:
                    scale = np.random.uniform(self.crop_ratio_min, self.
                        crop_ratio_max)
                elif self.crop_choice is not None:
                    scale = np.random.choice(self.crop_choice)
            else:
                scale = scale * 1.2
            for i in range(250):
                short_side = min(w, h)
                cw = int(scale * short_side)
                ch = cw
                if w == cw:
                    left = 0
                elif w > cw:
                    left = random.randint(0, w - cw)
                else:
                    left = random.randint(w - cw, 0)
                if h == ch:
                    top = 0
                elif h > ch:
                    top = random.randint(0, h - ch)
                else:
                    top = random.randint(h - ch, 0)
                patch = np.array((int(left), int(top), int(left + cw), int(
                    top + ch)), dtype=np.int)

                def is_center_of_bboxes_in_patch(boxes, patch):
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = (center[:, (0)] > patch[0]) * (center[:, (1)] >
                        patch[1]) * (center[:, (0)] < patch[2]) * (center[:,
                        (1)] < patch[3])
                    return mask
                mask = is_center_of_bboxes_in_patch(boxes, patch)
                if not mask.any():
                    continue
                for key in results.get('bbox_fields', []):
                    boxes = results[key].copy()
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    boxes = boxes[mask]
                    if self.bbox_clip_border:
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)
                    results[key] = boxes
                    label_key = self.bbox2label.get(key)
                    if label_key in results:
                        results[label_key] = results[label_key][mask]
                    if key == 'gt_bboxes':
                        for kps_key in results.get('keypoints_fields', []):
                            keypointss = results[kps_key].copy()
                            keypointss = keypointss[(mask), :, :]
                            if self.bbox_clip_border:
                                keypointss[:, :, :2] = keypointss[:, :, :2
                                    ].clip(max=patch[2:])
                                keypointss[:, :, :2] = keypointss[:, :, :2
                                    ].clip(min=patch[:2])
                            keypointss[:, :, (0)] -= patch[0]
                            keypointss[:, :, (1)] -= patch[1]
                            results[kps_key] = keypointss
                    mask_key = self.bbox2mask.get(key)
                    if mask_key in results:
                        results[mask_key] = results[mask_key][mask.nonzero()[0]
                            ].crop(patch)
                rimg = np.ones((ch, cw, 3), dtype=img.dtype) * 128
                patch_from = patch.copy()
                patch_from[0] = max(0, patch_from[0])
                patch_from[1] = max(0, patch_from[1])
                patch_from[2] = min(img.shape[1], patch_from[2])
                patch_from[3] = min(img.shape[0], patch_from[3])
                patch_to = patch.copy()
                patch_to[0] = max(0, patch_to[0] * -1)
                patch_to[1] = max(0, patch_to[1] * -1)
                patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
                patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])
                rimg[patch_to[1]:patch_to[3], patch_to[0]:patch_to[2], :
                    ] = img[patch_from[1]:patch_from[3], patch_from[0]:
                    patch_from[2], :]
                img = rimg
                results['img'] = img
                results['img_shape'] = img.shape
                return results


@PIPELINES.register_module()
class SegRescale:
    """Rescale semantic segmentation maps.
    Args:
        scale_factor (float): The scale factor of the final output.
        backend (str): Image rescale backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """


@PIPELINES.register_module()
class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """


@PIPELINES.register_module()
class Expand:
    """Random expand the image & bboxes.
    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.
    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """


@PIPELINES.register_module()
class MinIoURandomCrop:
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.
    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    Note:
        The keys for bboxes, labels and masks should be paired. That is,         `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and         `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """


@PIPELINES.register_module()
class Corrupt:
    """Corruption augmentation.
    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.
    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """


@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """


@PIPELINES.register_module()
class RandomCenterCropPad:
    """Random center crop and random around padding for CornerNet.
    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from :class:`RandomCrop`, the output
    shape may not equal to ``crop_size`` strictly. We choose a random value
    from ``ratios`` and the output shape could be larger or smaller than
    ``crop_size``. The padding operation is also different from :class:`Pad`,
    here we use around padding instead of right-bottom padding.
    The relation between output image (padding image) and original image:
    .. code:: text
                        output image
               +----------------------------+
               |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
               |          padded area       |
               +----------------------------+
    There are 5 main areas in the figure:
    - output image: output image of this operation, also called padding
      image in following instruction.
    - original image: input image of this operation.
    - padded area: non-intersect area of output image and original image.
    - cropped area: the overlap of output image and original image.
    - center range: a smaller area where random center chosen from.
      center range is computed by ``border`` and original image's shape
      to avoid our random center is too close to original image's border.
    Also this operation act differently in train and test mode, the summary
    pipeline is listed below.
    Train pipeline:
    1. Choose a ``random_ratio`` from ``ratios``, the shape of padding image
       will be ``random_ratio * crop_size``.
    2. Choose a ``random_center`` in center range.
    3. Generate padding image with center matches the ``random_center``.
    4. Initialize the padding image with pixel value equals to ``mean``.
    5. Copy the cropped area to padding image.
    6. Refine annotations.
    Test pipeline:
    1. Compute output shape according to ``test_pad_mode``.
    2. Generate padding image with center matches the original image
       center.
    3. Initialize the padding image with pixel value equals to ``mean``.
    4. Copy the ``cropped area`` to padding image.
    Args:
        crop_size (tuple | None): expected size after crop, final size will
            computed according to ratio. Requires (h, w) in train mode, and
            None in test mode.
        ratios (tuple): random select a ratio from tuple and crop image to
            (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode.
        border (int): max distance from center select area to image border.
            Only available in train mode.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
        test_mode (bool): whether involve random variables in transform.
            In train mode, crop_size is fixed, center coords and ratio is
            random selected from predefined lists. In test mode, crop_size
            is image's original shape, center coords and ratio is fixed.
        test_pad_mode (tuple): padding method and padding shape value, only
            available in test mode. Default is using 'logical_or' with
            127 as padding shape value.
            - 'logical_or': final_shape = input_shape | padding_shape_value
            - 'size_divisor': final_shape = int(
              ceil(input_shape / padding_shape_value) * padding_shape_value)
        test_pad_add_pix (int): Extra padding pixel in test mode. Default 0.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    """


@PIPELINES.register_module()
class CutOut:
    """CutOut operation.
    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.
    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """


@PIPELINES.register_module()
class Mosaic:
    """Mosaic augmentation.
    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.
    .. code:: text
                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+
     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch
    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
        prob (float): Probability of applying this transformation.
            Default to 1.0.
    """


@PIPELINES.register_module()
class MixUp:
    """MixUp data augmentation.
    .. code:: text
                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+
     The mixup transform steps are as follows:
        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.
    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """


@PIPELINES.register_module()
class RandomAffine:
    """Random affine transform data augmentation.
    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.
    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """


@PIPELINES.register_module()
class YOLOXHSVRandomAug:
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.
    Args:
        hue_delta (int): delta of hue. Default: 5.
        saturation_delta (int): delta of saturation. Default: 30.
        value_delta (int): delat of value. Default: 30.
    """


@PIPELINES.register_module()
class CopyPaste:
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:
    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.
    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Default: 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Default: 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Default: 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Default: True.
    """
