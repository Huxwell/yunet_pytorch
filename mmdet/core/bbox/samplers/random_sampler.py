import torch
from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self, num, pos_fraction, neg_pos_ub=-1,
        add_gt_as_proposals=True, **kwargs):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/bbox/samplers/random_sampler.py:L21 ')
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        print('Filip YuNet Minify: Function fidx=1 random_choice called in mmdet/core/bbox/samplers/random_sampler.py:L32 ')
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        print('Filip YuNet Minify: Function fidx=2 _sample_pos called in mmdet/core/bbox/samplers/random_sampler.py:L64 ')
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        print('Filip YuNet Minify: Function fidx=3 _sample_neg called in mmdet/core/bbox/samplers/random_sampler.py:L74 ')
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
