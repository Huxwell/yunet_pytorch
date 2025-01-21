import math
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
# from mmdet.core.utils import sync_random_seed


class ClassAwareSampler(Sampler):
    """Sampler that restricts data loading to the label of the dataset.

    A class-aware sampling strategy to effectively tackle the
    non-uniform class distribution. The length of the training data is
    consistent with source data. Simple improvements based on `Relay
    Backpropagation for Effective Learning of Deep Convolutional
    Neural Networks <https://arxiv.org/abs/1512.05830>`_

    The implementation logic is referred to
    https://github.com/Sense-X/TSD/blob/master/mmdet/datasets/samplers/distributed_classaware_sampler.py

    Args:
        dataset: Dataset used for sampling.
        samples_per_gpu (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU.
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
        num_sample_class (int): The number of samples taken from each
            per-label list. Default: 1
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=
        None, seed=0, num_sample_class=1):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/datasets/samplers/class_aware_sampler.py:L40 ')
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.samples_per_gpu = samples_per_gpu
        self.rank = rank
        self.epoch = 0
        # self.seed = sync_random_seed(seed)
        self.seed = seed
        assert num_sample_class > 0 and isinstance(num_sample_class, int)
        self.num_sample_class = num_sample_class
        assert hasattr(dataset, 'get_cat2imgs'
            ), 'dataset must have `get_cat2imgs` function'
        self.cat_dict = dataset.get_cat2imgs()
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.
            num_replicas / self.samples_per_gpu)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas
        self.num_cat_imgs = [len(x) for x in self.cat_dict.values()]
        self.valid_cat_inds = [i for i, length in enumerate(self.
            num_cat_imgs) if length != 0]
        self.num_classes = len(self.valid_cat_inds)

    def __iter__(self):
        print('Filip YuNet Minify: Function fidx=1 __iter__ called in mmdet/datasets/samplers/class_aware_sampler.py:L85 ')
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        label_iter_list = RandomCycleIter(self.valid_cat_inds, generator=g)
        data_iter_dict = dict()
        for i in self.valid_cat_inds:
            data_iter_dict[i] = RandomCycleIter(self.cat_dict[i], generator=g)

        def gen_cat_img_inds(cls_list, data_dict, num_sample_cls):
            """Traverse the categories and extract `num_sample_cls` image
            indexes of the corresponding categories one by one."""
            id_indices = []
            for _ in range(len(cls_list)):
                cls_idx = next(cls_list)
                for _ in range(num_sample_cls):
                    id = next(data_dict[cls_idx])
                    id_indices.append(id)
            return id_indices
        num_bins = int(math.ceil(self.total_size * 1.0 / self.num_classes /
            self.num_sample_class))
        indices = []
        for i in range(num_bins):
            indices += gen_cat_img_inds(label_iter_list, data_iter_dict,
                self.num_sample_class)
        if len(indices) >= self.total_size:
            indices = indices[:self.total_size]
        else:
            indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        print('Filip YuNet Minify: Function fidx=2 __len__ called in mmdet/datasets/samplers/class_aware_sampler.py:L131 ')
        return self.num_samples

    def set_epoch(self, epoch):
        print('Filip YuNet Minify: Function fidx=3 set_epoch called in mmdet/datasets/samplers/class_aware_sampler.py:L134 ')
        self.epoch = epoch


class RandomCycleIter:
    """Shuffle the list and do it again after the list have traversed.

    The implementation logic is referred to
    https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/datasets/loader/sampler.py

    Example:
        >>> label_list = [0, 1, 2, 4, 5]
        >>> g = torch.Generator()
        >>> g.manual_seed(0)
        >>> label_iter_list = RandomCycleIter(label_list, generator=g)
        >>> index = next(label_iter_list)
    Args:
        data (list or ndarray): The data that needs to be shuffled.
        generator: An torch.Generator object, which is used in setting the seed
            for generating random numbers.
    """

    def __init__(self, data, generator=None):
        print('Filip YuNet Minify: Function fidx=4 __init__ called in mmdet/datasets/samplers/class_aware_sampler.py:L156 ')
        self.data = data
        self.length = len(data)
        self.index = torch.randperm(self.length, generator=generator).numpy()
        self.i = 0
        self.generator = generator

    def __iter__(self):
        print('Filip YuNet Minify: Function fidx=5 __iter__ called in mmdet/datasets/samplers/class_aware_sampler.py:L163 ')
        return self

    def __len__(self):
        print('Filip YuNet Minify: Function fidx=6 __len__ called in mmdet/datasets/samplers/class_aware_sampler.py:L166 ')
        return len(self.data)

    def __next__(self):
        print('Filip YuNet Minify: Function fidx=7 __next__ called in mmdet/datasets/samplers/class_aware_sampler.py:L169 ')
        if self.i == self.length:
            self.index = torch.randperm(self.length, generator=self.generator
                ).numpy()
            self.i = 0
        idx = self.data[self.index[self.i]]
        self.i += 1
        return idx
