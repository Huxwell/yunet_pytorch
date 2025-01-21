import copy
import platform
import random
import warnings
from functools import partial
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader
from .samplers import ClassAwareSampler, DistributedGroupSampler, DistributedSampler, GroupSampler, InfiniteBatchSampler, InfiniteGroupBatchSampler
if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import ClassBalancedDataset, ConcatDataset, MultiImageMixDataset, RepeatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset([build_dataset(c, default_args) for c in
            cfg['datasets']], cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(build_dataset(cfg['dataset'], default_args),
            cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(build_dataset(cfg['dataset'],
            default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1,
    dist=True, shuffle=True, seed=None, runner_type='EpochBasedRunner',
    persistent_workers=False, class_aware_sampler=None, **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, Optional): Seed to be used. Default: None.
        runner_type (str): Type of runner. Default: `EpochBasedRunner`
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        class_aware_sampler (dict): Whether to use `ClassAwareSampler`
            during training. Default: None.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
    if runner_type == 'IterBasedRunner':
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(dataset, batch_size,
                world_size, rank, seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(dataset, batch_size,
                world_size, rank, seed=seed, shuffle=False)
        batch_size = 1
        sampler = None
    else:
        if class_aware_sampler is not None:
            num_sample_class = class_aware_sampler.get('num_sample_class', 1)
            sampler = ClassAwareSampler(dataset, samples_per_gpu,
                world_size, rank, seed=seed, num_sample_class=num_sample_class)
        elif dist:
            if shuffle:
                sampler = DistributedGroupSampler(dataset, samples_per_gpu,
                    world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(dataset, world_size, rank,
                    shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu
                ) if shuffle else None
        batch_sampler = None
    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    if TORCH_VERSION != 'parrots' and digit_version(TORCH_VERSION
        ) >= digit_version('1.7.0'):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn(
            'persistent_workers is invalid because your pytorch version is lower than 1.7.0'
            )
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=
        sampler, num_workers=num_workers, batch_sampler=batch_sampler,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=kwargs.pop('pin_memory', False), worker_init_fn=init_fn,
        **kwargs)
    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
