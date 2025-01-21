import copy
import numpy as np
import torch
from mmdet.utils.util_mixins import NiceRepr


class GeneralData(NiceRepr):
    """A general data structure of OpenMMlab.

    A data structure that stores the meta information,
    the annotations of the images or the model predictions,
    which can be used in communication between components.

    The attributes in `GeneralData` are divided into two parts,
    the `meta_info_fields` and the `data_fields` respectively.

        - `meta_info_fields`: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. All attributes in
          it are immutable once set,
          but the user can add new meta information with
          `set_meta_info` function, all information can be accessed
          with methods `meta_info_keys`, `meta_info_values`,
          `meta_info_items`.

        - `data_fields`: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          `.` , `[]`, `in`, `del`, `pop(str)` `get(str)`, `keys()`,
          `values()`, `items()`. Users can also apply tensor-like methods
          to all obj:`torch.Tensor` in the `data_fileds`,
          such as `.cuda()`, `.cpu()`, `.numpy()`, `device`, `.to()`
          `.detach()`, `.numpy()`

    Args:
        meta_info (dict, optional): A dict contains the meta information
            of single image. such as `img_shape`, `scale_factor`, etc.
            Default: None.
        data (dict, optional): A dict contains annotations of single image or
            model predictions. Default: None.

    Examples:
        >>> from mmdet.core import GeneralData
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = GeneralData(meta_info=img_meta)
        >>> img_shape in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> instance_data["det_scores"] = torch.Tensor([0.01, 0.1, 0.2, 0.3])
        >>> print(results)
        <GeneralData(

          META INFORMATION
        img_shape: (800, 1196, 3)
        pad_shape: (800, 1216, 3)

          DATA FIELDS
        shape of det_labels: torch.Size([4])
        shape of det_scores: torch.Size([4])

        ) at 0x7f84acd10f90>
        >>> instance_data.det_scores
        tensor([0.0100, 0.1000, 0.2000, 0.3000])
        >>> instance_data.det_labels
        tensor([0, 1, 2, 3])
        >>> instance_data['det_labels']
        tensor([0, 1, 2, 3])
        >>> 'det_labels' in instance_data
        True
        >>> instance_data.img_shape
        (800, 1196, 3)
        >>> 'det_scores' in instance_data
        True
        >>> del instance_data.det_scores
        >>> 'det_scores' in instance_data
        False
        >>> det_labels = instance_data.pop('det_labels', None)
        >>> det_labels
        tensor([0, 1, 2, 3])
        >>> 'det_labels' in instance_data
        >>> False
    """

    def __init__(self, meta_info=None, data=None):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/core/data_structures/general_data.py:L87 ')
        self._meta_info_fields = set()
        self._data_fields = set()
        if meta_info is not None:
            self.set_meta_info(meta_info=meta_info)
        if data is not None:
            self.set_data(data)

    def set_meta_info(self, meta_info):
        print('Filip YuNet Minify: Function fidx=1 set_meta_info called in mmdet/core/data_structures/general_data.py:L97 ')
        """Add meta information.

        Args:
            meta_info (dict): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
        """
        assert isinstance(meta_info, dict
            ), f'meta should be a `dict` but get {meta_info}'
        meta = copy.deepcopy(meta_info)
        for k, v in meta.items():
            if k in self._meta_info_fields:
                ori_value = getattr(self, k)
                if isinstance(ori_value, (torch.Tensor, np.ndarray)):
                    if (ori_value == v).all():
                        continue
                    else:
                        raise KeyError(
                            f'img_meta_info {k} has been set as {getattr(self, k)} before, which is immutable '
                            )
                elif ori_value == v:
                    continue
                else:
                    raise KeyError(
                        f'img_meta_info {k} has been set as {getattr(self, k)} before, which is immutable '
                        )
            else:
                self._meta_info_fields.add(k)
                self.__dict__[k] = v

    def set_data(self, data):
        print('Filip YuNet Minify: Function fidx=2 set_data called in mmdet/core/data_structures/general_data.py:L129 ')
        """Update a dict to `data_fields`.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions. Default: None.
        """
        assert isinstance(data, dict
            ), f'meta should be a `dict` but get {data}'
        for k, v in data.items():
            self.__setattr__(k, v)

    def new(self, meta_info=None, data=None):
        print('Filip YuNet Minify: Function fidx=3 new called in mmdet/core/data_structures/general_data.py:L141 ')
        """Return a new results with same image meta information.

        Args:
            meta_info (dict, optional): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
            data (dict, optional): A dict contains annotations of image or
                model predictions. Default: None.
        """
        new_data = self.__class__()
        new_data.set_meta_info(dict(self.meta_info_items()))
        if meta_info is not None:
            new_data.set_meta_info(meta_info)
        if data is not None:
            new_data.set_data(data)
        return new_data

    def keys(self):
        print('Filip YuNet Minify: Function fidx=4 keys called in mmdet/core/data_structures/general_data.py:L159 ')
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        return [key for key in self._data_fields]

    def meta_info_keys(self):
        print('Filip YuNet Minify: Function fidx=5 meta_info_keys called in mmdet/core/data_structures/general_data.py:L166 ')
        """
        Returns:
            list: Contains all keys in meta_info_fields.
        """
        return [key for key in self._meta_info_fields]

    def values(self):
        print('Filip YuNet Minify: Function fidx=6 values called in mmdet/core/data_structures/general_data.py:L173 ')
        """
        Returns:
            list: Contains all values in data_fields.
        """
        return [getattr(self, k) for k in self.keys()]

    def meta_info_values(self):
        print('Filip YuNet Minify: Function fidx=7 meta_info_values called in mmdet/core/data_structures/general_data.py:L180 ')
        """
        Returns:
            list: Contains all values in meta_info_fields.
        """
        return [getattr(self, k) for k in self.meta_info_keys()]

    def items(self):
        print('Filip YuNet Minify: Function fidx=8 items called in mmdet/core/data_structures/general_data.py:L187 ')
        for k in self.keys():
            yield k, getattr(self, k)

    def meta_info_items(self):
        print('Filip YuNet Minify: Function fidx=9 meta_info_items called in mmdet/core/data_structures/general_data.py:L191 ')
        for k in self.meta_info_keys():
            yield k, getattr(self, k)

    def __setattr__(self, name, val):
        print('Filip YuNet Minify: Function fidx=10 __setattr__ called in mmdet/core/data_structures/general_data.py:L195 ')
        if name in ('_meta_info_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, val)
            else:
                raise AttributeError(
                    f'{name} has been used as a private attribute, which is immutable. '
                    )
        else:
            if name in self._meta_info_fields:
                raise AttributeError(
                    f'`{name}` is used in meta information,which is immutable')
            self._data_fields.add(name)
            super().__setattr__(name, val)

    def __delattr__(self, item):
        print('Filip YuNet Minify: Function fidx=11 __delattr__ called in mmdet/core/data_structures/general_data.py:L211 ')
        if item in ('_meta_info_fields', '_data_fields'):
            raise AttributeError(
                f'{item} has been used as a private attribute, which is immutable. '
                )
        if item in self._meta_info_fields:
            raise KeyError(
                f'{item} is used in meta information, which is immutable.')
        super().__delattr__(item)
        if item in self._data_fields:
            self._data_fields.remove(item)
    __setitem__ = __setattr__
    __delitem__ = __delattr__

    def __getitem__(self, name):
        print('Filip YuNet Minify: Function fidx=12 __getitem__ called in mmdet/core/data_structures/general_data.py:L228 ')
        return getattr(self, name)

    def get(self, *args):
        print('Filip YuNet Minify: Function fidx=13 get called in mmdet/core/data_structures/general_data.py:L231 ')
        assert len(args) < 3, '`get` get more than 2 arguments'
        return self.__dict__.get(*args)

    def pop(self, *args):
        print('Filip YuNet Minify: Function fidx=14 pop called in mmdet/core/data_structures/general_data.py:L235 ')
        assert len(args) < 3, '`pop` get more than 2 arguments'
        name = args[0]
        if name in self._meta_info_fields:
            raise KeyError(
                f'{name} is a key in meta information, which is immutable')
        if args[0] in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f'{args[0]}')

    def __contains__(self, item):
        print('Filip YuNet Minify: Function fidx=15 __contains__ called in mmdet/core/data_structures/general_data.py:L252 ')
        return item in self._data_fields or item in self._meta_info_fields

    def to(self, *args, **kwargs):
        print('Filip YuNet Minify: Function fidx=16 to called in mmdet/core/data_structures/general_data.py:L257 ')
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
            new_data[k] = v
        return new_data

    def cpu(self):
        print('Filip YuNet Minify: Function fidx=17 cpu called in mmdet/core/data_structures/general_data.py:L267 ')
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            new_data[k] = v
        return new_data

    def mlu(self):
        print('Filip YuNet Minify: Function fidx=18 mlu called in mmdet/core/data_structures/general_data.py:L277 ')
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.mlu()
            new_data[k] = v
        return new_data

    def cuda(self):
        print('Filip YuNet Minify: Function fidx=19 cuda called in mmdet/core/data_structures/general_data.py:L287 ')
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_data[k] = v
        return new_data

    def detach(self):
        print('Filip YuNet Minify: Function fidx=20 detach called in mmdet/core/data_structures/general_data.py:L297 ')
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            new_data[k] = v
        return new_data

    def numpy(self):
        print('Filip YuNet Minify: Function fidx=21 numpy called in mmdet/core/data_structures/general_data.py:L307 ')
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            new_data[k] = v
        return new_data

    def __nice__(self):
        print('Filip YuNet Minify: Function fidx=22 __nice__ called in mmdet/core/data_structures/general_data.py:L316 ')
        repr = '\n \n  META INFORMATION \n'
        for k, v in self.meta_info_items():
            repr += f'{k}: {v} \n'
        repr += '\n   DATA FIELDS \n'
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                repr += f'shape of {k}: {v.shape} \n'
            else:
                repr += f'{k}: {v} \n'
        return repr + '\n'
