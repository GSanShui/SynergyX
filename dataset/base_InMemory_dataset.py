from itertools import repeat
from torch_geometric.data import Data, Dataset, InMemoryDataset
from abc import ABCMeta, abstractmethod
import os.path as osp
import os
import numpy as np
from tqdm import tqdm


class BaseInMemoryDataset(InMemoryDataset, metaclass=ABCMeta):
    r"""

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, transform, pre_transform, test_mode=False):
        super().__init__(root, transform, pre_transform)
        self.test_mode = test_mode
        self.transform = transform
        self.pre_transform = pre_transform

    @abstractmethod
    def process(self):
        """Abstract function for loading data

        All subclasses should overwrite this function
        """

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        return []

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


