import json
import pathlib
import random

import numpy as np
import torch.utils.data


class DotDict(dict):
    """
    DotDict, used for config

    Example:
        # >>> config = DotDict({'a': 1, 'b': {'c': 2}}})
        # >>> config.a
        # 1
        # >>> config.b.c
        # 2
    """

    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CurveTrainingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: pathlib.Path,
            crop_size: int = 128,
            volume_aug_rate: float = 0.5
    ):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
        self.files = []
        with open(root_dir / 'train.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.files.append(root_dir / line.strip())
        with open(root_dir / 'metadata.json', 'r', encoding='utf8') as f:
            self.metadata = DotDict(json.load(f))
        self.lengths = np.load(root_dir / 'lengths.npy')
        self.crop_size = crop_size
        self.volume_aug_rate = volume_aug_rate

        self.index_mapping = []
        for i, length in enumerate(self.lengths):
            self.index_mapping.extend([i] * (length // crop_size))
        if len(self.index_mapping) == 0:
            raise ValueError("All data is too short for cropping!")

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        data = np.load(self.files[self.index_mapping[idx]])
        spectrogram = data['spectrogram']
        curve = data['curve']
        if data['spectrogram'].shape[0] < self.crop_size:
            # choose another random file
            idx = random.randint(0, len(self.files) - 1)
            return self.__getitem__(idx)
        # volume augmentation
        if random.random() < self.volume_aug_rate:
            spectrogram = spectrogram + np.random.uniform(-3, 3)
        spectrogram = np.clip(spectrogram, a_min=-12, a_max=None)
        # crop data
        start = random.randint(0, spectrogram.shape[0] - self.crop_size)
        spectrogram = spectrogram[start:start + self.crop_size, :]
        curve = curve[start:start + self.crop_size]
        return spectrogram, curve


class CurveValidationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: pathlib.Path):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
        self.files = []
        with open(root_dir / 'valid.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.files.append(root_dir / line.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return data['spectrogram'], data['curve']
