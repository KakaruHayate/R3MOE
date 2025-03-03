import json
import pathlib
import random

import numpy as np
import torch.utils.data

from lib.transforms import high_band_mask

class CurveTrainingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: pathlib.Path,
            crop_size: int = 128,
            volume_aug_rate: float = 0.5
    ):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
        with open(root_dir / 'metadata.json', 'r', encoding='utf8') as f:
            self.metadata = json.load(f)
        self.files = []
        with open(root_dir / 'train.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.files.append(root_dir / line.strip())
        self.lengths = np.load(root_dir / 'lengths.npy')
        if len(self.files) != len(self.lengths):
            raise ValueError("Elements in train.txt and lengths.npy do not match!")
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


class CurveValidationDataset2(torch.utils.data.Dataset): # unseen测试集
    def __init__(self, root_dir: pathlib.Path):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
        self.files = []
        with open(root_dir / 'valid2.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.files.append(root_dir / line.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return data['spectrogram'], data['curve']


class CurveValidationDatasetUnlabel(torch.utils.data.Dataset): # unlabel
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
        return data['spectrogram']


class UnlabelTrainingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: pathlib.Path,
            crop_size: int = 128,
            volume_aug_rate: float = 0.5
    ):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
        with open(root_dir / 'metadata.json', 'r', encoding='utf8') as f:
            self.metadata = json.load(f)
        self.files = []
        with open(root_dir / 'train.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.files.append(root_dir / line.strip())
        self.lengths = np.load(root_dir / 'lengths.npy')
        if len(self.files) != len(self.lengths):
            raise ValueError("Elements in train.txt and lengths.npy do not match!")
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
        if data['spectrogram'].shape[0] < self.crop_size:
            # choose another random file
            idx = random.randint(0, len(self.files) - 1)
            return self.__getitem__(idx)
        # volume augmentation
        spectrogram1 = spectrogram
        spectrogram2 = spectrogram
        # 这里不对原始数据扰动，作为teacher的输入
        if random.random() < self.volume_aug_rate:
            spectrogram1 = spectrogram + np.random.uniform(-3, 3)
        if random.random() < self.volume_aug_rate:
            spectrogram2 = spectrogram + np.random.uniform(-3, 3)
        spectrogram = np.clip(spectrogram, a_min=-12, a_max=None)
        spectrogram1 = np.clip(spectrogram1, a_min=-12, a_max=None)
        spectrogram2 = np.clip(spectrogram2, a_min=-12, a_max=None)
        # crop data
        start = random.randint(0, spectrogram.shape[0] - self.crop_size)
        spectrogram = spectrogram[start:start + self.crop_size, :]
        spectrogram1 = spectrogram1[start:start + self.crop_size, :]
        spectrogram2 = spectrogram2[start:start + self.crop_size, :]
        aug_spectrogram1 = spectrogram1
        aug_spectrogram2 = spectrogram2
        # 时域频域都没什么太好的办法，dropout作为构造的兜底
        if random.random() < 0.5:
            aug_spectrogram1 = high_band_mask(aug_spectrogram1)
        if random.random() < 0.5:
            aug_spectrogram2 = high_band_mask(aug_spectrogram2)
        return spectrogram, aug_spectrogram1, aug_spectrogram2
