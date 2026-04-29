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
        volume_aug_rate: float = 0.5,
        use_spk_id: bool = False
    ):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)

        with open(root_dir / 'metadata.json', 'r', encoding='utf8') as f:
            self.metadata = json.load(f)

        self.use_spk_id = use_spk_id
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.volume_aug_rate = volume_aug_rate

        # 读取文件列表
        self.files = []
        with open(root_dir / 'train.txt', 'r', encoding='utf8') as f:
            for line in f:
                file_path = root_dir / line.strip()
                self.files.append(file_path)

        # 读取长度（帧数）
        self.lengths = np.load(root_dir / 'lengths.npy')
        assert len(self.files) == len(self.lengths), \
            "Mismatch between train.txt and lengths.npy!"

        # 只保留长度 >= crop_size 的有效文件
        self.valid_indices = [i for i, l in enumerate(self.lengths) if l >= self.crop_size]
        if not self.valid_indices:
            raise ValueError("No files with length >= crop_size!")

        self.valid_lengths = self.lengths[self.valid_indices]   # 有效长度数组
        self.total_valid_frames = int(sum(self.valid_lengths))  # 总帧数

        # 计算 epoch 总采样数（与原逻辑一致）
        self.epoch_samples = self.total_valid_frames // self.crop_size

        # 处理 speaker ID
        if self.use_spk_id:
            with open(root_dir / 'spk_mapping.json', 'r', encoding='utf-8') as f:
                spk_mapping = json.load(f)
            self.spk_ids = []    # 只保留有效文件的 speaker id
            for i in self.valid_indices:
                relative_path = self.files[i].relative_to(root_dir)
                spk_name = relative_path.parts[0]
                self.spk_ids.append(spk_mapping[spk_name])

    def __len__(self):
        # epoch 长度 = 总有效帧数 // crop_size
        return self.epoch_samples

    def __getitem__(self, idx):
        # idx 在此方案中被忽略，每次完全随机采样
        # 1. 按帧数加权随机选择一个有效文件
        #    使用累积分分布函数高效采样
        weights = self.valid_lengths / self.total_valid_frames
        i = np.random.choice(len(self.valid_indices), p=weights)

        file_idx = self.valid_indices[i]
        data = np.load(self.files[file_idx])
        spectrogram = data['spectrogram']    # (T, mel)
        curve = data['curve']                # (T,)
        
        #mask = curve > 0
        #curve[mask] -= 0.1

        # 2. 随机裁剪
        T = spectrogram.shape[0]
        start = random.randint(0, T - self.crop_size)
        cropped_spectrogram = spectrogram[start:start + self.crop_size, :]
        cropped_curve = curve[start:start + self.crop_size]

        # 3. 音量增强
        if random.random() < self.volume_aug_rate:
            cropped_spectrogram = cropped_spectrogram + np.random.uniform(-3, 3)
        cropped_spectrogram = np.clip(cropped_spectrogram, a_min=-12, a_max=None)

        # 4. 返回
        if self.use_spk_id:
            return cropped_spectrogram, cropped_curve, self.spk_ids[i]
        else:
            return cropped_spectrogram, cropped_curve, None


class OldCurveTrainingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: pathlib.Path,
            crop_size: int = 128,
            volume_aug_rate: float = 0.5, 
            use_spk_id: bool = False
    ):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
        with open(root_dir / 'metadata.json', 'r', encoding='utf8') as f:
            self.metadata = json.load(f)
        # spk_ids
        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            with open(root_dir / 'spk_mapping.json', 'r', encoding='utf-8') as f:
                self.spk_mapping = json.load(f)
            self.spk_ids = []
        self.files = []
        with open(root_dir / 'train.txt', 'r', encoding='utf8') as f:
            for line in f:
                file_path = root_dir / line.strip()
                self.files.append(file_path)
                
                # 预计算spk_ids
                if self.use_spk_id:
                    relative_path = file_path.relative_to(root_dir)
                    spk_name = relative_path.parts[0]
                    self.spk_ids.append(self.spk_mapping[spk_name])
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
        file_idx = self.index_mapping[idx]
        data = np.load(self.files[file_idx])
        spectrogram = data['spectrogram']
        curve = data['curve']
        
        #mask = curve > 0
        #curve[mask] -= 0.1
        
        if data['spectrogram'].shape[0] < self.crop_size:
            # choose another random file
            idx = random.randint(0, len(self.files) - 1)
            return self.__getitem__(idx)
        # volume augmentation
        if random.random() < self.volume_aug_rate:
            spectrogram = spectrogram + np.random.uniform(-3, 3)
        spectrogram = np.clip(spectrogram, a_min=-12, a_max=None)
        start = random.randint(0, spectrogram.shape[0] - self.crop_size)
        cropped_spectrogram = spectrogram[start:start + self.crop_size, :]
        cropped_curve = curve[start:start + self.crop_size]
        if self.use_spk_id:
            return cropped_spectrogram, cropped_curve, self.spk_ids[file_idx]
        return cropped_spectrogram, cropped_curve, None


class CurveValidationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: pathlib.Path, use_spk_id: bool = False):
        if not isinstance(root_dir, pathlib.Path):
            root_dir = pathlib.Path(root_dir)
            
        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            with open(root_dir / 'spk_mapping.json', 'r', encoding='utf-8') as f:
                self.spk_mapping = json.load(f)
            self.spk_ids = []

        self.files = []
        with open(root_dir / 'valid.txt', 'r', encoding='utf8') as f:
            for line in f:
                file_path = root_dir / line.strip()
                self.files.append(file_path)
                if self.use_spk_id:
                    relative_path = file_path.relative_to(root_dir)
                    spk_name = relative_path.parts[0]
                    self.spk_ids.append(self.spk_mapping[spk_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        curve = data['curve']
        
        #mask = curve > 0
        #curve[mask] -= 0.1
        
        if self.use_spk_id:
            return data['spectrogram'], curve, self.spk_ids[idx]
        return data['spectrogram'], curve, None


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
        
        curve = data['curve']
        
        #mask = curve > 0
        #curve[mask] -= 0.1
        
        return data['spectrogram'], curve


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

        self.root_dir = root_dir
        self.crop_size = crop_size
        self.volume_aug_rate = volume_aug_rate

        # 读取文件列表
        self.files = []
        with open(root_dir / 'train.txt', 'r', encoding='utf8') as f:
            for line in f:
                self.files.append(root_dir / line.strip())

        # 读取长度（帧数）
        self.lengths = np.load(root_dir / 'lengths.npy')
        if len(self.files) != len(self.lengths):
            raise ValueError("Elements in train.txt and lengths.npy do not match!")

        # 只保留长度 >= crop_size 的有效文件
        self.valid_indices = [i for i, l in enumerate(self.lengths) if l >= self.crop_size]
        if not self.valid_indices:
            raise ValueError("All data is too short for cropping!")

        self.valid_lengths = self.lengths[self.valid_indices]   # 有效帧数数组
        self.total_valid_frames = int(sum(self.valid_lengths))  # 总有效帧数

        # 每个 epoch 的采样次数（总帧数 // crop_size）
        self.epoch_samples = self.total_valid_frames // self.crop_size

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, idx):
        # 按帧数加权随机抽样一个有效文件
        weights = self.valid_lengths / self.total_valid_frames
        i = np.random.choice(len(self.valid_indices), p=weights)

        file_idx = self.valid_indices[i]
        data = np.load(self.files[file_idx])
        spectrogram = data['spectrogram']    # (T, mel)

        # 随机裁剪起点
        T = spectrogram.shape[0]
        start = random.randint(0, T - self.crop_size)

        # 获取裁剪区域
        cropped_spectrogram = spectrogram[start:start + self.crop_size, :]

        # 创建两个增强副本（不修改原始谱图作为teacher输入）
        spectrogram1 = cropped_spectrogram.copy()
        spectrogram2 = cropped_spectrogram.copy()

        # 音量增强分别独立应用
        if random.random() < self.volume_aug_rate:
            spectrogram1 += np.random.uniform(-3, 3)
        if random.random() < self.volume_aug_rate:
            spectrogram2 += np.random.uniform(-3, 3)

        # 全局数值裁剪
        spectrogram1 = np.clip(spectrogram1, a_min=-12, a_max=None)
        spectrogram2 = np.clip(spectrogram2, a_min=-12, a_max=None)
        cropped_spectrogram = np.clip(cropped_spectrogram, a_min=-12, a_max=None)

        # 原注释的 dropout 增强可在此处添加

        return cropped_spectrogram, spectrogram1, spectrogram2


class OldUnlabelTrainingDataset(torch.utils.data.Dataset):
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
        # if random.random() < 0.5:
        #     aug_spectrogram1 = high_band_mask(aug_spectrogram1)
        # if random.random() < 0.5:
        #     aug_spectrogram2 = high_band_mask(aug_spectrogram2)
        return spectrogram, aug_spectrogram1, aug_spectrogram2
