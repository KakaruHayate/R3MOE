import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


class PitchAdjustableMelSpectrogram:
    def __init__(
            self,
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            f_min=40,
            f_max=16000,
            n_mels=128,
            center=False,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.center = center

        self.mel_basis = {}
        self.hann_window = {}

    def __call__(self, y, key_shift=0, speed=1.0):
        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length * speed))

        mel_basis_key = f"{self.f_max}_{y.device}"
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        hann_window_key = f"{key_shift}_{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                win_size_new, device=y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((win_size_new - hop_length) // 2),
                int((win_size_new - hop_length+1) // 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length,
            win_length=win_size_new,
            window=self.hann_window[hann_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()

        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))

            spec = spec[:, :size, :] * self.win_size / win_size_new

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)

        return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def curve2latent(curve: torch.Tensor, dims: int, vmin: float, vmax: float, deviation: float):
    interval = (vmax - vmin) / dims
    sigma = deviation / interval
    mu = (curve[:, :, None] - vmin) / interval
    x = torch.arange(dims, device=curve.device).float().reshape(1, 1, -1)  # [1, 1, N]
    latent = ((x - mu) / sigma).pow(2).div(-2).exp()  # gaussian blur, [B, T_n, N]
    return latent


def latent2curve(latent: torch.Tensor, vmin: float, vmax: float, deviation: float):
    dims = int(latent.shape[-1])
    interval = (vmax - vmin) / dims
    width = int(3 * deviation / interval)  # 3 * sigma
    idx = torch.arange(dims, device=latent.device)[None, None, :]  # [1, 1, N]
    idx_values = idx * interval + vmin
    center = torch.argmax(latent, dim=-1, keepdim=True)  # [B, T, 1]
    start = torch.clip(center - width, min=0)  # [B, T, 1]
    end = torch.clip(center + width + 1, max=dims)  # [B, T, 1]
    idx_masks = (idx >= start) & (idx < end)  # [B, T, N]
    weights = latent * idx_masks  # [B, T, N]
    product_sum = torch.sum(weights * idx_values, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    curve = product_sum / (weight_sum + (weight_sum == 0))  # avoid dividing by zero, [B, T]
    curve = torch.clamp(curve, min=vmin, max=vmax)
    return curve
