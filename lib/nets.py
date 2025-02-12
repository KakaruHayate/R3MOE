import torch
import torch.nn as nn

# import weight_norm from different version of pytorch
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

from . import transforms
from .model_conformer_naive import ConformerNaiveEncoder


class CFNaiveCurveEstimator(nn.Module):
    """
    Conformer-based Mel-spectrogram Prediction Encoderc in Fast Context-based Pitch Estimation

    Args:
        in_dims (int): Number of input channels, should be same as the number of bins of mel-spectrogram.
        out_dims (int): Number of output dimensions, also class numbers.
        vmin (float): Minimum curve value.
        vmax (float): Maximum curve value.
        deviation (float): Deviation of curve value for gaussian blurring.
        hidden_dims (int): Number of hidden dimensions.
        n_layers (int): Number of conformer layers.
        use_fa_norm (bool): Whether to use fast attention norm, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        attn_dropout (float): Dropout rate of attention module, default 0.
    """

    def __init__(
            self,
            in_dims: int,
            out_dims: int = 256,
            vmin: float = 0.,
            vmax: float = 1.,
            deviation: float = 0.01,
            hidden_dims: int = 512,
            n_layers: int = 6,
            n_heads: int = 8,
            use_fa_norm: bool = False,
            conv_only: bool = False,
            conv_dropout: float = 0.,
            attn_dropout: float = 0.,
    ):
        super().__init__()
        self.input_channels = in_dims
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vmin = vmin
        self.vmax = vmax
        self.deviation = deviation
        self.use_fa_norm = use_fa_norm

        # Input stack, convert mel-spectrogram to hidden_dims
        self.input_stack = nn.Sequential(
            nn.Conv1d(in_dims, hidden_dims, 3, 1, 1),
            nn.GroupNorm(4, hidden_dims),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1)
        )
        # Conformer Encoder
        self.net = ConformerNaiveEncoder(
            num_layers=n_layers,
            num_heads=n_heads,
            dim_model=hidden_dims,
            use_norm=use_fa_norm,
            conv_only=conv_only,
            conv_dropout=conv_dropout,
            atten_dropout=attn_dropout,
        )
        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dims)
        # Output stack, convert hidden_dims to out_dims
        self.output_proj = weight_norm(
            nn.Linear(hidden_dims, out_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
        return:
            torch.Tensor: Predicted curve latent, shape (B, T, out_dims).
        """
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.net(x)
        x = self.norm(x)
        x = self.output_proj(x)
        x = torch.sigmoid(x)
        return x  # latent (B, T, out_dims)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input curve, shape (B, T).
        return:
            torch.Tensor: Predicted curve latent, shape (B, T, out_dims).
        """
        x = transforms.curve2latent(x, dims=self.out_dims, vmin=self.vmin, vmax=self.vmax, deviation=self.deviation)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input curve latent, shape (B, T, out_dims).
        return:
            torch.Tensor: Predicted curve, shape (B, T).
        """
        x = transforms.latent2curve(x, vmin=self.vmin, vmax=self.vmax, deviation=self.deviation)
        return x

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.forward(x)
        curve = self.decode(latent)
        return curve
