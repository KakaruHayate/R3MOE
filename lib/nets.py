import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMCurveEstimator(nn.Module):
    """
    BiLSTM-based Mel-spectrogram Prediction Encoderc in Curve Estimation

    Args:
        in_dims (int): Number of input channels, should be same as the number of bins of mel-spectrogram.
        vmin (float): Minimum curve value.
        vmax (float): Maximum curve value.
        hidden_dims (int): Number of hidden dimensions.
        n_layers (int): Number of conformer layers.
        conv_dropout (float): Dropout rate of conv module, default 0.
    """

    def __init__(
            self,
            in_dims: int,
            vmin: float = 0.,
            vmax: float = 1.,
            hidden_dims: int = 512,
            n_layers: int = 2,
            conv_dropout: float = 0.2,
    ):
        super().__init__()
        self.input_channels = in_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.vmin = vmin
        self.vmax = vmax

        # Input stack, convert mel-spectrogram to hidden_dims
        self.input_stack = nn.Sequential(
            nn.Conv1d(in_dims, hidden_dims, 3, 1, 1, bias=False),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(conv_dropout), 
            nn.Conv1d(hidden_dims, hidden_dims, 3, 1, 1, bias=False)
        )
        # LSTM
        self.rnn = nn.LSTM(
            input_size=hidden_dims,
            hidden_size=hidden_dims,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            nn.Linear(hidden_dims, 1),
            nn.Sigmoid()
        )
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
        return:
            torch.Tensor: Predicted curve, shape (B, T).
        """
        self.rnn.flatten_parameters()
        x = self.input_stack(x.transpose(-1, -2)).transpose(-1, -2)
        x, _ = self.rnn(x)
        x = self.output_proj(x)
        return x.squeeze(-1)  # normalized curve (B, T)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input curve, shape (B, T).
        return:
            torch.Tensor: Normalized curve, shape (B, T).
        """
        x = (x - self.vmin) / (self.vmax - self.vmin)
        x = x.clamp(0., 1.)
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input normalized curve, shape (B, T).
        return:
            torch.Tensor: Curve, shape (B, T).
        """
        x = x * (self.vmax - self.vmin) + self.vmin
        return x

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        curve = self.denormalize(x)
        return curve
