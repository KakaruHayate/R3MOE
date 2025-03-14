import torch
import torch.nn as nn
import torch.nn.functional as F

# from icecream import ic


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class BiLSTMCurveEstimator(nn.Module):
    """
    BiLSTM-based Mel-spectrogram Prediction Encoderc in Curve Estimation

    Args:
        in_dims (int): Number of input channels, should be same as the number of bins of mel-spectrogram.
        vmin (float): Minimum curve value.
        vmax (float): Maximum curve value.
        hidden_dims (int): Number of hidden dimensions.
        n_layers (int): Number of conformer layers.
        dropout (float): Dropout rate of conv module, default 0.
        num_speakers (int): speaker_cls nums.
    """

    def __init__(
            self,
            in_dims: int,
            vmin: float = 0.,
            vmax: float = 1.,
            hidden_dims: int = 512,
            n_layers: int = 2,
            dropout: float = 0.2,
            num_speakers: int = 50
    ):
        super().__init__()
        self.input_channels = in_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.vmin = vmin
        self.vmax = vmax

        # LSTM
        self.rnn = nn.LSTM(
            input_size=in_dims,
            hidden_size=hidden_dims,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True, 
            dropout=dropout
        )
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1),
            nn.Sigmoid()
        )
        self.speaker_cls = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), 
            nn.AdaptiveAvgPool1d(1)
        )
        self.speaker_linear = nn.Linear(hidden_dims, num_speakers)
        # GRL
        self.grl = GradientReversalLayer()
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

        self.k_emb = nn.Parameter(torch.ones(num_speakers))

    def forward(self, x: torch.Tensor, spk_id: torch.Tensor=None) -> torch.Tensor:
        # k(spk) * f(mel) → seen_target
        # f(mel) → unseen_target
        """
        Args:
            x (torch.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
            spk_id (torch.Tensor): Speaker ids, shape (B, )
        return:
            curve_pred (torch.Tensor): Predicted curve, shape (B, T). # normalized curve
            weighted_curve_pred (torch.Tensor): Predicted curve with weight, shape (B, T). # normalized curve
            spk_emb (torch.Tensor): (B, hidden)
            speaker_logits (torch.Tensor): (B, )
        """
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x) # (B, T, 2*hidden)
        curve_pred = self.output_proj(x) # (B, T, 1)

        if spk_id is not None:
            speaker_feat = self.grl(x) # (B, T, 2*hidden)
            spk_emb = self.speaker_cls(speaker_feat).squeeze(1)  # (B, hidden)
            speaker_logits = self.speaker_linear(spk_emb)  # (B, num_speakers)
            # 斜率参数
            k_selected = self.k_emb[spk_id].view(-1, *([1]*(curve_pred.dim()-1)))
            weighted_curve_pred = (curve_pred * k_selected).squeeze(-1)
        else:
            speaker_logits = None
            spk_emb = None
            weighted_curve_pred = None

        return curve_pred.squeeze(-1), weighted_curve_pred, spk_emb, speaker_logits

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
