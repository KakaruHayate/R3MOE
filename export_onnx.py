import argparse
import pathlib
import sys
import yaml

import torch
import torch.nn as nn
import torchaudio
import onnx
import onnxslim

from lib.nets import BiLSTMCurveEstimator
from librosa.filters import mel as librosa_mel_fn

def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect
    sig = inspect.signature(kwarg_obj)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return dict_to_filter.copy()
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.KEYWORD_ONLY
    ]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if
                     filter_key in dict_to_filter}
    return filtered_dict

class ONNXMelSpectrogram(nn.Module):
    """专门为 ONNX 导出改造的 Mel 频谱提取模块，避免动态字典和设备判断干扰"""
    def __init__(self, sample_rate, n_fft, win_size, hop_length, f_min, f_max, n_mels, center=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.center = center

        # 预计算 Mel Basis 并注册为 buffer，确保能够作为常量图导出
        mel_basis = librosa_mel_fn(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        # 预计算 Hann 窗
        self.register_buffer("hann_window", torch.hann_window(self.win_size))
        
        # 预计算 Padding 尺寸
        self.pad_left = int((self.win_size - self.hop_length) // 2)
        self.pad_right = int((self.win_size - self.hop_length + 1) // 2)

    def forward(self, y):
        # y: [Batch, Time]
        # Reflection Padding
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (self.pad_left, self.pad_right),
            mode="reflect",
        ).squeeze(1)

        # STFT (ONNX 导出不支持 complex 数据类型，改为输出实部/虚部并手动求模)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,  # 改为 False
        )
        # 此时 spec 的 shape 为 [B, n_fft/2+1, T, 2]，最后一维是实部和虚部
        # 手动计算幅值 (Magnitude) = sqrt(real^2 + imag^2)，加上 1e-9 防止出现梯度/计算异常
        spec = torch.sqrt(spec.pow(2).sum(dim=-1) + 1e-9)

        # Mel 滤波
        mel_spec = torch.matmul(self.mel_basis, spec)
        return mel_spec

class CurveEstimatorONNXWrapper(nn.Module):
    """封装整个 Pipeline：16kHz输入 -> (重采样) -> Mel -> 压缩 -> 模型 -> 反归一化"""
    def __init__(self, model, dataset_args, input_sr=44100):
        super().__init__()
        self.model_sr = dataset_args["sample_rate"]
        self.input_sr = input_sr
        
        # 如果模型原生采样率不是 16kHz，加入重采样节点
        if self.model_sr != self.input_sr:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.input_sr,
                new_freq=self.model_sr,
                lowpass_filter_width=128
            )
        else:
            self.resampler = None
            
        self.mel_transform = ONNXMelSpectrogram(
            sample_rate=self.model_sr,
            n_fft=dataset_args["win_size"],
            win_size=dataset_args["win_size"],
            hop_length=dataset_args["hop_size"],
            f_min=dataset_args["f_min"],
            f_max=dataset_args["f_max"],
            n_mels=dataset_args["mel_bins"],
            center=True
        )
        self.model = model

    def forward(self, waveform):
        # waveform: [B, T]
        if self.resampler is not None:
            waveform = self.resampler(waveform)
            
        # 1. 提取 Mel 频谱
        mel_spec = self.mel_transform(waveform)
        
        # 2. 动态范围压缩 (DRC) 并转置 => [B, T, Mel]
        mel = torch.log(torch.clamp(mel_spec, min=1e-5)).transpose(1, 2)
        
        # 3. 模型预测出归一化曲线
        norm_curve = self.model(mel)
        
        # 4. 反归一化 (Denormalize)
        curve = self.model.denormalize(norm_curve)
        
        return curve

def export_onnx(model_path: pathlib.Path, output_path: pathlib.Path = None):
    print(f"正在加载模型配置与权重: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    config_path = model_path.with_name("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        dataset_args, model_args = config["dataset_args"], config["model_args"]

    # 初始化原版模型
    base_model = BiLSTMCurveEstimator(**filter_kwargs(model_args, BiLSTMCurveEstimator))
    
    # 过滤 k_filter (多说话人 Embedding)，按 eval.py 逻辑处理
    state_dict = torch.load(model_path, map_location="cpu")
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'k_filter' not in k}
    base_model.load_state_dict(filtered_state_dict, strict=False)
    
    # 包装 Pipeline
    wrapper = CurveEstimatorONNXWrapper(model=base_model, dataset_args=dataset_args, input_sr=44100)
    wrapper.eval()
    
    # 构建 Dummy 伪输入 (16kHz 单声道音频，假设时长 2 秒)
    dummy_waveform = torch.randn(1, 16000 * 2)
    
    if output_path is None:
        output_path = model_path.with_suffix(".onnx")
        
    print(f"开始导出 ONNX (opset=17)...")
    # STFT 算子需要 Opset >= 17 才原生支持 Complex Return
    torch.onnx.export(
        wrapper,
        (dummy_waveform,),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["curve"],
        dynamic_axes={
            "waveform": {0: "batch_size", 1: "time"},
            "curve": {0: "batch_size", 1: "frames"}
        }
    )
    print(f"原生 ONNX 已导出至: {output_path}")

    # 使用 onnxslim 化简
    try:
        print(f"使用 onnxslim 进行模型化简...")
        slim_model = onnxslim.slim(str(output_path))
        onnx.save(slim_model, str(output_path))
        print(f"✨ 化简完成！最终模型路径: {output_path}")
    except Exception as e:
        print(f"化简失败，跳过该步骤: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CurveEstimator to ONNX with built-in Mel Spectrogram.")
    parser.add_argument("-m", "--model", type=pathlib.Path, required=True,
                        help="Path to the PyTorch model checkpoint (.pt or .pth).")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None,
                        help="Output path for the ONNX file. Defaults to model directory.")
    
    args = parser.parse_args()

    try:
        export_onnx(args.model, args.output)
    except Exception as e:
        print(f"\n[错误]: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)