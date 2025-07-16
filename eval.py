# eval.py
import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.transforms
import yaml
from scipy.io import wavfile

from lib.nets import BiLSTMCurveEstimator
from lib.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch


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


class CurveEstimator:
    def __init__(self, model_path: pathlib.Path, device: str):
        if not isinstance(model_path, pathlib.Path):
            model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Curve estimator model path {model_path} does not exist.")
        config_path = model_path.with_name("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Curve estimator config path {config_path} does not exist.")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            dataset_args, model_args = config["dataset_args"], config["model_args"]
        self.device = device
        self.mel_spec_transform = PitchAdjustableMelSpectrogram(
            sample_rate=dataset_args["sample_rate"],
            n_fft=dataset_args["win_size"],
            win_length=dataset_args["win_size"],
            hop_length=dataset_args["hop_size"],
            f_min=dataset_args["f_min"],
            f_max=dataset_args["f_max"],
            n_mels=dataset_args["mel_bins"],
            center=True
        )
        self.model = BiLSTMCurveEstimator(
            **filter_kwargs(model_args, BiLSTMCurveEstimator)
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.model.to(device)
        self.resample_kernels = {}

    @torch.no_grad()
    def estimate(self, waveform: np.ndarray, sr: int, length: int) -> np.ndarray:
        waveform = torch.from_numpy(waveform).float().to(self.device).unsqueeze(0)
        if sr != self.mel_spec_transform.sample_rate:
            if sr not in self.resample_kernels:
                self.resample_kernels[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.mel_spec_transform.sample_rate,
                    lowpass_filter_width=128
                ).to(self.device)
            waveform = self.resample_kernels[sr](waveform)
        
        # 保存重采样后的波形以供后续使用
        self.processed_waveform_torch = waveform.clone()
        
        mel = dynamic_range_compression_torch(
            self.mel_spec_transform(waveform), clip_val=1e-5
        ).transpose(1, 2)
        pred_curve = self.model(mel).squeeze(0).cpu().numpy()
        pred_curve = np.interp(
            np.linspace(0, len(pred_curve) - 1, length),
            np.arange(len(pred_curve)),
            pred_curve.squeeze()
        ).astype(np.float32)
        return pred_curve

def evaluate_and_visualize(model_path: pathlib.Path, wav_path: pathlib.Path, output_dir: pathlib.Path, device: str):
    """
    加载模型，评估单个WAV文件，保存结果并进行可视化。
    """
    print("Step 1: 初始化模型...")
    estimator = CurveEstimator(model_path=model_path, device=device)
    print(f"模型已加载到设备: {estimator.device}")

    print(f"\nStep 2: 读取并预处理音频文件: {wav_path}")
    sr, waveform = wavfile.read(wav_path)
    
    # 转换为单声道浮点数
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    
    print(f"原始采样率: {sr}, 音频长度: {len(waveform) / sr:.2f}s")

    # 计算模型输出的帧数
    hop_size = estimator.mel_spec_transform.hop_length
    n_frames = len(waveform) // hop_size + 1
    
    print(f"\nStep 3: 运行模型预测...")
    pred_curve = estimator.estimate(waveform, sr, length=n_frames)
    print(f"预测曲线生成完毕，长度: {len(pred_curve)}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存npy文件
    npy_path = output_dir / f"{wav_path.stem}_pred_curve.npy"
    np.save(npy_path, pred_curve)
    print(f"\nStep 4: 预测曲线已保存到: {npy_path}")

    print("\nStep 5: 生成并保存可视化图像...")
    
    # 使用 estimator 内部处理过的波形来计算用于可视化的梅尔谱
    waveform_torch = estimator.processed_waveform_torch
    mel_spec = estimator.mel_spec_transform(waveform_torch)
    mel_spec = dynamic_range_compression_torch(mel_spec)
    mel_spec = mel_spec.squeeze(0).cpu().numpy()

    # 创建绘图
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 绘制梅尔频谱图
    # 注意：imshow 的 y 轴是反的，所以 origin='lower' 很重要
    img = ax.imshow(
        mel_spec,
        aspect='auto',
        origin='lower',
        interpolation='none',
        cmap='magma'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Mel Spectrogram')

    # 设置主Y轴（频率）
    # 使用模型的参数来设置Y轴刻度
    y_ticks = np.linspace(0, mel_spec.shape[0] - 1, num=5, dtype=int)
    f_max = estimator.mel_spec_transform.f_max
    if f_max is None:
        f_max = estimator.mel_spec_transform.sample_rate / 2
    y_tick_labels = np.linspace(estimator.mel_spec_transform.f_min, f_max, num=5, dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel("Frequency [Hz]")

    # 设置X轴（时间）
    duration = len(waveform) / sr
    ax.set_xlim(0, mel_spec.shape[1] -1) # 限制X轴范围为帧数
    x_ticks = np.linspace(0, mel_spec.shape[1] - 1, num=6)
    x_tick_labels = [f"{t:.2f}" for t in np.linspace(0, duration, num=6)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Time [s]")
    
    ax.set_title(f"Predicted Mouth Openning Curve vs. Mel Spectrogram\n(File: {wav_path.name})")

    # 创建共享X轴的次Y轴
    ax2 = ax.twinx()
    
    # 绘制预测曲线
    # 确保曲线长度与频谱图的帧数匹配
    time_steps = np.arange(len(pred_curve))
    ax2.plot(time_steps, pred_curve, 'c-', linewidth=2, label='Predicted Curve')
    
    # 设置次Y轴
    ax2.set_ylabel("Predicted Value", color='c')
    ax2.tick_params(axis='y', labelcolor='c')
    ax2.legend(loc='upper right')
    
    # 调整布局并保存
    plt.tight_layout()
    png_path = output_dir / f"{wav_path.stem}_visualization.png"
    plt.savefig(png_path, dpi=300)
    plt.show()
    # plt.close(fig)
    
    print(f"可视化图像已保存到: {png_path}")
    print("\n完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a curve estimation model and visualize the output.")
    parser.add_argument("-m", "--model", type=pathlib.Path, required=True,
                        help="Path to the model checkpoint file (.pth). "
                             "A 'config.yaml' must exist in the same directory.")
    parser.add_argument("-w", "--wav", type=pathlib.Path, required=True,
                        help="Path to the input audio file (.wav).")
    parser.add_argument("-o", "--output_dir", type=pathlib.Path, default=None,
                        help="Directory to save the output .npy and .png files. "
                             "Defaults to a new folder named 'outputs' in the same directory as the wav file.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (e.g., 'cuda', 'cpu').")
    
    args = parser.parse_args()

    # 如果未指定输出目录，则在音频文件旁边创建一个
    if args.output_dir is None:
        args.output_dir = args.wav.parent / "outputs"

    try:
        evaluate_and_visualize(args.model, args.wav, args.output_dir, args.device)
    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 发生未知错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
