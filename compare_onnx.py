import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from scipy.io import wavfile

# 从你的 eval.py 中导入原生推理类
try:
    from eval import CurveEstimator
except ImportError:
    print("错误: 无法导入 eval.py，请确保 compare_onnx.py 与 eval.py 在同一级目录下。")
    sys.exit(1)

def compare_and_visualize(pt_path: pathlib.Path, onnx_path: pathlib.Path, wav_path: pathlib.Path, output_dir: pathlib.Path):
    print(">>> 正在初始化对比环境...")
    
    # 1. 初始化 PyTorch 模型
    print(f"加载 PyTorch 模型: {pt_path}")
    pt_estimator = CurveEstimator(model_path=pt_path, device="cpu")
    
    # 2. 初始化 ONNX 模型
    print(f"加载 ONNX 模型: {onnx_path}")
    providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
    
    # 3. 读取并处理音频 (模拟 eval.py 的原生处理)
    print(f"读取音频文件: {wav_path}")
    sr, pt_waveform = wavfile.read(wav_path)
    if pt_waveform.ndim > 1:
        pt_waveform = pt_waveform.mean(axis=1)
    if pt_waveform.dtype != np.float32:
        pt_waveform = pt_waveform.astype(np.float32) / np.iinfo(pt_waveform.dtype).max
        
    # 4. PyTorch 原生推理
    print("运行 PyTorch 推理...")
    pt_curve = pt_estimator.estimate(pt_waveform, sr)
    
    # 5. 准备 ONNX 输入 (现要求输入 44.1kHz 单声道音频)
    print("准备 ONNX 输入并运行推理...")
    waveform_tensor = torch.from_numpy(pt_waveform).float().unsqueeze(0)
    
    # 根据你导出的设定，将 ONNX 输入重采样对齐为 44100Hz
    if sr != 44100:
        print(f"检测到输入采样率为 {sr}Hz，正在重采样至 44100Hz 送入 ONNX 模型...")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
        onnx_waveform = resampler(waveform_tensor)
    else:
        onnx_waveform = waveform_tensor
        
    onnx_inputs = {"waveform": onnx_waveform.numpy()}
    onnx_curve = ort_session.run(None, onnx_inputs)[0].squeeze()
    
    # 6. 对齐与误差计算
    # 由于 padding 逻辑和重采样的舍入，长度可能会差 1-2 帧
    min_len = min(len(pt_curve), len(onnx_curve))
    pt_curve_aligned = pt_curve[:min_len]
    onnx_curve_aligned = onnx_curve[:min_len]
    
    abs_diff = np.abs(pt_curve_aligned - onnx_curve_aligned)
    mse = np.mean(abs_diff ** 2)
    max_error = np.max(abs_diff)
    mean_error = np.mean(abs_diff)
    
    print(f"\n[误差统计]")
    print(f"PyTorch 曲线长度: {len(pt_curve)}")
    print(f"ONNX    曲线长度: {len(onnx_curve)}")
    print(f"对齐后验证长度:   {min_len}")
    print(f"平均绝对误差 (MAE): {mean_error:.6f}")
    print(f"最大绝对误差 (Max): {max_error:.6f}")
    print(f"均方误差 (MSE):     {mse:.6e}")

    # 7. 可视化
    print("\n绘制对比图...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    time_steps = np.arange(min_len)
    
    # 曲线重叠对比图
    ax1.plot(time_steps, pt_curve_aligned, label="PyTorch (Original)", color='blue', alpha=0.7, linewidth=2)
    ax1.plot(time_steps, onnx_curve_aligned, label="ONNX (Exported)", color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.set_title(f"Curve Comparison: PyTorch vs ONNX\n(File: {wav_path.name})", fontsize=14)
    ax1.set_ylabel("Predicted Value")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right")
    
    # 添加误差信息文本框
    info_text = f"MSE: {mse:.2e}\nMax Error: {max_error:.2e}\nMAE: {mean_error:.2e}"
    ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 误差分布图
    ax2.plot(time_steps, abs_diff, color='purple', label="Absolute Difference", linewidth=1.5)
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Absolute Error")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    save_path = output_dir / f"{wav_path.stem}_pt_vs_onnx.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    print(f"对比图已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX model outputs.")
    parser.add_argument("-p", "--pt", type=pathlib.Path, required=True,
                        help="Path to the original PyTorch model (.pt).")
    parser.add_argument("-x", "--onnx", type=pathlib.Path, required=True,
                        help="Path to the exported ONNX model (.onnx).")
    parser.add_argument("-w", "--wav", type=pathlib.Path, required=True,
                        help="Path to the input audio file (.wav).")
    parser.add_argument("-o", "--output_dir", type=pathlib.Path, default=pathlib.Path("./comparisons"),
                        help="Directory to save the comparison images.")
    
    args = parser.parse_args()

    try:
        compare_and_visualize(args.pt, args.onnx, args.wav, args.output_dir)
    except Exception as e:
        print(f"\n[错误]: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)