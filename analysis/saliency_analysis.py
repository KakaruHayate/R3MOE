# -*- coding: utf-8 -*-
"""
解释性分析：R3MOE 开口度模型到底在频谱的哪些部分做决策？

两条主线：
  A. 梯度显著性 (saliency / SmoothGrad)：网络输出对 log-mel 输入的梯度，
     直接显示「网络在每个时频点的关注强度」。
  B. 声学验证 (parselmouth)：提取 F1/F2/RMS/谱质心/F0，
     检验 saliency 是否真的集中在第一共振峰 F1 区域。

产出：analysis/outputs/ 下若干 PNG + 控制台数字结论。
"""
import sys
import pathlib

import numpy as np
import torch
import torchaudio.transforms as T
import yaml
import librosa
import parselmouth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.stats import pearsonr, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from lib.nets import BiLSTMCurveEstimator
from lib.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch

EXP = ROOT / "experiments" / "0508_s2k_noise_aug_0.15"
MODEL_PATH = EXP / "ema_model_4.pt"
WAV_PATH = pathlib.Path("*.wav")
OUT = ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def filter_kwargs(d, obj):
    import inspect
    sig = inspect.signature(obj)
    keys = [p.name for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    return {k: d[k] for k in keys if k in d}


def load_model():
    cfg = yaml.safe_load(open(EXP / "config.yaml", encoding="utf-8"))
    da, ma = cfg["dataset_args"], cfg["model_args"]
    mel_tf = PitchAdjustableMelSpectrogram(
        sample_rate=da["sample_rate"], n_fft=da["win_size"], win_length=da["win_size"],
        hop_length=da["hop_size"], f_min=da["f_min"], f_max=da["f_max"],
        n_mels=da["mel_bins"], center=True,
    )
    model = BiLSTMCurveEstimator(**filter_kwargs(ma, BiLSTMCurveEstimator))
    sd = torch.load(MODEL_PATH, map_location="cpu")
    sd = {k: v for k, v in sd.items() if "k_filter" not in k}
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    return model, mel_tf, da


def compute_logmel(waveform, sr, mel_tf):
    """复刻 eval.py 的预处理，返回 log-mel (1, T, n_mels) 与重采样后的 16k 波形。"""
    wav = torch.from_numpy(waveform).float().to(DEVICE).unsqueeze(0)
    if sr != mel_tf.sample_rate:
        wav = T.Resample(sr, mel_tf.sample_rate, lowpass_filter_width=128).to(DEVICE)(wav)
    spec = mel_tf(wav)                                  # (1, n_mels, T)
    logmel = dynamic_range_compression_torch(spec, clip_val=1e-5).transpose(1, 2)  # (1, T, n_mels)
    return logmel, wav.squeeze(0).detach().cpu().numpy()


@torch.no_grad()
def predict_curve(model, logmel):
    return model.infer(logmel).squeeze(0).cpu().numpy()


def saliency_smoothgrad(model, logmel, n_samples=16, noise_sigma_ratio=0.10):
    """SmoothGrad: 对 sum(归一化输出) 关于 log-mel 输入求梯度，多次加噪平均 |grad|。
    返回 (T, n_mels) 的显著性图。

    方案：临时切 train() 让 cuDNN 允许反向，同时关所有 dropout 保证确定性。
    """
    # 保存原模式，切 train + 关 dropout
    was_training = model.training
    model.train(True)
    orig_dropout_p = model.input_stack[3].p
    model.input_stack[3].p = 0.0

    x0 = logmel.detach()
    std = (x0.max() - x0.min()).item() * noise_sigma_ratio
    acc = torch.zeros_like(x0)
    for i in range(n_samples):
        noise = torch.randn_like(x0) * std if i > 0 else torch.zeros_like(x0)
        x = (x0 + noise).clone().requires_grad_(True)
        out = model.forward(x)            # 归一化曲线 (1, T)，含 sigmoid
        model.zero_grad(set_to_none=True)
        out.sum().backward()
        acc += x.grad.abs()

    # 恢复原模式
    model.input_stack[3].p = orig_dropout_p
    model.train(was_training)

    return (acc / n_samples).squeeze(0).cpu().numpy()   # (T, n_mels)


def occlusion_band_importance(model, logmel, base_curve_norm, n_bands=20):
    """遮挡敏感度：把每个频带置为静音地板(log clip_val)，测输出平均绝对变化。
    返回 (band_lo, band_hi, importance) 列表，更忠实但更慢。"""
    floor = float(np.log(1e-5))
    n_mels = logmel.shape[-1]
    edges = np.linspace(0, n_mels, n_bands + 1).astype(int)
    results = []
    with torch.no_grad():
        for b in range(n_bands):
            lo, hi = edges[b], edges[b + 1]
            if hi <= lo:
                continue
            x = logmel.clone()
            x[..., lo:hi] = floor
            cur = model.forward(x).squeeze(0).cpu().numpy()
            imp = float(np.mean(np.abs(cur - base_curve_norm)))
            results.append((lo, hi, imp))
    return results


def extract_acoustic(wav16, sr, n_frames, hop):
    """用 parselmouth + librosa 在 50fps 网格上提取声学特征。"""
    t_grid = (np.arange(n_frames) * hop / sr)
    snd = parselmouth.Sound(wav16.astype(np.float64), sampling_frequency=sr)

    # 共振峰 (Burg LPC)
    formant = snd.to_formant_burg(time_step=hop / sr, max_number_of_formants=5,
                                  maximum_formant=5000)
    f1 = np.array([formant.get_value_at_time(1, t) for t in t_grid])
    f2 = np.array([formant.get_value_at_time(2, t) for t in t_grid])

    # 基频
    pitch = snd.to_pitch(time_step=hop / sr)
    f0 = np.array([pitch.get_value_at_time(t) for t in t_grid])

    # RMS(dB) 与谱质心 (librosa, 同 hop)
    rms = librosa.feature.rms(y=wav16, frame_length=1024, hop_length=hop, center=True)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    cent = librosa.feature.spectral_centroid(y=wav16, sr=sr, n_fft=1024, hop_length=hop, center=True)[0]

    def fit(a):
        a = np.asarray(a, dtype=float)
        if len(a) >= n_frames:
            return a[:n_frames]
        return np.pad(a, (0, n_frames - len(a)), mode="edge")

    return {
        "t": t_grid, "F1": f1, "F2": f2, "F0": f0,
        "RMS_dB": fit(rms_db), "centroid": fit(cent),
    }


def corr_report(curve, feats):
    """浊音帧上计算曲线与各特征相关性（NaN=清音/静音）。"""
    voiced = ~np.isnan(feats["F0"]) & ~np.isnan(feats["F1"])
    print(f"\n=== 相关性 (仅浊音帧 N={voiced.sum()} / {len(curve)}) ===")
    print(f"{'特征':<12}{'Pearson r':>12}{'Spearman ρ':>14}")
    rows = {}
    for name in ["F1", "F2", "F0", "RMS_dB", "centroid"]:
        v = feats[name]
        m = voiced & ~np.isnan(v)
        if m.sum() < 10:
            continue
        pr = pearsonr(curve[m], v[m])[0]
        sr = spearmanr(curve[m], v[m])[0]
        rows[name] = pr
        print(f"{name:<12}{pr:>12.3f}{sr:>14.3f}")
    return rows, voiced


def pick_window(curve, sr, hop, win_sec=10.0):
    """选预测曲线方差最大的窗口（最具动态的片段）。"""
    fps = sr / hop
    w = int(win_sec * fps)
    if w >= len(curve):
        return 0, len(curve)
    best, bi = -1, 0
    step = max(1, w // 4)
    for i in range(0, len(curve) - w, step):
        v = np.var(curve[i:i + w])
        if v > best:
            best, bi = v, i
    return bi, bi + w


def main():
    print(f"设备: {DEVICE}\n模型: {MODEL_PATH.name}")
    model, mel_tf, da = load_model()
    sr_native, wav = wavfile.read(WAV_PATH)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    print(f"音频: {WAV_PATH.name}  {len(wav)/sr_native:.1f}s @ {sr_native}Hz")

    logmel, wav16 = compute_logmel(wav, sr_native, mel_tf)   # (1,T,80)
    sr, hop = da["sample_rate"], da["hop_size"]
    n_frames = logmel.shape[1]
    print(f"log-mel: {tuple(logmel.shape)}  帧率 {sr/hop:.0f}fps")

    curve = predict_curve(model, logmel)                     # 反归一化 [-0.15,1]
    curve_norm = (curve - model.vmin) / (model.vmax - model.vmin)

    print("计算 SmoothGrad 显著性...")
    sal = saliency_smoothgrad(model, logmel)                 # (T,80)
    print("计算遮挡敏感度...")
    occ = occlusion_band_importance(model, logmel, curve_norm, n_bands=20)

    print("提取声学特征 (parselmouth)...")
    feats = extract_acoustic(wav16, sr, n_frames, hop)

    rows, voiced = corr_report(curve, feats)

    # mel bin 中心频率
    mel_f = librosa.mel_frequencies(n_mels=da["mel_bins"], fmin=da["f_min"],
                                    fmax=da["f_max"] or sr / 2)

    # 全局频率重要性
    sal_freq = sal.mean(0)                                   # 每个 mel bin 的平均显著性
    sal_freq_v = sal[voiced].mean(0) if voiced.sum() > 0 else sal_freq
    peak_bin = int(np.argmax(sal_freq_v))
    print(f"\n=== 全局频率注意力 ===")
    print(f"梯度显著性峰值频带: mel#{peak_bin} ≈ {mel_f[peak_bin]:.0f} Hz")
    # 重心
    centroid_hz = float(np.sum(sal_freq_v * mel_f) / np.sum(sal_freq_v))
    print(f"显著性频率重心: {centroid_hz:.0f} Hz")
    occ_peak = max(occ, key=lambda r: r[2])
    print(f"遮挡敏感峰值频带: {mel_f[occ_peak[0]]:.0f}-{mel_f[min(occ_peak[1],79)]:.0f} Hz "
          f"(Δ={occ_peak[2]:.4f})")
    f1_med = np.nanmedian(feats["F1"][voiced]) if voiced.sum() else np.nan
    print(f"实测 F1 中位数 (浊音): {f1_med:.0f} Hz")

    # ---------- 可视化 ----------
    b0, b1 = pick_window(curve, sr, hop, win_sec=10.0)
    t0 = b0 * hop / sr
    extent = [t0, b1 * hop / sr, 0, len(mel_f)]
    mel_win = logmel[0, b0:b1].cpu().numpy().T     # (80, W)
    sal_win = sal[b0:b1].T                          # (80, W)
    tt = feats["t"][b0:b1]

    def freq_yaxis(ax):
        idx = np.linspace(0, len(mel_f) - 1, 6).astype(int)
        ax.set_yticks(idx); ax.set_yticklabels([f"{mel_f[i]:.0f}" for i in idx])
        ax.set_ylabel("Freq [Hz]")

    # Fig1: mel + 曲线 + F1/F2
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(mel_win, aspect="auto", origin="lower", cmap="magma", extent=extent)
    f1b = np.interp(feats["F1"][b0:b1], (0, mel_f[-1]), (0, len(mel_f)))
    f2b = np.interp(feats["F2"][b0:b1], (0, mel_f[-1]), (0, len(mel_f)))
    ax.plot(tt, f1b, ".", ms=3, color="cyan", label="F1 (Praat)")
    ax.plot(tt, f2b, ".", ms=2, color="lime", alpha=0.6, label="F2")
    freq_yaxis(ax); ax.set_xlabel("Time [s]")
    ax2 = ax.twinx(); ax2.plot(tt, curve[b0:b1], "w-", lw=2, label="Predicted Opening"); ax2.set_ylim(-0.2, 1.1)
    ax2.set_ylabel("Mouth Opening", color="w")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.set_title("Mel-Spectrogram + Predicted Opening + Formants F1/F2")
    plt.tight_layout(); plt.savefig(OUT / "fig1_mel_curve_formants.png", dpi=150); plt.close()

    # Fig2: saliency 热图 + F1
    fig, ax = plt.subplots(figsize=(15, 6))
    vmax = np.percentile(sal_win, 99)
    im = ax.imshow(sal_win, aspect="auto", origin="lower", cmap="viridis",
                   extent=extent, vmax=vmax)
    ax.plot(tt, f1b, ".", ms=3, color="red", label="F1 (Praat)")
    fig.colorbar(im, ax=ax, label="|d(output)/d(log-mel)|")
    freq_yaxis(ax); ax.set_xlabel("Time [s]"); ax.legend(loc="upper left")
    ax.set_title("Network Attention (SmoothGrad Saliency) vs F1 — red dots should fall on bright bands")
    plt.tight_layout(); plt.savefig(OUT / "fig2_saliency_vs_f1.png", dpi=150); plt.close()

    # Fig3: 全局频率重要性
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mel_f, sal_freq_v / sal_freq_v.max(), "-o", ms=3, label="Grad saliency (voiced mean)")
    occ_x = [(mel_f[lo] + mel_f[min(hi, 79)]) / 2 for lo, hi, _ in occ]
    occ_y = np.array([imp for _, _, imp in occ]); occ_y = occ_y / occ_y.max()
    ax.plot(occ_x, occ_y, "-s", ms=4, color="orange", label="Occlusion sensitivity")
    ax.axvspan(300, 1000, color="red", alpha=0.12, label="Typical F1 range")
    if not np.isnan(f1_med):
        ax.axvline(f1_med, color="red", ls="--", label=f"Median F1 {f1_med:.0f}Hz")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Normalized Importance"); ax.set_xlim(0, 5000)
    ax.legend(); ax.set_title("Global Frequency Importance Distribution")
    plt.tight_layout(); plt.savefig(OUT / "fig3_freq_importance.png", dpi=150); plt.close()

    # Fig4: 散点
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axs, ["F1", "RMS_dB"]):
        v = feats[name]; m = voiced & ~np.isnan(v)
        ax.scatter(v[m], curve[m], s=3, alpha=0.2)
        r = rows.get(name, float("nan"))
        ax.set_xlabel(name); ax.set_ylabel("Predicted Opening")
        ax.set_title(f"Mouth Opening vs {name}  (r={r:.3f})")
    plt.tight_layout(); plt.savefig(OUT / "fig4_scatter.png", dpi=150); plt.close()

    print(f"\n图已保存到 {OUT}")
    for p in ["fig1_mel_curve_formants", "fig2_saliency_vs_f1",
              "fig3_freq_importance", "fig4_scatter"]:
        print(f"  {p}.png")


if __name__ == "__main__":
    main()
