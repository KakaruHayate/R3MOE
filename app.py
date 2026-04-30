import gradio as gr
import pathlib
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torchaudio
import os
import sys

# 引用你现有的模块
from lib.nets import BiLSTMCurveEstimator
from lib.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch

# ==========================================
# 配置区域 (请在此处修改你的模型路径)
# ==========================================

MODEL_REGISTRY = {
    # "显示在下拉框的名称": "实际的.pth文件路径"
    "1212s2k": "./experiments/1212s2k/ema_model_8.pt",
    "0430_s2k": "./experiments/0430_s2k/ema_model_2.pt"
}

# 默认选中的模型名称 (必须在上面的 keys 中)
DEFAULT_MODEL = list(MODEL_REGISTRY.keys())[0]

# ==========================================
# 辅助函数与类定义
# ==========================================

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
    def __init__(self):
        # 强制使用 CPU
        self.device = "cpu"
        self.model = None
        self.mel_spec_transform = None
        self.current_model_path = None 

    def load_model(self, model_path_str: str):
        path_obj = pathlib.Path(model_path_str)
        
        if self.current_model_path == str(path_obj) and self.model is not None:
            return

        if not path_obj.exists():
            raise FileNotFoundError(f"模型文件不存在: {path_obj}")
            
        config_path = path_obj.with_name("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        print(f"正在加载模型 (CPU模式): {path_obj} ...")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            dataset_args, model_args = config["dataset_args"], config["model_args"]
        
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
        # 强制 map_location="cpu"
        state_dict = torch.load(path_obj, map_location="cpu")
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'k_filter' not in k}
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.resample_kernels = {}
        
        self.last_mel_spec = None
        self.current_model_path = str(path_obj)
        print("模型加载完成。")

    @torch.no_grad()
    def estimate(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型尚未加载！")

        waveform_t = torch.from_numpy(waveform).float().to(self.device).unsqueeze(0)
        
        if sr != self.mel_spec_transform.sample_rate:
            if sr not in self.resample_kernels:
                self.resample_kernels[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.mel_spec_transform.sample_rate,
                    lowpass_filter_width=128
                ).to(self.device)
            waveform_t = self.resample_kernels[sr](waveform_t)
        
        mel_spec = self.mel_spec_transform(waveform_t)
        self.last_mel_spec = mel_spec 
        
        mel = dynamic_range_compression_torch(mel_spec, clip_val=1e-5).transpose(1, 2)
        pred_curve = self.model(mel).squeeze(0).cpu().numpy().squeeze()
        
        return pred_curve

# 全局实例
estimator_instance = None

# ==========================================
# 核心逻辑
# ==========================================

def run_inference(model_key, audio_path):
    global estimator_instance
    
    if model_key not in MODEL_REGISTRY:
        raise gr.Error(f"未知的模型选项: {model_key}")
    
    target_path = MODEL_REGISTRY[model_key]
    
    try:
        if estimator_instance is None:
            estimator_instance = CurveEstimator()
        
        # 加载指定权重的模型
        estimator_instance.load_model(target_path)
            
    except Exception as e:
        raise gr.Error(f"模型加载失败: {str(e)}")

    if audio_path is None:
        raise gr.Error("请上传音频文件！")
        
    print(f"正在处理音频: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=None, mono=True)
    
    pred_curve = estimator_instance.estimate(waveform, sr)
    
    fig = create_plot(estimator_instance, waveform, sr, pred_curve, os.path.basename(audio_path))
    
    return fig

def create_plot(estimator, waveform, sr, pred_curve, filename):
    mel_spec = dynamic_range_compression_torch(estimator.last_mel_spec)
    mel_spec = mel_spec.squeeze(0).cpu().numpy()
    
    # 修改点：增大画布尺寸 (宽, 高)
    fig, ax = plt.subplots(figsize=(20, 8))
    
    img = ax.imshow(
        mel_spec,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        cmap='magma'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Mel Spectrogram')
    
    y_ticks = np.linspace(0, mel_spec.shape[0] - 1, num=5, dtype=int)
    f_max = estimator.mel_spec_transform.f_max or estimator.mel_spec_transform.sample_rate / 2
    y_tick_labels = np.linspace(estimator.mel_spec_transform.f_min, f_max, num=5, dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel("Frequency [Hz]")
    
    n_frames = mel_spec.shape[1]
    duration = len(waveform) / sr
    
    ax.set_xlim(0, n_frames - 1)
    x_ticks = np.linspace(0, n_frames - 1, num=6)
    x_tick_labels = [f"{t:.2f}" for t in np.linspace(0, duration, num=6)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel("Time [s]")
    ax.set_title(f"Analysis Result: {filename}")
    
    ax2 = ax.twinx()
    time_steps = np.arange(len(pred_curve))
    
    if len(pred_curve) != n_frames:
         pred_curve = np.interp(
            np.linspace(0, len(pred_curve) - 1, n_frames),
            np.arange(len(pred_curve)),
            pred_curve
        )
         time_steps = np.arange(n_frames)

    ax2.plot(time_steps, pred_curve, 'c-', linewidth=2, label='Predicted Curve', alpha=0.9)
    ax2.set_ylabel("Parameter Value", color='c')
    ax2.tick_params(axis='y', labelcolor='c')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# ==========================================
# Gradio 界面
# ==========================================

# 修改点：增大容器最大宽度，使大图能撑开
css = """
#col-container {max-width: 1200px; margin-left: auto; margin-right: auto;}
"""

with gr.Blocks(title="Lip Sync Curve WebUI", css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        # 修改点：文案变更
        gr.Markdown("# 👄 口型参数曲线提取 (Curve Estimator)")
        gr.Markdown("上传音频文件，系统将自动生成对应的口型参数曲线。")
        
        with gr.Row():
            # 修改点：删除了 device 选项，只保留模型选择
            model_selector = gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()),
                value=DEFAULT_MODEL,
                label="选择模型权重",
                interactive=True
            )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="上传音频 (mp3, wav, flac...)", 
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                submit_btn = gr.Button("开始分析", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                plot_output = gr.Plot(label="可视化结果")

    submit_btn.click(
        fn=run_inference,
        inputs=[model_selector, audio_input],
        outputs=[plot_output]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0", 
        inbrowser=True,
        # share=True # 如需公网分享请取消注释
    )
