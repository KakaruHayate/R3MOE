import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import pathlib
import yaml
import time
import librosa
import soundfile as sf
import platform
import subprocess
import sys

# --- Cross-Platform Audio Playback Support ---
OS_NAME = platform.system() 

try:
    if OS_NAME == 'Windows':
        import winsound
        HAS_WINSOUND = True
    else:
        HAS_WINSOUND = False
except ImportError:
    HAS_WINSOUND = False

# --- Drag and Drop Support ---
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

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
    return {k: dict_to_filter[k] for k in filter_keys if k in dict_to_filter}

class SinusoidalSmoothingConv1d(torch.nn.Conv1d):
    def __init__(self, kernel_size):
        super().__init__(in_channels=1, out_channels=1, kernel_size=kernel_size, 
                         bias=False, padding='same', padding_mode='replicate')
        smooth_kernel = torch.sin(torch.linspace(0, 1, kernel_size) * np.pi)
        smooth_kernel /= smooth_kernel.sum()
        self.weight.data = smooth_kernel[None, None]
        self.weight.requires_grad = False

class MouthBakerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouth Opening Baker | Acoustic Hybrid ARKit")
        self.root.geometry("1200x950") # 增加高度容纳新推子
        
        self.bg_color = "#252526"      
        self.panel_bg = "#1e1e1e"      
        self.fg_color = "#cccccc"      
        self.accent_color = "#E91E63"  
        self.centroid_color = "#00BCD4" 
        self.btn_bg = "#3a3d41"        
        self.border_color = "#3e3e42"  
        self.font_main = ("Segoe UI", 9)
        self.font_bold = ("Segoe UI", 9, "bold")
        
        self.root.configure(bg=self.bg_color)
        self.apply_modern_style()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.raw_data = None
        self.raw_centroid = None 
        
        self.wav_path = None
        self.playback_wav_path = None 
        self.audio_duration = 0
        self.hparams = None
        self.timestep = 0.01
        
        self.is_playing = False
        self.start_time = 0
        self.anim_jaw = None
        self.anim_cent = None
        self.audio_process = None 

        self.setup_ui()
        self.auto_load_default()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        if HAS_DND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop_file)

    def apply_modern_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=self.font_main)
        
        style.configure("TButton", background=self.btn_bg, foreground=self.fg_color, 
                        borderwidth=0, focuscolor=self.bg_color, font=self.font_main, padding=5)
        style.map("TButton", 
                  background=[('active', '#505357'), ('pressed', self.accent_color)],
                  foreground=[('active', '#ffffff')])
                  
        style.configure("Accent.TButton", background=self.accent_color, foreground="#ffffff", font=self.font_bold)
        style.map("Accent.TButton", background=[('active', '#d81b60')])
        
        style.configure("TEntry", fieldbackground=self.panel_bg, foreground="#ffffff", 
                        insertcolor="#ffffff", borderwidth=1, bordercolor=self.border_color)
        
        style.configure("Horizontal.TScale", background=self.bg_color, troughcolor=self.panel_bg, 
                        slidercolor=self.accent_color, borderwidth=0)
        
        style.configure("TLabelframe", background=self.bg_color, bordercolor=self.border_color, borderwidth=1)
        style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.accent_color, font=self.font_bold)
        
        style.configure("TSeparator", background=self.border_color)
        style.configure("TCombobox", fieldbackground=self.panel_bg, background=self.btn_bg, foreground="#ffffff")

    def on_closing(self):
        self.stop_playback(force_kill_audio=True)
        plt.close(self.fig) 
        self.root.destroy()
        sys.exit(0) 

    def auto_load_default(self):
        if getattr(sys, 'frozen', False):
            application_path = pathlib.Path(sys.executable).parent
        else:
            application_path = pathlib.Path(__file__).parent
            
        base_path = application_path / "experiments" / "1212s2k"
        model_path = base_path / "ema_model_8.pt"
        config_path = base_path / "config.yaml"
        if model_path.exists() and config_path.exists():
            try:
                self.load_model_and_config(model_path, config_path)
            except Exception as e:
                print(f"Auto-load failed: {e}")

    def load_model_and_config(self, model_path, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.hparams = yaml.safe_load(f)
            
        dataset_args = self.hparams["dataset_args"]
        model_args = self.hparams["model_args"]
        
        self.model = BiLSTMCurveEstimator(**filter_kwargs(model_args, BiLSTMCurveEstimator))
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        self.model.to(self.device)
        
        self.mel_extractor = PitchAdjustableMelSpectrogram(
            sample_rate=dataset_args["sample_rate"], n_fft=dataset_args["win_size"],
            win_length=dataset_args["win_size"], hop_length=dataset_args["hop_size"],
            f_min=dataset_args["f_min"], f_max=dataset_args["f_max"],
            n_mels=dataset_args["mel_bins"], center=True
        )
        self.timestep = dataset_args["hop_size"] / dataset_args["sample_rate"]
        self.status_label.config(text=f"Ready / 就绪: {model_path.name}", foreground="#4caf50") 

    def setup_ui(self):
        ctrl_frame = ttk.Frame(self.root, padding=15)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)

        dnd_text = "📁 Load Audio / 加载音频 (Drag & Drop)" if HAS_DND else "📁 Load Audio / 加载音频"
        ttk.Button(ctrl_frame, text=dnd_text, command=self.load_audio_ui).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(ctrl_frame, text="⚙ Reload Model/Config / 重选模型与配置", command=self.manual_load_ui).pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(ctrl_frame, text="Standby / 待命", foreground="#888888")
        self.status_label.pack(fill=tk.X, pady=5)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=10)

        # ====== 独立平滑器区域 ======
        ttk.Label(ctrl_frame, text="Mouth Opening Smooth / 张合平滑 (s):").pack(anchor=tk.W)
        self.smooth_val = tk.DoubleVar(value=0.12)
        self.smooth_label = ttk.Label(ctrl_frame, text="0.12", foreground="#ffffff")
        self.smooth_label.pack(anchor=tk.E, pady=(0, 2))
        ttk.Scale(ctrl_frame, from_=0, to=0.5, variable=self.smooth_val, command=self.on_param_change).pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Centroid Smooth / 质心平滑 (推荐>0.3):").pack(anchor=tk.W, pady=(5,0))
        self.cent_smooth_val = tk.DoubleVar(value=0.36)
        self.cent_smooth_label = ttk.Label(ctrl_frame, text="0.36", foreground="#ffffff")
        self.cent_smooth_label.pack(anchor=tk.E, pady=(0, 2))
        ttk.Scale(ctrl_frame, from_=0.1, to=1.0, variable=self.cent_smooth_val, command=self.on_param_change).pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Extraction Scale / 提取倍率:").pack(anchor=tk.W, pady=(10,0))
        self.scale_val = tk.DoubleVar(value=1.0)
        self.scale_label = ttk.Label(ctrl_frame, text="1.00", foreground="#ffffff")
        self.scale_label.pack(anchor=tk.E, pady=(0, 2))
        ttk.Scale(ctrl_frame, from_=0.1, to=3.0, variable=self.scale_val, command=self.on_param_change).pack(fill=tk.X)

        ttk.Label(ctrl_frame, text="Output FPS / 输出帧率:").pack(anchor=tk.W, pady=(10,5))
        self.fps_val = tk.IntVar(value=30)
        ttk.Entry(ctrl_frame, textvariable=self.fps_val).pack(fill=tk.X)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=10)
        
        shape_frame = ttk.LabelFrame(ctrl_frame, text=" VIEWPORT SETTINGS ")
        shape_frame.pack(fill=tk.X, pady=5, ipadx=10, ipady=10)
        
        ttk.Label(shape_frame, text="Anim Mode / 动画模式:").pack(anchor=tk.W)
        self.anim_mode_val = tk.StringVar(value="Acoustic Hybrid ARKit (DSP+DL)")
        mode_cb = ttk.Combobox(shape_frame, textvariable=self.anim_mode_val, state="readonly", 
                               values=["Acoustic Hybrid ARKit (DSP+DL)", "Squash & Stretch (Physics)", 
                                       "Sprite Sheet (4 Frames)", "Standard (Linear)"])
        mode_cb.pack(fill=tk.X, pady=(0, 10))
        mode_cb.bind("<<ComboboxSelected>>", self.on_shape_change)
        
        ttk.Label(shape_frame, text="Mouth Width / 嘴巴宽度:").pack(anchor=tk.W)
        self.mouth_width_val = tk.DoubleVar(value=50)
        ttk.Scale(shape_frame, from_=10, to=150, variable=self.mouth_width_val, command=self.on_shape_change).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(shape_frame, text="Corner Sharpness / 唇角锐度:").pack(anchor=tk.W)
        self.mouth_curve_val = tk.DoubleVar(value=2.0)
        ttk.Scale(shape_frame, from_=1.0, to=3.5, variable=self.mouth_curve_val, command=self.on_shape_change).pack(fill=tk.X)

        ttk.Button(ctrl_frame, text="↺ Reset All Settings / 重置设置", command=self.reset_settings).pack(fill=tk.X, pady=(10, 0))

        self.btn_play = ttk.Button(ctrl_frame, text="▶ PLAY / 播放动画", command=self.toggle_playback, style="Accent.TButton")
        self.btn_play.pack(fill=tk.X, pady=10)
        
        ttk.Label(ctrl_frame, text="EXPORT / 导出:").pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(ctrl_frame, text="⭳ CSV (1D MouthOpen Only)", command=lambda: self.export("csv")).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl_frame, text="⭳ CSV (ARKit 6 BlendShapes)", command=lambda: self.export("arkit_csv")).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl_frame, text="⭳ JSON (AE Expression)", command=lambda: self.export("json")).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl_frame, text="⭳ NPY (Raw Data)", command=lambda: self.export("npy")).pack(fill=tk.X, pady=2)

        self.preview_frame = ttk.Frame(self.root, padding=15)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 修正了 Curve Editor 的文本描述
        plot_frame = ttk.LabelFrame(self.preview_frame, text=" CURVE EDITOR (Pink: Mouth Opening, Cyan: Spectral Centroid) ")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 3), facecolor=self.panel_bg) 
        self.ax.set_facecolor(self.panel_bg) 
        self.ax.tick_params(colors=self.fg_color) 
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.border_color) 
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        anim_container = ttk.LabelFrame(self.preview_frame, text=" VIEWPORT (REAL-TIME) ")
        anim_container.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        self.anim_canvas = tk.Canvas(anim_container, height=220, bg=self.panel_bg, highlightthickness=0)
        self.anim_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.mouth_polygon = self.anim_canvas.create_polygon(0,0,0,0, fill=self.accent_color, outline="#ffffff", width=2, smooth=False)
        self.anim_canvas.bind("<Configure>", lambda e: self.draw_mouth(0.0, 0.0, 0.0))

    def reset_settings(self):
        self.smooth_val.set(0.12)
        self.cent_smooth_val.set(0.36)
        self.scale_val.set(1.0)
        self.fps_val.set(30)
        self.mouth_width_val.set(50)
        self.mouth_curve_val.set(2.0)
        self.anim_mode_val.set("Acoustic Hybrid ARKit (DSP+DL)")
        self.on_param_change()
        self.on_shape_change()

    def on_drop_file(self, event):
        file_path = event.data.strip('{}')
        self.process_audio_file(file_path)

    def load_audio_ui(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded.")
            return
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")])
        if path:
            self.process_audio_file(path)

    def process_audio_file(self, path):
        if self.model is None: return
        self.stop_playback(force_kill_audio=True)
        self.raw_data = None
        self.raw_centroid = None
        self.anim_jaw = None
        self.anim_cent = None
        
        self.wav_path = pathlib.Path(path)
        self.status_label.config(text=f"Processing / 处理中: {self.wav_path.name}...", foreground="#ffb900") 
        self.root.update()

        try:
            waveform, sr = librosa.load(self.wav_path, sr=None, mono=True)
            self.audio_duration = len(waveform) / sr 
            
            if self.wav_path.suffix.lower() != '.wav':
                temp_wav = pathlib.Path.cwd() / "temp_preview.wav"
                sf.write(temp_wav, waveform, sr)
                self.playback_wav_path = temp_wav
            else:
                self.playback_wav_path = self.wav_path

            target_sr = self.hparams["dataset_args"]["sample_rate"]
            if sr != target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            
            audio_t = torch.from_numpy(waveform).float()
            mel = self.mel_extractor(audio_t[None])
            mel = dynamic_range_compression_torch(mel, clip_val=1e-5).transpose(1, 2).to(self.device)
            
            with torch.no_grad():
                dl_out = self.model(mel).squeeze(0).squeeze(-1).cpu() 
            
            win_size = self.hparams["dataset_args"]["win_size"]
            hop_size = self.hparams["dataset_args"]["hop_size"]
            cent = librosa.feature.spectral_centroid(y=waveform, sr=target_sr, n_fft=win_size, hop_length=hop_size)[0]
            
            cent_norm = np.clip((cent - 500) / 3500.0, 0.0, 1.0)
            
            min_len = min(dl_out.shape[0], cent_norm.shape[0])
            self.raw_data = dl_out[:min_len]
            self.raw_centroid = torch.from_numpy(cent_norm[:min_len]).float()
                
            self.update_plot()
            self.draw_mouth(0.0, 0.0, 0.0) 
            self.status_label.config(text=f"Ready / 就绪: {self.wav_path.name}", foreground="#4caf50")
            
        except Exception as e:
            messagebox.showerror("Audio Load Error", str(e))
            self.status_label.config(text="Load Error", foreground="#f44336")

    def on_param_change(self, _=None):
        self.smooth_label.config(text=f"{self.smooth_val.get():.2f}")
        self.cent_smooth_label.config(text=f"{self.cent_smooth_val.get():.2f}") # 更新独立标签
        self.scale_label.config(text=f"{self.scale_val.get():.2f}")
        self.update_plot()
        if not self.is_playing: self.draw_mouth(0.0, 0.0, 0.0)

    def on_shape_change(self, _=None):
        if not self.is_playing:
            self.draw_mouth(0.0, 0.0, 0.0)

    def stop_playback(self, force_kill_audio=True):
        """
        force_kill_audio: 如果是自然播完，设为False，让系统自己处理最后几毫秒的音频尾巴；
        如果是用户点击停止、关闭窗口、或加载新音频，则设为True，强杀进程。
        """
        self.is_playing = False
        self.btn_play.config(text="▶ PLAY / 播放动画", style="Accent.TButton")
        self.draw_mouth(0.0, 0.0, 0.0)
        
        if force_kill_audio:
            if OS_NAME == 'Windows' and HAS_WINSOUND:
                winsound.PlaySound(None, winsound.SND_PURGE)
            elif OS_NAME == 'Darwin' and self.audio_process is not None:
                self.audio_process.terminate()
                self.audio_process = None

    def toggle_playback(self):
        if OS_NAME not in ['Windows', 'Darwin']:
            messagebox.showwarning("Notice", "Playback is only supported on Windows/Mac.")
            return
            
        if self.playback_wav_path is None or self.raw_data is None: return

        if self.is_playing:
            # 用户主动点击，强制打断音频
            self.stop_playback(force_kill_audio=True)
        else:
            self.anim_jaw, self.anim_cent = self.process_data()
            self.is_playing = True
            self.btn_play.config(text="■ STOP / 停止预览", style="TButton") 
            
            # 如果是 Mac，确保清理之前的幽灵进程
            if OS_NAME == 'Darwin' and self.audio_process is not None:
                self.audio_process.terminate()
                
            if OS_NAME == 'Windows' and HAS_WINSOUND:
                winsound.PlaySound(str(self.playback_wav_path), winsound.SND_ASYNC | winsound.SND_FILENAME)
            elif OS_NAME == 'Darwin':
                self.audio_process = subprocess.Popen(['afplay', str(self.playback_wav_path)])
            
            self.start_time = time.time()
            self.update_animation()

    def update_animation(self):
        if not self.is_playing: return
        
        elapsed = time.time() - self.start_time
        fps = self.fps_val.get()
        
        if elapsed < (self.audio_duration + 0.1): 
            
            current_frame = int(elapsed * fps)
            
            current_frame = min(current_frame, len(self.anim_jaw) - 1)
            
            jaw_val = self.anim_jaw[current_frame]
            cent_val = self.anim_cent[current_frame]
            
            vel = 0.0
            if current_frame > 0:
                vel = (jaw_val - self.anim_jaw[current_frame - 1]) * fps
                
            self.draw_mouth(jaw_val, vel, cent_val)
            self.root.after(10, self.update_animation)
        else:
            self.stop_playback(force_kill_audio=False)

    def draw_mouth(self, jaw_val, velocity=0.0, centroid_val=0.0):
        w = self.anim_canvas.winfo_width()
        h = self.anim_canvas.winfo_height()
        if w < 10 or h < 10: return
        cx, cy = w / 2, h / 2
        
        mode = self.anim_mode_val.get()
        a = self.mouth_width_val.get()   
        b = 2 + jaw_val * 60                 
        n = self.mouth_curve_val.get()   
        
        points = []
        
        # ==========================================
        # 魔法 1：声学混合的终极 ARKit 映射模式
        # ==========================================
        if mode == "Acoustic Hybrid ARKit (DSP+DL)":
            jawOpen = jaw_val
            brightness = centroid_val 
            
            mouthFunnel = max(0.0, jawOpen * (1.0 - brightness) * 1.5)
            mouthStretch = jawOpen * brightness * 1.0
            mouthPucker = max(0.0, (0.2 - jawOpen) * 3.0 * (1.0 - brightness)) if 0.05 < jawOpen < 0.2 else 0.0
            mouthShrugUpper = max(0.0, min(velocity * 0.1, 1.0)) * 0.5
            
            # 【修复 Zooming 错觉】：大幅下调横向乘区系数 (从0.5降到0.25)，限制整体形变幅度
            width_scale = 1.0 + mouthStretch * 0.25 - mouthFunnel * 0.3 - mouthPucker * 0.35
            current_width = max(10.0, a * width_scale)
            
            upper_y = cy - 2 - mouthShrugUpper * 40
            lower_y = cy + 2 + jawOpen * 70
            
            xs_upper = np.linspace(-current_width, current_width, 25)
            for x in xs_upper:
                y = upper_y + (cy - upper_y) * (abs(x / current_width)**n)
                points.extend([cx + x, y])
                
            xs_lower = np.linspace(current_width, -current_width, 25)
            for x in xs_lower:
                y = lower_y + (cy - lower_y) * (abs(x / current_width)**n)
                points.extend([cx + x, y])
                
        # ==========================================
        # 旧版模式
        # ==========================================
        else:
            if mode == "Sprite Sheet (4 Frames)":
                jaw_val = round(jaw_val * 3) / 3.0
                b = 2 + jaw_val * 60 
            elif mode == "Squash & Stretch (Physics)":
                # 【修复 Zooming】：降低缩放惩罚
                a = a * (1.0 - jaw_val * 0.15)
                
            for t in np.linspace(0, 2 * np.pi, 40):
                cos_t = np.cos(t)
                sin_t = np.sin(t)
                x = cx + a * (abs(cos_t) ** (2/n)) * np.sign(cos_t)
                y = cy + b * (abs(sin_t) ** (2/n)) * np.sign(sin_t)
                points.extend([x, y])
            
        self.anim_canvas.coords(self.mouth_polygon, *points)

    def manual_load_ui(self):
        m_path = filedialog.askopenfilename(title="Select Model / 选择模型", filetypes=[("Model", "*.pt *.pth")])
        if not m_path: return 
        c_path = filedialog.askopenfilename(title="Select Config / 选择配置", filetypes=[("Config", "*.yaml")])
        if not c_path: return 
        try:
            self.load_model_and_config(pathlib.Path(m_path), pathlib.Path(c_path))
        except Exception as e:
            messagebox.showerror("Error / 错误", str(e))

    def process_data(self):
        if self.raw_data is None or self.raw_centroid is None: return None, None
        
        data_jaw = self.raw_data.clone().to(self.device).unsqueeze(0).unsqueeze(0)
        data_cent = self.raw_centroid.clone().to(self.device).unsqueeze(0).unsqueeze(0)
        
        # 1. 独立平滑 Jaw Opening
        sw_jaw = self.smooth_val.get()
        if sw_jaw > 0:
            ks_jaw = int(round(sw_jaw / self.timestep))
            if ks_jaw > 1:
                smoother_jaw = SinusoidalSmoothingConv1d(ks_jaw).to(self.device)
                data_jaw = smoother_jaw(data_jaw)
                
        # 2. 独立平滑 Spectral Centroid (使用专属的大窗口推子)
        sw_cent = self.cent_smooth_val.get()
        if sw_cent > 0:
            ks_cent = int(round(sw_cent / self.timestep))
            if ks_cent > 1:
                smoother_cent = SinusoidalSmoothingConv1d(ks_cent).to(self.device)
                data_cent = smoother_cent(data_cent)
                
        data_jaw = torch.clamp(data_jaw * self.scale_val.get(), 0, 1)
        data_cent = torch.clamp(data_cent, 0, 1)
        
        target_n = int(max(1, self.audio_duration * self.fps_val.get()))
        
        data_jaw = F.interpolate(data_jaw, size=target_n, mode='linear', align_corners=False)
        data_cent = F.interpolate(data_cent, size=target_n, mode='linear', align_corners=False)
        
        return data_jaw.squeeze().cpu().numpy(), data_cent.squeeze().cpu().numpy()

    def update_plot(self):
        if self.raw_data is None: return
        data_jaw, data_cent = self.process_data()
        
        self.ax.clear()
        self.ax.plot(data_jaw, color=self.accent_color, lw=1.5, label='Mouth Opening')
        self.ax.fill_between(range(len(data_jaw)), data_jaw, color=self.accent_color, alpha=0.1)
        
        self.ax.plot(data_cent, color=self.centroid_color, lw=1.0, alpha=0.5, label='Spectral Brightness')
        
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.grid(True, color='#333333', linestyle='--', alpha=0.8)
        self.canvas.draw()

    def export(self, fmt):
        if self.raw_data is None: return
        data_jaw, data_cent = self.process_data()
        
        suffix = "_arkit" if fmt == "arkit_csv" else "_mouth"
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv" if fmt == "arkit_csv" else f".{fmt}", 
            initialfile=f"{self.wav_path.stem}{suffix}"
        )
        if save_path:
            try:
                if fmt == "csv": 
                    np.savetxt(save_path, data_jaw, delimiter=",", header="MouthOpening", comments='', fmt='%.6f')
                
                elif fmt == "arkit_csv":
                    fps = self.fps_val.get()
                    dt = 1.0 / fps
                    velocities = np.gradient(data_jaw) / dt
                    
                    with open(save_path, "w", newline='') as f:
                        f.write("Frame,Time(s),jawOpen,mouthFunnel,mouthPucker,mouthStretchLeft,mouthStretchRight,mouthShrugUpper\n")
                        for i in range(len(data_jaw)):
                            jaw = data_jaw[i]
                            cent = data_cent[i]
                            vel = velocities[i]
                            
                            funnel = max(0.0, jaw * (1.0 - cent) * 1.5)
                            pucker = max(0.0, (0.2 - jaw) * 3.0 * (1.0 - cent)) if 0.05 < jaw < 0.2 else 0.0
                            stretch = jaw * cent * 1.0
                            shrug = max(0.0, min(vel * 0.1, 1.0)) * 0.5
                            
                            t = i * dt
                            f.write(f"{i},{t:.4f},{jaw:.6f},{funnel:.6f},{pucker:.6f},{stretch:.6f},{stretch:.6f},{shrug:.6f}\n")
                            
                elif fmt == "json":
                    with open(save_path, "w") as f: 
                        rounded_data = np.round(data_jaw, decimals=6)
                        json.dump({"fps": self.fps_val.get(), "data": rounded_data.tolist()}, f)
                elif fmt == "npy": 
                    np.save(save_path, data_jaw)
                messagebox.showinfo("Success / 成功", f"File exported successfully:\n{pathlib.Path(save_path).name}")
            except Exception as e:
                messagebox.showerror("Export Error / 导出失败", str(e))

if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        
    app = MouthBakerUI(root)
    root.mainloop()