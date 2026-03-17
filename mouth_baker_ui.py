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
        self.root.title("Mouth Opening Baker | NLE Style")
        self.root.geometry("1200x880") # 稍微增加一点高度容纳新选项
        
        self.bg_color = "#252526"      
        self.panel_bg = "#1e1e1e"      
        self.fg_color = "#cccccc"      
        self.accent_color = "#E91E63"  
        self.btn_bg = "#3a3d41"        
        self.border_color = "#3e3e42"  
        self.font_main = ("Segoe UI", 9)
        self.font_bold = ("Segoe UI", 9, "bold")
        
        self.root.configure(bg=self.bg_color)
        self.apply_modern_style()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.raw_data = None
        self.wav_path = None
        self.playback_wav_path = None 
        self.audio_duration = 0
        self.hparams = None
        self.timestep = 0.01
        
        self.is_playing = False
        self.start_time = 0
        self.anim_data = None
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
        
        # Combo box style
        style.configure("TCombobox", fieldbackground=self.panel_bg, background=self.btn_bg, foreground="#ffffff")

    def on_closing(self):
        self.stop_playback()
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

        ttk.Label(ctrl_frame, text="Smooth Width / 平滑宽度 (s):").pack(anchor=tk.W)
        self.smooth_val = tk.DoubleVar(value=0.12)
        self.smooth_label = ttk.Label(ctrl_frame, text="0.12", foreground="#ffffff")
        self.smooth_label.pack(anchor=tk.E, pady=(0, 2))
        ttk.Scale(ctrl_frame, from_=0, to=0.5, variable=self.smooth_val, command=self.on_param_change).pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Extraction Scale / 提取倍率:").pack(anchor=tk.W, pady=(10,0))
        self.scale_val = tk.DoubleVar(value=1.0)
        self.scale_label = ttk.Label(ctrl_frame, text="1.00", foreground="#ffffff")
        self.scale_label.pack(anchor=tk.E, pady=(0, 2))
        ttk.Scale(ctrl_frame, from_=0.1, to=3.0, variable=self.scale_val, command=self.on_param_change).pack(fill=tk.X)

        ttk.Label(ctrl_frame, text="Output FPS / 输出帧率:").pack(anchor=tk.W, pady=(10,5))
        self.fps_val = tk.IntVar(value=30)
        ttk.Entry(ctrl_frame, textvariable=self.fps_val).pack(fill=tk.X)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=10)
        
        # --- VIEWPORT SETTINGS 升级版 ---
        shape_frame = ttk.LabelFrame(ctrl_frame, text=" VIEWPORT SETTINGS ")
        shape_frame.pack(fill=tk.X, pady=5, ipadx=10, ipady=10)
        
        # 新增：动画预览模式选择
        ttk.Label(shape_frame, text="Anim Mode / 动画模式:").pack(anchor=tk.W)
        self.anim_mode_val = tk.StringVar(value="Standard (Linear)")
        mode_cb = ttk.Combobox(shape_frame, textvariable=self.anim_mode_val, state="readonly", 
                               values=["Standard (Linear)", "Squash & Stretch (Physics)", "Sprite Sheet (4 Frames)"])
        mode_cb.pack(fill=tk.X, pady=(0, 10))
        mode_cb.bind("<<ComboboxSelected>>", self.on_shape_change)
        
        ttk.Label(shape_frame, text="Mouth Width / 嘴巴宽度:").pack(anchor=tk.W)
        self.mouth_width_val = tk.DoubleVar(value=50)
        ttk.Scale(shape_frame, from_=10, to=150, variable=self.mouth_width_val, command=self.on_shape_change).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(shape_frame, text="Curvature / 曲率 (1=菱形, 2=椭圆, 3=方形):").pack(anchor=tk.W)
        self.mouth_curve_val = tk.DoubleVar(value=2.0)
        ttk.Scale(shape_frame, from_=1.0, to=3.5, variable=self.mouth_curve_val, command=self.on_shape_change).pack(fill=tk.X)

        ttk.Button(ctrl_frame, text="↺ Reset All Settings / 重置设置", command=self.reset_settings).pack(fill=tk.X, pady=(15, 0))

        self.btn_play = ttk.Button(ctrl_frame, text="▶ PLAY / 播放动画", command=self.toggle_playback, style="Accent.TButton")
        self.btn_play.pack(fill=tk.X, pady=15)
        
        ttk.Label(ctrl_frame, text="EXPORT / 导出:").pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(ctrl_frame, text="⭳ CSV (Unity/Live2D)", command=lambda: self.export("csv")).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl_frame, text="⭳ JSON (AE Expression)", command=lambda: self.export("json")).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl_frame, text="⭳ NPY (Raw Data)", command=lambda: self.export("npy")).pack(fill=tk.X, pady=2)

        self.preview_frame = ttk.Frame(self.root, padding=15)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        plot_frame = ttk.LabelFrame(self.preview_frame, text=" CURVE EDITOR ")
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
        self.anim_canvas = tk.Canvas(anim_container, height=200, bg=self.panel_bg, highlightthickness=0)
        self.anim_canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.mouth_polygon = self.anim_canvas.create_polygon(0,0,0,0, fill=self.accent_color, outline="#ffffff", width=2, smooth=False)
        self.anim_canvas.bind("<Configure>", lambda e: self.draw_mouth(0.0))

    def reset_settings(self):
        self.smooth_val.set(0.12)
        self.scale_val.set(1.0)
        self.fps_val.set(30)
        self.mouth_width_val.set(50)
        self.mouth_curve_val.set(2.0)
        self.anim_mode_val.set("Standard (Linear)")
        self.on_param_change()
        self.on_shape_change()

    def on_drop_file(self, event):
        file_path = event.data.strip('{}')
        self.process_audio_file(file_path)

    def load_audio_ui(self):
        if self.model is None:
            messagebox.showerror("Error / 错误", "Model not loaded. / 模型未加载。")
            return
        path = filedialog.askopenfilename(filetypes=[("Audio Files / 音频文件", "*.wav *.mp3 *.flac *.ogg"), ("All Files / 所有文件", "*.*")])
        if path:
            self.process_audio_file(path)

    def process_audio_file(self, path):
        if self.model is None: return
        
        self.stop_playback()
        self.raw_data = None
        self.anim_data = None
        
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
                self.raw_data = self.model(mel).squeeze(0).squeeze(-1).cpu() 
                
            self.update_plot()
            self.draw_mouth(0.0) 
            self.status_label.config(text=f"Ready / 就绪: {self.wav_path.name}", foreground="#4caf50")
            
        except Exception as e:
            messagebox.showerror("Audio Load Error / 音频加载失败", str(e))
            self.status_label.config(text="Load Error / 加载出错", foreground="#f44336")

    def on_param_change(self, _=None):
        self.smooth_label.config(text=f"{self.smooth_val.get():.2f}")
        self.scale_label.config(text=f"{self.scale_val.get():.2f}")
        self.update_plot()
        if not self.is_playing: self.draw_mouth(0.0)

    def on_shape_change(self, _=None):
        if not self.is_playing:
            self.draw_mouth(0.0)

    def stop_playback(self):
        self.is_playing = False
        self.btn_play.config(text="▶ PLAY / 播放动画", style="Accent.TButton")
        self.draw_mouth(0.0)
        
        if OS_NAME == 'Windows' and HAS_WINSOUND:
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif OS_NAME == 'Darwin' and self.audio_process is not None:
            self.audio_process.terminate()
            self.audio_process = None

    def toggle_playback(self):
        if OS_NAME not in ['Windows', 'Darwin']:
            messagebox.showwarning("Notice / 提示", "Playback feature is only supported on Windows or macOS. / 播放功能仅支持 Win/Mac。")
            return
            
        if self.playback_wav_path is None or self.raw_data is None: return

        if self.is_playing:
            self.stop_playback()
        else:
            self.anim_data = self.process_data()
            self.is_playing = True
            self.btn_play.config(text="■ STOP / 停止预览", style="TButton") 
            
            if OS_NAME == 'Windows' and HAS_WINSOUND:
                winsound.PlaySound(str(self.playback_wav_path), winsound.SND_ASYNC | winsound.SND_FILENAME)
            elif OS_NAME == 'Darwin':
                self.audio_process = subprocess.Popen(['afplay', str(self.playback_wav_path)])
            
            self.start_time = time.time()
            self.update_animation()

    def update_animation(self):
        if not self.is_playing: return
        
        elapsed = time.time() - self.start_time
        current_frame = int(elapsed * self.fps_val.get())
        
        if current_frame < len(self.anim_data):
            self.draw_mouth(self.anim_data[current_frame])
            self.root.after(10, self.update_animation)
        else:
            self.stop_playback()

    def draw_mouth(self, val):
        w = self.anim_canvas.winfo_width()
        h = self.anim_canvas.winfo_height()
        if w < 10 or h < 10: return
        cx, cy = w / 2, h / 2
        
        mode = self.anim_mode_val.get()
        
        # 魔法 1：序列帧模式 (Sprite Sheet Quantization)
        if mode == "Sprite Sheet (4 Frames)":
            # 将平滑的 0~1 强行切分为 4 个离散状态: 0.0, 0.33, 0.66, 1.0
            steps = 3 
            val = round(val * steps) / steps
            
        a = self.mouth_width_val.get()   
        b = 2 + val * 60                 
        n = self.mouth_curve_val.get()   
        
        # 魔法 2：挤压与拉伸物理模式 (Squash & Stretch)
        if mode == "Squash & Stretch (Physics)":
            # 嘴巴往下张得越大，横向宽度 'a' 就会往内收缩 (最大收缩 20%)
            a = a * (1.0 - val * 0.2)
        
        points = []
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
        if self.raw_data is None: return None
        data = self.raw_data.clone().to(self.device).unsqueeze(0).unsqueeze(0)
        
        sw = self.smooth_val.get()
        if sw > 0:
            ks = int(round(sw / self.timestep))
            if ks > 1:
                smoother = SinusoidalSmoothingConv1d(ks).to(self.device)
                data = smoother(data)
                
        data = torch.clamp(data * self.scale_val.get(), 0, 1)
        target_n = int(max(1, self.audio_duration * self.fps_val.get()))
        data = F.interpolate(data, size=target_n, mode='linear', align_corners=False)
        return data.squeeze().cpu().numpy()

    def update_plot(self):
        data = self.process_data()
        if data is not None:
            self.ax.clear()
            self.ax.plot(data, color=self.accent_color, lw=1.5)
            self.ax.fill_between(range(len(data)), data, color=self.accent_color, alpha=0.1)
            
            self.ax.set_ylim(-0.05, 1.05)
            self.ax.grid(True, color='#333333', linestyle='--', alpha=0.8)
            self.canvas.draw()

    def export(self, fmt):
        data = self.process_data()
        if data is None: return
        save_path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}", initialfile=f"{self.wav_path.stem}_mouth"
        )
        if save_path:
            try:
                if fmt == "csv": 
                    np.savetxt(save_path, data, delimiter=",", header="MouthOpening", comments='', fmt='%.6f')
                elif fmt == "json":
                    with open(save_path, "w") as f: 
                        rounded_data = np.round(data, decimals=6)
                        json.dump({"fps": self.fps_val.get(), "data": rounded_data.tolist()}, f)
                elif fmt == "npy": 
                    np.save(save_path, data)
                messagebox.showinfo("Success / 成功", f"File exported successfully / 文件导出成功:\n{pathlib.Path(save_path).name}")
            except Exception as e:
                messagebox.showerror("Export Error / 导出失败", str(e))

if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        
    app = MouthBakerUI(root)
    root.mainloop()