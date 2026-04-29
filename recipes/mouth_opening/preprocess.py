import re
import json
import pathlib
import sys
import warnings

import click
import librosa
import numpy
import pandas
import torch
import tqdm
from scipy.interpolate import interp1d

from funasr import AutoModel
from fireredvad import FireRedVad, FireRedVadConfig
from silero_vad import load_silero_vad, get_speech_timestamps
import torchaudio
import onnxruntime as ort
import yaml

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())
from lib.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch

JAW_OPEN = 0
MOUTH_CLOSE = 1
CORRECTED_JAW_OPEN = 2
SUBTRACTED_JAW_OPEN = 3
LIPS_DISTANCE = 4


# ---------- 通用辅助函数 ----------
def find_segments_dynamic(arr, time_scale, threshold=0.5, max_gap=5, ap_threshold=10):
    """从概率序列中提取超越阈值的连续片段（呼吸检测）"""
    segments = []
    start = None
    gap_count = 0
    for i in range(len(arr)):
        if arr[i] >= threshold:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                if gap_count < max_gap:
                    gap_count += 1
                else:
                    end = i - gap_count - 1
                    if end >= start and (end - start) >= ap_threshold:
                        segments.append((start * time_scale, end * time_scale))
                    start = None
                    gap_count = 0
    if start is not None and (len(arr) - start) >= ap_threshold:
        segments.append((start * time_scale, (len(arr) - 1) * time_scale))
    return segments


def load_breath_model(breath_model_path):
    """加载呼吸检测 ONNX 模型及其配置，返回 session, config, time_scale"""
    config_path = pathlib.Path(breath_model_path).with_name('config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Breath model config not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    session = ort.InferenceSession(breath_model_path)
    time_scale = 1.0 / (config['audio_sample_rate'] / config['hop_size'])
    return session, config, time_scale


def run_breath_detection(session, config, audio, ap_threshold=0.5, ap_dur=0.08):
    """返回呼吸片段列表 [(start_sec, end_sec)]，相对于传入的音频"""
    input_data = audio[None, :].astype(numpy.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    prob = outputs[0][0]          # (T,)
    time_scale = 1.0 / (config['audio_sample_rate'] / config['hop_size'])
    ap_threshold_frames = int(ap_dur / time_scale)
    return find_segments_dynamic(prob, time_scale, threshold=ap_threshold,
                                 ap_threshold=ap_threshold_frames)


def merge_intervals(intervals, gap_merge):
    """合并相邻间隔 ≤ gap_merge 的区间，返回排序后的列表"""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= gap_merge:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def subtract_intervals(intervals, sub_intervals):
    """从 intervals 中去除与 sub_intervals 重叠的部分"""
    if not sub_intervals:
        return intervals
    result = []
    for s, e in intervals:
        parts = [(s, e)]
        for a, b in sorted(sub_intervals):
            new_parts = []
            for p_s, p_e in parts:
                if p_e <= a or p_s >= b:
                    new_parts.append((p_s, p_e))
                else:
                    if p_s < a:
                        new_parts.append((p_s, a))
                    if p_e > b:
                        new_parts.append((b, p_e))
            parts = new_parts
            if not parts:
                break
        result.extend(parts)
    return result


def map_segments_to_segment(segments_absolute, seg_start, seg_duration):
    """
    将绝对时间片段映射到当前音频段（seg_start 起始，长度 seg_duration）的相对时间。
    segments_absolute: [(start_sec, end_sec), ...] 相对于完整音频文件
    返回：相对于音频段头的片段列表
    """
    seg_end = seg_start + seg_duration
    mapped = []
    for s, e in segments_absolute:
        if e <= seg_start or s >= seg_end:
            continue
        mapped.append((
            max(s - seg_start, 0.0),
            min(e - seg_start, seg_duration)
        ))
    return mapped


# ---------- 预处理主流程 ----------
@click.command()
@click.argument('source_dir', type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument('target_dir', type=click.Path(path_type=pathlib.Path))
@click.option('--val_list', type=click.Path(exists=True, path_type=pathlib.Path),
              help='Validation file list (relative paths, one per line).')
@click.option('--val_num', default=5, type=int)
@click.option('--attr_type', default=SUBTRACTED_JAW_OPEN, type=int)
@click.option('--subtraction_offset', default=0.0, type=float,
              help='Offset for subtracted jawOpen (jawOpen - mouthClose + offset), no upper bound.')
@click.option("--use_mask", is_flag=True, default=False,
              help="Enable multi-source VAD + breath + ASS masking.")
@click.option('--breath_model_path', type=click.Path(exists=True, path_type=pathlib.Path),
              help='Path to breath detection ONNX model (required if --use_mask).')
@click.option('--firered_vad_path', type=click.Path(exists=True, path_type=pathlib.Path),
              help='Path to FireRedVAD model directory.')
@click.option('--silero_vad_path', type=click.Path(exists=True, path_type=pathlib.Path),
              help='Optional path to silero VAD .jit file. If not given, default model will be downloaded/used.')
@click.option('--mask_gap_merge', default=0.08, type=float,
              help='Merge valid segments with gap smaller than this value (seconds).')
@click.option('--sample_rate', default=16000, type=int)
@click.option('--mel_bins', default=80, type=int)
@click.option('--hop_size', default=320, type=int)
@click.option('--win_size', default=1024, type=int)
@click.option('--f_min', default=0, type=int)
@click.option('--f_max', default=None, type=int)
def preprocess(source_dir, target_dir, val_list, val_num, attr_type,
               subtraction_offset, use_mask, breath_model_path,
               firered_vad_path, silero_vad_path, mask_gap_merge,
               sample_rate, mel_bins, hop_size, win_size, f_min, f_max):
    metadata = {
        "sample_rate": sample_rate,
        "mel_bins": mel_bins,
        "hop_size": hop_size,
        "win_size": win_size,
        "f_min": f_min,
        "f_max": f_max
    }

    # ---------- 初始化多 VAD 及呼吸模型 ----------
    fsmn_vad = None
    firered_vad = None
    silero_model = None
    breath_session = None
    breath_config = None

    if use_mask:
        if not breath_model_path:
            raise click.UsageError("--breath_model_path is required when --use_mask is set.")

        # fsmn-vad
        fsmn_vad = AutoModel(
            model="fsmn-vad", model_revision="v2.0.4",
            disable_update=True, log_level="ERROR",
            disable_pbar=True, disable_log=True
        )
        print('fsmn-vad ready')
        # FireRedVAD
        if firered_vad_path:
            vad_cfg = FireRedVadConfig(
                use_gpu=False,
                smooth_window_size=5,
                speech_threshold=0.4,
                min_speech_frame=20,
                max_speech_frame=2000,
                min_silence_frame=20,
                merge_silence_frame=0,
                extend_speech_frame=0,
                chunk_max_frame=30000
            )
            firered_vad = FireRedVad.from_pretrained(str(firered_vad_path), vad_cfg)
            print('firered_vad ready')
        else:
            print("Warning: FireRedVAD path not provided, skipping FireRedVAD.")

        # silero-vad
        if silero_vad_path:
            silero_model = load_silero_vad(model_path=str(silero_vad_path))
        else:
            silero_model = load_silero_vad()   # 自动下载/使用缓存
            print("Using default silero VAD (auto-downloaded if necessary).")

        # 呼吸检测模型
        breath_session, breath_config, _ = load_breath_model(breath_model_path)

    csv_list = sorted(source_dir.rglob("mouth_data.csv"))
    len_list = []
    npz_list = []

    mel_spec_transform = PitchAdjustableMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=win_size,
        win_length=win_size,
        hop_length=hop_size,
        f_min=f_min,
        f_max=f_max,
        n_mels=mel_bins,
        center=True
    )

    with tqdm.tqdm(csv_list) as bar:
        for csv_file in bar:
            df = pandas.read_csv(csv_file)
            xs = df["TimeStamp"].values
            x_raw_diff = df["jawOpen"].values - df["mouthClose"].values
            jaw_open = df["jawOpen"].values
            mouth_close = df["mouthClose"].values
            lips_distance = df["LipsDistance"].values if "LipsDistance" in df.columns else None
            original_x0 = xs[0]

            # 自动裁剪开头错误帧（不影响后续对齐）
            crop_end_time = None
            err_segs = process_error_value(xs, x_raw_diff)
            if err_segs:
                crop_end_time = err_segs[0][1]

            if crop_end_time is not None and crop_end_time > original_x0:
                keep = xs >= crop_end_time
                xs = xs[keep]
                x_raw_diff = x_raw_diff[keep]
                jaw_open = jaw_open[keep]
                mouth_close = mouth_close[keep]
                if lips_distance is not None:
                    lips_distance = lips_distance[keep]

            if len(xs) < 2:
                bar.write(f"Warning: insufficient data after crop in {csv_file}, skipped")
                continue

            # 选择目标曲线（仅下限约束，无上限）
            if attr_type == JAW_OPEN:
                ys = jaw_open
            elif attr_type == MOUTH_CLOSE:
                ys = mouth_close
            elif attr_type == CORRECTED_JAW_OPEN:
                ys = jaw_open * (1 - mouth_close)
            elif attr_type == SUBTRACTED_JAW_OPEN:
                ys = x_raw_diff + subtraction_offset
            elif attr_type == LIPS_DISTANCE:
                if lips_distance is None:
                    bar.write(f"Warning: LipsDistance column missing in {csv_file}, skipped")
                    continue
                ys = lips_distance
            else:
                raise ValueError(f"Invalid attr_type: {attr_type}")

            if len(ys) < 2:
                bar.write(f"Warning: empty data in {csv_file}")
                continue

            offset = round(sample_rate * xs[0])          # 裁剪后音频起点（样本）
            size = round(sample_rate * (xs[-1] - xs[0])) # 所需音频长度（样本）
            xs_rel = xs - xs[0]                          # 曲线时间从 0 开始

            # 处理每个 wav 文件
            for audio_file in csv_file.parent.glob("*.wav"):
                audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
                if len(audio) < offset + size:
                    audio = numpy.pad(audio, (0, offset + size - len(audio)))
                audio_segment = audio[offset: offset + size]

                # 梅尔频谱
                mel = dynamic_range_compression_torch(mel_spec_transform(
                    torch.from_numpy(audio_segment)[None]
                ), clip_val=1e-9)[0].T.cpu().numpy()

                # 曲线插值
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    interp_fn = interp1d(xs_rel, ys, kind="linear",
                                         fill_value="extrapolate", bounds_error=False)
                    t_mel = numpy.linspace(0, len(audio_segment) / sample_rate, mel.shape[0])
                    curve = interp_fn(t_mel).astype(numpy.float32)

                # ---------- 多源掩码构建 ----------
                if use_mask:
                    seg_start_time = offset / sample_rate          # 当前音频段在原文件中的起始时间（秒）
                    seg_duration = len(audio_segment) / sample_rate

                    valid_segments = []   # 存放相对时间

                    # 1. fsmn-vad (返回毫秒)
                    try:
                        raw = fsmn_vad.generate(str(audio_file))[0]["value"]
                        abs_s = [(s/1000.0, e/1000.0) for s, e in raw]
                        valid_segments.extend(map_segments_to_segment(abs_s, seg_start_time, seg_duration))
                    except Exception as e:
                        bar.write(f"Warning: fsmn-vad failed for {audio_file}: {e}")

                    # 2. FireRedVAD (返回秒)
                    if firered_vad:
                        try:
                            # 重采样到 16kHz 并保存临时文件
                            audio_fr, _ = librosa.load(audio_file, sr=16000, mono=True)
                            import tempfile, os, soundfile as sf
                            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                            os.close(tmp_fd)
                            sf.write(tmp_path, audio_fr, 16000, subtype='PCM_16')
                            result, _ = firered_vad.detect(tmp_path)
                            os.unlink(tmp_path)
                            abs_s = result['timestamps']
                            valid_segments.extend(map_segments_to_segment(abs_s, seg_start_time, seg_duration))
                        except Exception as e:
                            bar.write(f"Warning: FireRedVAD failed for {audio_file}: {e}")

                    # 3. silero-vad (返回秒)
                    if silero_model:
                        try:
                            wav_silero, sr_sil = torchaudio.load(audio_file)
                            if sr_sil != 16000:
                                resampler = torchaudio.transforms.Resample(sr_sil, 16000)
                                wav_silero = resampler(wav_silero)
                            ts = get_speech_timestamps(wav_silero.squeeze(), silero_model, return_seconds=True)
                            abs_s = [(t['start'], t['end']) for t in ts]
                            valid_segments.extend(map_segments_to_segment(abs_s, seg_start_time, seg_duration))
                        except Exception as e:
                            bar.write(f"Warning: silero VAD failed for {audio_file}: {e}")

                    # 4. 呼吸检测（需要重采样至模型指定采样率）
                    if breath_session:
                        try:
                            breath_sr = breath_config['audio_sample_rate']
                            if breath_sr != sample_rate:
                                breath_audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=breath_sr)
                            else:
                                breath_audio = audio
                            # 呼吸检测作用于完整音频，返回绝对时间
                            abs_breath = run_breath_detection(breath_session, breath_config, breath_audio)
                            valid_segments.extend(map_segments_to_segment(abs_breath, seg_start_time, seg_duration))
                        except Exception as e:
                            bar.write(f"Warning: breath detection failed for {audio_file}: {e}")

                    # 5. ASS 排除区域（若文件存在）
                    ass_segments_rel = []
                    ass_path = audio_file.with_name("mask.ass")
                    if ass_path.exists():
                        abs_ass = ass_to_time_array(ass_path)
                        ass_segments_rel = map_segments_to_segment(abs_ass, seg_start_time, seg_duration)

                    # 6. 汇总：所有 VAD+呼吸 的并集，去掉 ASS 部分，再合并短间隙
                    union_valid = merge_intervals(valid_segments, 0.0)
                    cleaned = subtract_intervals(union_valid, ass_segments_rel)
                    final_valid = merge_intervals(cleaned, mask_gap_merge)

                    # 7. 根据帧长度生成 mask
                    mask = numpy.zeros(curve.shape[0], dtype=numpy.float32)
                    frame_time = hop_size / sample_rate
                    for s, e in final_valid:
                        start_frame = int(round(s / frame_time))
                        end_frame = int(round(e / frame_time))
                        start_frame = max(0, min(start_frame, mask.shape[0]))
                        end_frame = max(0, min(end_frame, mask.shape[0]))
                        mask[start_frame:end_frame] = 1.0
                    curve *= mask

                # 保存 npz
                target_file = target_dir / audio_file.relative_to(source_dir).with_suffix(".npz")
                target_file.parent.mkdir(parents=True, exist_ok=True)
                numpy.savez(target_file, spectrogram=mel, curve=curve)
                len_list.append(mel.shape[0])
                npz_list.append(target_file.relative_to(target_dir).as_posix())

    # ---------- 划分训练/验证集 ----------
    if val_list is not None:
        with open(val_list, "r", encoding="utf8") as f:
            val_files = {pathlib.Path(line.strip()).with_suffix(".npz").as_posix() for line in f}
        val_indices = [i for i, npz_file in enumerate(npz_list) if npz_file in val_files]
    else:
        val_indices = sorted(numpy.random.choice(len(len_list), min(val_num, len(len_list)), replace=False))

    lengths = []
    with open(target_dir / "train.txt", "w") as f:
        for i, npz_file in enumerate(npz_list):
            if i not in val_indices:
                f.write(str(npz_file) + "\n")
                lengths.append(len_list[i])
    with open(target_dir / "valid.txt", "w") as f:
        for i in val_indices:
            f.write(npz_list[i] + "\n")

    numpy.save(target_dir / "lengths.npy", lengths)
    with open(target_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"Processed {len(npz_list)} valid samples.")


def process_error_value(timestamps, values):
    if len(values) < 2:
        return []
    first_val = values[0]
    change_idx = -1
    for i in range(1, len(values)):
        if values[i] != first_val:
            change_idx = i
            break
    if change_idx == -1:
        return [(timestamps[0], timestamps[-1])]
    elif change_idx == 1:
        return []
    else:
        return [(timestamps[0], timestamps[change_idx])]


def ass_to_time_array(ass_file):
    with open(ass_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    event_lines = [line for line in lines if line.startswith("Dialogue") or line.startswith("Comment")]
    time_array = []
    for event_line in event_lines:
        times = re.findall(r"\d:\d{2}:\d{2,3}.\d{2}", event_line)
        if times:
            start_time, end_time = times
            start_seconds = sum(float(x) * 60 ** (2 - i) for i, x in enumerate(start_time.split(":")))
            end_seconds = sum(float(x) * 60 ** (2 - i) for i, x in enumerate(end_time.split(":")))
            time_array.append((start_seconds, end_seconds))
    return time_array


if __name__ == "__main__":
    preprocess()