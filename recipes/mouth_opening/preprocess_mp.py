import re
import json
import pathlib
import sys
import threading
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

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

import tempfile, os, soundfile as sf

JAW_OPEN = 0
MOUTH_CLOSE = 1
CORRECTED_JAW_OPEN = 2
SUBTRACTED_JAW_OPEN = 3
LIPS_DISTANCE = 4


# ---------- 通用辅助函数 ----------
def find_segments_dynamic(arr, time_scale, threshold=0.5, max_gap=5, ap_threshold=10):
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
    config_path = pathlib.Path(breath_model_path).with_name('config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Breath model config not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    session = ort.InferenceSession(breath_model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    time_scale = 1.0 / (config['audio_sample_rate'] / config['hop_size'])
    return session, config, time_scale


def run_breath_detection(session, config, audio, ap_threshold=0.5, ap_dur=0.08):
    input_data = audio[None, :].astype(numpy.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    prob = outputs[0][0]
    time_scale = 1.0 / (config['audio_sample_rate'] / config['hop_size'])
    ap_threshold_frames = int(ap_dur / time_scale)
    return find_segments_dynamic(prob, time_scale, threshold=ap_threshold,
                                 ap_threshold=ap_threshold_frames)


def merge_intervals(intervals, gap_merge):
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
    """将绝对时间片段映射到音频段相对时间"""
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


# ---------- 多线程工作函数 ----------
def process_single(csv_file, wav_file, args, lock_vad):
    source_dir = args["source_dir"]
    target_dir = args["target_dir"]
    attr_type = args["attr_type"]
    subtraction_offset = args["subtraction_offset"]
    use_mask = args["use_mask"]
    sample_rate = args["sample_rate"]
    hop_size = args["hop_size"]
    mel_spec_transform = args["mel_spec_transform"]
    mask_gap_merge = args["mask_gap_merge"]

    fsmn_vad = args.get("fsmn_vad")
    firered_vad = args.get("firered_vad")
    silero_model = args.get("silero_model")
    breath_session = args.get("breath_session")
    breath_config = args.get("breath_config")

    try:
        df = pandas.read_csv(csv_file)
        xs = df["TimeStamp"].values
        x_raw_diff = df["jawOpen"].values - df["mouthClose"].values
        jaw_open = df["jawOpen"].values
        mouth_close = df["mouthClose"].values
        lips_distance = df["LipsDistance"].values if "LipsDistance" in df.columns else None
        original_x0 = xs[0]

        # 裁剪错误帧
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
            return False, None, None, f"skip: insufficient data after crop (len {len(xs)})"

        # 选择曲线
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
                return False, None, None, "skip: LipsDistance column missing"
            ys = lips_distance
        else:
            raise ValueError(f"Invalid attr_type: {attr_type}")

        if len(ys) < 2:
            return False, None, None, f"skip: empty attribute data (len {len(ys)})"

        offset = round(sample_rate * xs[0])
        size = round(sample_rate * (xs[-1] - xs[0]))
        xs_rel = xs - xs[0]

        audio, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
        if len(audio) < offset + size:
            audio = numpy.pad(audio, (0, offset + size - len(audio)))
        audio_segment = audio[offset: offset + size]

        mel = dynamic_range_compression_torch(mel_spec_transform(
            torch.from_numpy(audio_segment)[None]
        ), clip_val=1e-9)[0].T.cpu().numpy()

        t_mel = numpy.linspace(0, len(audio_segment) / sample_rate, mel.shape[0])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            interp_fn = interp1d(xs_rel, ys, kind="linear", fill_value="extrapolate", bounds_error=False)
            curve = interp_fn(t_mel).astype(numpy.float32)
        #if numpy.any(numpy.isnan(curve)):
        #    curve = numpy.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- 多源掩码 ----------
        if use_mask:
            seg_start_time = offset / sample_rate
            seg_duration = len(audio_segment) / sample_rate
            valid_segments = []

            # 所有模型推理加锁保证线程安全
            with lock_vad:
                if fsmn_vad:
                    try:
                        raw = fsmn_vad.generate(str(wav_file))[0]["value"]
                        abs_s = [(s/1000.0, e/1000.0) for s, e in raw]
                        valid_segments.extend(map_segments_to_segment(abs_s, seg_start_time, seg_duration))
                    except Exception:
                        pass

                if firered_vad:
                    try:
                        audio_fr, _ = librosa.load(wav_file, sr=16000, mono=True)
                        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                        os.close(tmp_fd)
                        sf.write(tmp_path, audio_fr, 16000, subtype='PCM_16')
                        result, _ = firered_vad.detect(tmp_path)
                        os.unlink(tmp_path)
                        abs_s = result['timestamps']
                        valid_segments.extend(map_segments_to_segment(abs_s, seg_start_time, seg_duration))
                    except Exception:
                        pass

                if silero_model:
                    try:
                        wav_silero, sr_sil = torchaudio.load(wav_file)
                        if sr_sil != 16000:
                            resampler = torchaudio.transforms.Resample(sr_sil, 16000)
                            wav_silero = resampler(wav_silero)
                        ts = get_speech_timestamps(wav_silero.squeeze(), silero_model, return_seconds=True)
                        abs_s = [(t['start'], t['end']) for t in ts]
                        valid_segments.extend(map_segments_to_segment(abs_s, seg_start_time, seg_duration))
                    except Exception:
                        pass

                if breath_session:
                    try:
                        breath_sr = breath_config['audio_sample_rate']
                        breath_audio = audio if breath_sr == sample_rate else librosa.resample(
                            audio, orig_sr=sample_rate, target_sr=breath_sr
                        )
                        abs_breath = run_breath_detection(breath_session, breath_config, breath_audio)
                        valid_segments.extend(map_segments_to_segment(abs_breath, seg_start_time, seg_duration))
                    except Exception:
                        pass

            # ASS 排除区域（无需锁）
            ass_segments_rel = []
            ass_path = csv_file.parent / "mask.ass"
            if ass_path.exists():
                abs_ass = ass_to_time_array(ass_path)
                ass_segments_rel = map_segments_to_segment(abs_ass, seg_start_time, seg_duration)

            # 汇总
            union_valid = merge_intervals(valid_segments, 0.0)
            cleaned = subtract_intervals(union_valid, ass_segments_rel)
            final_valid = merge_intervals(cleaned, mask_gap_merge)

            mask = numpy.zeros(curve.shape[0], dtype=numpy.float32)
            frame_time = hop_size / sample_rate
            for s, e in final_valid:
                start_frame = int(round(s / frame_time))
                end_frame = int(round(e / frame_time))
                start_frame = max(0, min(start_frame, mask.shape[0]))
                end_frame = max(0, min(end_frame, mask.shape[0]))
                mask[start_frame:end_frame] = 1.0
            curve *= mask

        target_file = target_dir / wav_file.relative_to(source_dir).with_suffix(".npz")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez(target_file, spectrogram=mel, curve=curve)

        return True, mel.shape[0], target_file.relative_to(target_dir).as_posix(), None

    except Exception as e:
        return False, None, None, f"skip: exception - {str(e)}"


# ---------- 主命令 ----------
@click.command()
@click.argument('source_dir', type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument('target_dir', type=click.Path(path_type=pathlib.Path))
@click.option('--val_list', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option('--val_num', default=5, type=int)
@click.option('--attr_type', default=SUBTRACTED_JAW_OPEN, type=int)
@click.option('--subtraction_offset', default=0.0, type=float)
@click.option("--use_mask", is_flag=True, default=False)
@click.option('--breath_model_path', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option('--firered_vad_path', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option('--silero_vad_path', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option('--mask_gap_merge', default=0.08, type=float)
@click.option('--sample_rate', default=16000, type=int)
@click.option('--mel_bins', default=80, type=int)
@click.option('--hop_size', default=320, type=int)
@click.option('--win_size', default=1024, type=int)
@click.option('--f_min', default=0, type=int)
@click.option('--f_max', default=None, type=int)
@click.option('--num_workers', default=None, type=int)
@click.option('--val_encoding', default=None, type=str)
def preprocess(source_dir, target_dir, val_list, val_num, attr_type,
               subtraction_offset, use_mask, breath_model_path,
               firered_vad_path, silero_vad_path, mask_gap_merge,
               sample_rate, mel_bins, hop_size, win_size, f_min, f_max,
               num_workers, val_encoding):
    metadata = {
        "sample_rate": sample_rate,
        "mel_bins": mel_bins,
        "hop_size": hop_size,
        "win_size": win_size,
        "f_min": f_min,
        "f_max": f_max
    }

    # 初始化共享模型
    fsmn_vad = None
    firered_vad = None
    silero_model = None
    breath_session = None
    breath_config = None
    lock_vad = threading.Lock() if use_mask else None

    if use_mask:
        if not breath_model_path:
            raise click.UsageError("--breath_model_path is required when --use_mask is set.")

        fsmn_vad = AutoModel(
            model="fsmn-vad", model_revision="v2.0.4",
            disable_update=True, log_level="ERROR",
            disable_pbar=True, disable_log=True
        )

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
        else:
            print("Warning: FireRedVAD path not provided, skipping FireRedVAD.")

        if silero_vad_path:
            silero_model = load_silero_vad(model_path=str(silero_vad_path))
        else:
            silero_model = load_silero_vad()
            print("Using default silero VAD (auto-downloaded if necessary).")

        breath_session, breath_config, _ = load_breath_model(breath_model_path)

    # 收集任务
    tasks = []
    csv_list = sorted(source_dir.rglob("mouth_data.csv"))
    for csv_file in csv_list:
        wav_files = list(csv_file.parent.glob("*.wav"))
        if not wav_files:
            rel_path = csv_file.relative_to(source_dir).as_posix()
            print(f"Warning: no .wav found for {rel_path}, skipped")
            continue
        for wav_file in wav_files:
            tasks.append((csv_file, wav_file))

    if not tasks:
        print("No valid (csv, wav) pairs found. Exiting.")
        return

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

    shared_args = {
        "source_dir": source_dir,
        "target_dir": target_dir,
        "attr_type": attr_type,
        "subtraction_offset": subtraction_offset,
        "use_mask": use_mask,
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "mel_spec_transform": mel_spec_transform,
        "mask_gap_merge": mask_gap_merge,
        "fsmn_vad": fsmn_vad,
        "firered_vad": firered_vad,
        "silero_model": silero_model,
        "breath_session": breath_session,
        "breath_config": breath_config,
    }

    if num_workers is None:
        num_workers = os.cpu_count() or 16
    max_workers = min(num_workers, 16)
    print(f"Using {max_workers} worker threads, total tasks: {len(tasks)}")

    results = {}
    messages = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single, csv_file, wav_file, shared_args, lock_vad): (csv_file, wav_file)
            for csv_file, wav_file in tasks
        }
        with tqdm.tqdm(total=len(tasks), desc="Processing") as pbar:
            for future in as_completed(future_to_task):
                csv_file, wav_file = future_to_task[future]
                try:
                    success, length, npz_path, msg = future.result()
                    if success:
                        results[(csv_file, wav_file)] = (length, npz_path)
                        if msg:
                            messages.append((wav_file.relative_to(source_dir).as_posix(), msg))
                    else:
                        rel_path = wav_file.relative_to(source_dir).as_posix()
                        messages.append((rel_path, msg))
                except Exception as e:
                    rel_path = wav_file.relative_to(source_dir).as_posix()
                    messages.append((rel_path, f"skip: unhandled exception - {str(e)}"))
                pbar.update(1)

    if messages:
        msg_csv = target_dir / "processing_messages.csv"
        msg_csv.parent.mkdir(parents=True, exist_ok=True)
        df_msg = pandas.DataFrame(messages, columns=["file", "message"])
        df_msg.to_csv(msg_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(messages)} messages to {msg_csv}")

    len_list = []
    npz_list = []
    for csv_file, wav_file in tasks:
        if (csv_file, wav_file) in results:
            length, npz_path = results[(csv_file, wav_file)]
            len_list.append(length)
            npz_list.append(npz_path)

    if not npz_list:
        print("No valid samples processed. Exiting.")
        return

    # 划分训练/验证集
    if val_list is not None:
        if val_encoding:
            with open(val_list, "r", encoding=val_encoding) as f:
                val_files = {pathlib.Path(line.strip()).with_suffix(".npz").as_posix() for line in f}
        else:
            val_files = read_val_list(val_list)
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

    print(f"Processed {len(npz_list)} valid samples, {len(messages)} messages (skips/warnings).")


def read_val_list(val_list_path, encoding_candidates=('utf-8-sig', 'gbk', 'latin-1')):
    for enc in encoding_candidates:
        try:
            with open(val_list_path, 'r', encoding=enc) as f:
                return {pathlib.Path(line.strip()).with_suffix(".npz").as_posix() for line in f}
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to decode {val_list_path} with any of {encoding_candidates}")


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