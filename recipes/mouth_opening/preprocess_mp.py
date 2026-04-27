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

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())
from lib.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch

JAW_OPEN = 0
MOUTH_CLOSE = 1
CORRECTED_JAW_OPEN = 2
SUBTRACTED_JAW_OPEN = 3
LIPS_DISTANCE = 4


def read_val_list(val_list_path, encoding_candidates=('utf-8-sig', 'gbk', 'latin-1')):
    for enc in encoding_candidates:
        try:
            with open(val_list_path, 'r', encoding=enc) as f:
                return {pathlib.Path(line.strip()).with_suffix(".npz").as_posix() for line in f}
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to decode {val_list_path} with any of {encoding_candidates}")


def process_single(csv_file, wav_file, args, lock_vad=None):
    """
    处理一个 (csv, wav) 对，返回 (success, length, npz_rel_path, message)
    message 用于记录 skip/warning 原因
    """
    source_dir = args["source_dir"]
    target_dir = args["target_dir"]
    attr_type = args["attr_type"]
    subtraction_offset = args["subtraction_offset"]
    use_mask = args["use_mask"]
    sample_rate = args["sample_rate"]
    hop_size = args["hop_size"]
    mel_spec_transform = args["mel_spec_transform"]
    vad = args.get("vad")

    try:
        # 读取 CSV
        df = pandas.read_csv(csv_file)
        xs = df["TimeStamp"].values
        x_raw_diff = df["jawOpen"].values - df["mouthClose"].values
        jaw_open = df["jawOpen"].values
        mouth_close = df["mouthClose"].values
        lips_distance = df["LipsDistance"].values if "LipsDistance" in df.columns else None

        original_x0 = df["TimeStamp"].values[0]
        crop_end_time = None

        # 自动检测并裁剪开头错误段
        err_segs = process_error_value(df["TimeStamp"].values, x_raw_diff)
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
            ys = numpy.clip(x_raw_diff + subtraction_offset, a_min=0.0, a_max=1.0)
        elif attr_type == LIPS_DISTANCE:
            if lips_distance is None:
                return False, None, None, "skip: LipsDistance column missing in CSV"
            ys = lips_distance
        else:
            raise ValueError(f"Invalid attr_type: {attr_type}")

        if len(ys) < 2:
            return False, None, None, f"skip: empty attribute data (len {len(ys)})"

        offset = round(sample_rate * xs[0])
        size = round(sample_rate * (xs[-1] - xs[0]))
        xs_rel = xs - xs[0]

        # 加载音频
        audio, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
        if len(audio) < offset + size:
            audio = numpy.pad(audio, (0, offset + size - len(audio)))
        audio = audio[offset:offset + size]

        # 计算 mel 谱
        mel = dynamic_range_compression_torch(mel_spec_transform(
            torch.from_numpy(audio)[None]
        ), clip_val=1e-9)[0].T.cpu().numpy()

        # 插值曲线
        t_mel = numpy.linspace(0, len(audio) / sample_rate, mel.shape[0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            interp_fn = interp1d(xs_rel, ys, kind="linear", fill_value="extrapolate", bounds_error=False)
            curve = interp_fn(t_mel).astype(numpy.float32)
            warning_msg = None
            for wi in w:
                if issubclass(wi.category, RuntimeWarning) and "divide" in str(wi.message):
                    warning_msg = f"warning: interpolation division by zero (duplicate x values) for {csv_file.relative_to(source_dir).as_posix()}"
                    break

        # 修复 NaN
        if numpy.any(numpy.isnan(curve)):
            curve = pandas.Series(curve).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            warning_msg = (warning_msg or "") + " -> NaN fixed by ffill/bfill"

        # Mask 处理
        if use_mask:
            ass_path = csv_file.parent / "mask.ass"
            if ass_path.exists():
                segments = ass_to_time_array(ass_path)
                time_scale = 1
                is_ass = True
            else:
                if lock_vad is not None:
                    with lock_vad:
                        if args.get("vad") is None:
                            from funasr import AutoModel
                            args["vad"] = AutoModel(
                                model="fsmn-vad", model_revision="v2.0.4", disable_update=True,
                                log_level="ERROR", disable_pbar=True, disable_log=True
                            )
                        vad_local = args["vad"]
                else:
                    vad_local = vad
                segments = vad_local.generate(wav_file.as_posix())[0]["value"]
                time_scale = 1 / 1000
                is_ass = False

            mask = numpy.zeros_like(curve)
            for start_ms, end_ms in segments:
                start = min(round(start_ms * time_scale * sample_rate / hop_size), mask.shape[0])
                end = min(round(end_ms * time_scale * sample_rate / hop_size), mask.shape[0])
                mask[start:end] = 1
            if is_ass:
                mask = 1 - mask
            curve *= mask

        # 保存 npz
        target_file = target_dir / wav_file.relative_to(source_dir).with_suffix(".npz")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez(target_file, spectrogram=mel, curve=curve)

        message = warning_msg if warning_msg else None
        return True, mel.shape[0], target_file.relative_to(target_dir).as_posix(), message

    except Exception as e:
        return False, None, None, f"skip: exception - {str(e)}"


@click.command()
@click.argument('source_dir', type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument('target_dir', type=click.Path(path_type=pathlib.Path))
@click.option('--val_list', type=click.Path(exists=True, path_type=pathlib.Path),
              help='Validation file list in a text file.')
@click.option('--val_num', default=5, type=int)
@click.option('--attr_type', default=SUBTRACTED_JAW_OPEN, type=int)
@click.option('--subtraction_offset', default=0.05, type=float,
              help='Offset for subtracted jawOpen attribute (X = jawOpen - mouthClose + offset).')
@click.option("--use_mask", is_flag=True, default=False)
@click.option('--sample_rate', default=16000, type=int)
@click.option('--mel_bins', default=80, type=int)
@click.option('--hop_size', default=320, type=int)
@click.option('--win_size', default=1024, type=int)
@click.option('--f_min', default=0, type=int)
@click.option('--f_max', default=None, type=int)
@click.option('--num_workers', default=None, type=int,
              help='Number of worker threads. Defaults to CPU count.')
@click.option('--val_encoding', default=None, type=str,
              help='Encoding for val_list file (e.g., utf-8, gbk). Auto-detect if not given.')
def preprocess(source_dir, target_dir, val_list, val_num, attr_type,
               subtraction_offset, use_mask, sample_rate, mel_bins, hop_size,
               win_size, f_min, f_max, num_workers, val_encoding):
    metadata = {
        "sample_rate": sample_rate,
        "mel_bins": mel_bins,
        "hop_size": hop_size,
        "win_size": win_size,
        "f_min": f_min,
        "f_max": f_max
    }

    # 收集所有 (csv, wav) 任务对
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

    # 共享资源
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
        "vad": None,
    }
    lock_vad = threading.Lock() if use_mask else None

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

    # 保存消息记录
    if messages:
        msg_csv = target_dir / "processing_messages.csv"
        msg_csv.parent.mkdir(parents=True, exist_ok=True)
        df_msg = pandas.DataFrame(messages, columns=["file", "message"])
        df_msg.to_csv(msg_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(messages)} messages to {msg_csv}")

    # 收集成功处理的样本
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