import json
import pathlib
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import librosa
import numpy
import torch
import tqdm
import pandas

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())
from lib.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch


def read_val_list(val_list_path, encoding_candidates=('utf-8-sig', 'gbk', 'latin-1')):
    """尝试多种编码读取 val_list 文件，返回集合"""
    for enc in encoding_candidates:
        try:
            with open(val_list_path, 'r', encoding=enc) as f:
                return {pathlib.Path(line.strip()).with_suffix(".npz").as_posix() for line in f}
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to decode {val_list_path} with any of {encoding_candidates}")


def process_single(audio_file, args):
    """
    处理单个音频文件，计算 mel 谱并保存 npz。
    返回 (success, length, npz_rel_path, duration_sec, message)
    """
    source_dir = args["source_dir"]
    target_dir = args["target_dir"]
    sample_rate = args["sample_rate"]
    mel_spec_transform = args["mel_spec_transform"]

    try:
        # 加载音频
        audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
        duration = len(audio) / sample_rate
        # 计算梅尔谱
        mel = dynamic_range_compression_torch(
            mel_spec_transform(torch.from_numpy(audio)[None]),
            clip_val=1e-9
        )[0].T.cpu().numpy()

        # 保存 npz
        target_file = target_dir / audio_file.relative_to(source_dir).with_suffix(".npz")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez(target_file, spectrogram=mel)

        return True, mel.shape[0], target_file.relative_to(target_dir).as_posix(), duration, None

    except Exception as e:
        rel_path = audio_file.relative_to(source_dir).as_posix()
        return False, None, None, 0.0, f"skip: {str(e)}"


@click.command()
@click.argument("source_dir", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("target_dir", type=click.Path(path_type=pathlib.Path))
@click.option("--val_list", type=click.Path(exists=True, path_type=pathlib.Path),
              help="Validation file list in a text file.")
@click.option("--val_num", default=5, type=int,
              help="Number of validation samples for random selection if val_list not provided.")
@click.option("--sample_rate", default=16000, type=int)
@click.option("--mel_bins", default=80, type=int)
@click.option("--hop_size", default=320, type=int)
@click.option("--win_size", default=1024, type=int)
@click.option("--f_min", default=0, type=int)
@click.option("--f_max", default=None, type=int)
@click.option("--num_workers", default=None, type=int,
              help="Number of worker threads. Defaults to CPU count.")
@click.option("--val_encoding", default=None, type=str,
              help="Encoding for val_list file (e.g., utf-8, gbk). Auto-detect if not given.")
def preprocess(source_dir, target_dir, val_list, val_num, sample_rate,
               mel_bins, hop_size, win_size, f_min, f_max, num_workers, val_encoding):
    """多线程预处理：提取梅尔谱，支持常见音频格式"""
    metadata = {
        "sample_rate": sample_rate,
        "mel_bins": mel_bins,
        "hop_size": hop_size,
        "win_size": win_size,
        "f_min": f_min,
        "f_max": f_max
    }

    # 收集所有音频文件（支持多种扩展名）
    extensions = ["*.m4a", "*.wav", "*.mp3", "*.aac"]
    audio_list = sorted([f for ext in extensions for f in source_dir.rglob(ext)])
    if not audio_list:
        print("No audio files found.")
        return

    # 共享的梅尔谱变换器（只读，线程安全）
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
        "sample_rate": sample_rate,
        "mel_spec_transform": mel_spec_transform,
    }

    # 线程数控制（避免内存过载）
    if num_workers is None:
        num_workers = os.cpu_count() or 16
    max_workers = min(num_workers, 16)  # 保守限制，可自行调整
    print(f"Using {max_workers} worker threads, total files: {len(audio_list)}")

    results = {}          # audio_file -> (length, npz_path, duration)
    messages = []         # list of (file, message)
    total_duration = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single, audio_file, shared_args): audio_file
            for audio_file in audio_list
        }
        with tqdm.tqdm(total=len(audio_list), desc="Extracting Mel") as pbar:
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    success, length, npz_path, duration, msg = future.result()
                    if success:
                        results[audio_file] = (length, npz_path, duration)
                        total_duration += duration
                        if msg:
                            messages.append((audio_file.relative_to(source_dir).as_posix(), msg))
                    else:
                        messages.append((audio_file.relative_to(source_dir).as_posix(), msg))
                except Exception as e:
                    rel_path = audio_file.relative_to(source_dir).as_posix()
                    messages.append((rel_path, f"unhandled exception: {str(e)}"))
                pbar.update(1)

    # 保存处理消息（含跳过记录）
    if messages:
        msg_csv = target_dir / "processing_messages.csv"
        msg_csv.parent.mkdir(parents=True, exist_ok=True)
        df_msg = pandas.DataFrame(messages, columns=["file", "message"])
        df_msg.to_csv(msg_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(messages)} messages to {msg_csv}")

    # 收集成功处理的 npz 列表（保持原始顺序便于复现）
    len_list = []
    npz_list = []
    for audio_file in audio_list:
        if audio_file in results:
            length, npz_path, _ = results[audio_file]
            len_list.append(length)
            npz_list.append(npz_path)

    if not npz_list:
        print("No valid samples processed. Exiting.")
        return

    # 划分验证集
    if val_list is not None:
        if val_encoding:
            with open(val_list, "r", encoding=val_encoding) as f:
                val_files = {pathlib.Path(line.strip()).with_suffix(".npz").as_posix() for line in f}
        else:
            val_files = read_val_list(val_list)
        val_indices = [i for i, npz_file in enumerate(npz_list) if npz_file in val_files]
    else:
        val_indices = sorted(numpy.random.choice(len(len_list), min(val_num, len(len_list)), replace=False))

    # 写入 train.txt 和 valid.txt
    with open(target_dir / "train.txt", "w", encoding="utf8") as f:
        for npz_file in npz_list:
            f.write(str(npz_file) + "\n")

    with open(target_dir / "valid.txt", "w", encoding="utf8") as f:
        for i in val_indices:
            f.write(npz_list[i] + "\n")

    # 保存所有样本的长度（用于训练时排序或采样）
    numpy.save(target_dir / "lengths.npy", len_list)

    with open(target_dir / "metadata.json", "w", encoding="utf8") as f:
        json.dump(metadata, f, indent=2)

    # 打印统计信息
    print(f"Processed {len(npz_list)} valid samples, {len(messages)} messages (skips/warnings).")
    print(f"Train.txt contains {len(npz_list)} entries, valid.txt contains {len(val_indices)} entries.")
    hours = total_duration / 3600
    minutes = (total_duration % 3600) / 60
    seconds = total_duration % 60
    print(f"Total audio duration: {total_duration:.2f} seconds ({hours:.0f}h {minutes:.0f}m {seconds:.2f}s)")


if __name__ == "__main__":
    preprocess()