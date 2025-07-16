import re
import json
import pathlib
import sys

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
CORRECTED_JAW_OPEN = 2  # jawOpen * (1 - mouthClose)
SUBTRACTED_JAW_OPEN = 3  # jawOpen - mouthClose
LIPS_DISTANCE = 4


@click.command()
@click.argument('source_dir', type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument('target_dir', type=click.Path(path_type=pathlib.Path))
@click.option(
    '--val_list', type=click.Path(exists=True, path_type=pathlib.Path),
    help='Validation file list in a text file. Use relative path to the source directory, one file per line.'
)
@click.option(
    '--val_num', default=5, type=int,
    help='Number of validation samples for random selection if val_list is not provided.'
)
@click.option(
    '--attr_type', default=SUBTRACTED_JAW_OPEN, type=int,
    help=(
            'Attribute type for processing '
            '[0: jawOpen, 1: mouthClose, 2: jawOpen * (1 - mouthClose), '
            '3: jawOpen - mouthClose), 4. LipsDistance].')
)
@click.option(
    '--subtraction_offset', default=0.05, type=float,
    help='Offset for subtracted jawOpen attribute (X = jawOpen - mouthClose + offset).'
)
@click.option(
    "--use_mask", is_flag=True,
    help="Use Aegisub mask file or VAD model to mask non-vocal parts."
)
# @click.option('--use_noise', is_flag=True, help='Replace masked regions with Gaussian noise instead of zero.')
# @click.option('--noise_eps', default=0.001, type=float, help='Noise epsilon.')
@click.option('--sample_rate', default=16000, type=int, help='Sample rate for audio processing.')
@click.option('--mel_bins', default=80, type=int, help='Number of mel bins for spectrogram.')
@click.option('--hop_size', default=320, type=int, help='Hop size for spectrogram.')
@click.option('--win_size', default=1024, type=int, help='Window size for spectrogram.')
@click.option('--f_min', default=0, type=int, help='Minimum frequency for spectrogram.')
@click.option('--f_max', default=None, type=int, help='Maximum frequency for spectrogram.')
def preprocess(
        source_dir: pathlib.Path,
        target_dir: pathlib.Path,
        val_list: pathlib.Path,
        val_num: int,
        attr_type: int,
        subtraction_offset: float,
        use_mask: bool,
        # use_noise: bool,
        # noise_eps: float,
        sample_rate: int,
        mel_bins: int,
        hop_size: int,
        win_size: int,
        f_min: int,
        f_max: int
):
    metadata = {
        "sample_rate": sample_rate,
        "mel_bins": mel_bins,
        "hop_size": hop_size,
        "win_size": win_size,
        "f_min": f_min,
        "f_max": f_max
    }
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
    vad = None

    with tqdm.tqdm(csv_list) as bar:
        for csv_file in bar:
            # read mouth opening data
            df = pandas.read_csv(csv_file)
            xs = df["TimeStamp"].values
            if attr_type == JAW_OPEN:
                ys = df["jawOpen"].values
            elif attr_type == MOUTH_CLOSE:
                ys = df["mouthClose"].values
            elif attr_type == CORRECTED_JAW_OPEN:
                ys = df["jawOpen"].values * (1 - df["mouthClose"].values)
            elif attr_type == SUBTRACTED_JAW_OPEN:
                ys = numpy.clip(df["jawOpen"].values - df["mouthClose"].values + subtraction_offset, a_min=0, a_max=1)
            elif attr_type == LIPS_DISTANCE:
                ys = df["LipsDistance"].values
            else:
                raise ValueError(f"Invalid attr_type: {attr_type}")
            if len(ys) < 2:
                bar.write(f"Warning: empty data in {csv_file}")
                continue
            offset = round(sample_rate * xs[0])
            size = round(sample_rate * (xs[-1] - xs[0]))
            xs -= xs[0]
            error_value_segment = process_error_value(xs, ys)
            # read audio data
            num_samples = None
            for audio_file in csv_file.parent.glob("*.wav"):
                audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
                if num_samples is None:
                    num_samples = len(audio)
                elif num_samples != len(audio):
                    bar.write(f"Warning: mismatched audio length in {audio_file}")
                    continue
                if len(audio) < offset + size:
                    audio = numpy.pad(audio, (0, offset + size - len(audio)))
                audio = audio[offset: offset + size]
                # log mel spectrogram
                mel = dynamic_range_compression_torch(mel_spec_transform(
                    torch.from_numpy(audio)[None]
                ), clip_val=1e-9)[0].T.cpu().numpy()
                # interpolate mouth opening data
                interp_fn = interp1d(xs, ys, kind="linear", fill_value="extrapolate")
                t_mel = numpy.linspace(0, len(audio) / sample_rate, mel.shape[0])
                curve = numpy.ndarray.astype(interp_fn(t_mel), numpy.float32)
                if use_mask:
                    ass_path = pathlib.Path(audio_file).with_name("mask.ass")
                    is_ass = ass_path.exists()
                    if is_ass:
                        # mask non-vocal parts with mask file
                        segments = ass_to_time_array(ass_path)
                        time_scale = 1
                    else:
                        # mask non-vocal parts using VAD
                        if vad is None:
                            from funasr import AutoModel
                            vad = AutoModel(
                                model="fsmn-vad", model_revision="v2.0.4", disable_update=True,
                                log_level="ERROR", disable_pbar=True, disable_log=True
                            )
                        segments = vad.generate(audio_file.as_posix())[0]["value"]
                        time_scale = 1 / 1000
                    mask = numpy.zeros_like(curve)
                    for start_ms, end_ms in segments:
                        start = min(round(start_ms * time_scale * sample_rate / hop_size), mask.shape[0])
                        end = min(round(end_ms * time_scale * sample_rate / hop_size), mask.shape[0])
                        mask[start: end] = 1
                    if is_ass:
                        mask = 1 - mask
                    curve *= mask
                # error value process
                frame_indices = []
                for error_start, error_end in error_value_segment:
                    error_start = round(error_start * sample_rate / hop_size)
                    error_end = round(error_end * sample_rate / hop_size)
                    frame_indices.extend(range(error_start, error_end + 1))
                mel = numpy.delete(mel, frame_indices, axis=0)
                curve = numpy.delete(curve, frame_indices, axis=0)
                # save npz
                target_file = target_dir / audio_file.relative_to(source_dir).with_suffix(".npz")
                target_file.parent.mkdir(parents=True, exist_ok=True)
                numpy.savez(target_file, spectrogram=mel, curve=curve)
                len_list.append(mel.shape[0])
                npz_list.append(target_file.relative_to(target_dir).as_posix())
    # split training and validation set
    if val_list is not None:
        with open(val_list, "r", encoding="utf8") as f:
            val_files = {pathlib.Path(line.strip()).with_suffix(".npz").as_posix() for line in f}
        val_indices = []
        for i, npz_file in enumerate(npz_list):
            if npz_file in val_files:
                val_indices.append(i)
    else:
        val_indices = sorted(numpy.random.choice(len(len_list), val_num, replace=False))
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


def process_error_value(timestamps, values):
    error_value_segment = []
    first_error_value = values[0]
    start_of_error = timestamps[0]
    for i in range(1, len(values)):
        if values[i] != first_error_value:
            error_value_segment.append((start_of_error, timestamps[i]))
            break

    return error_value_segment


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
