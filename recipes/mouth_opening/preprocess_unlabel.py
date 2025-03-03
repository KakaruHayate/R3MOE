import json
import pathlib
import sys

import click
import librosa
import numpy
import torch
import tqdm

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())
from lib.transforms import PitchAdjustableMelSpectrogram, dynamic_range_compression_torch

@click.command()
@click.argument("source_dir", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("target_dir", type=click.Path(path_type=pathlib.Path))
@click.option(
    "--val_list", type=click.Path(exists=True, path_type=pathlib.Path),
    help="Validation file list in a text file. Use relative path to the source directory, one file per line."
)
@click.option(
    "--val_num", default=5, type=int,
    help="Number of validation samples for random selection if val_list is not provided."
)
@click.option("--sample_rate", default=16000, type=int, help="Sample rate for audio processing.")
@click.option("--mel_bins", default=80, type=int, help="Number of mel bins for spectrogram.")
@click.option("--hop_size", default=320, type=int, help="Hop size for spectrogram.")
@click.option("--win_size", default=1024, type=int, help="Window size for spectrogram.")
@click.option("--f_min", default=0, type=int, help="Minimum frequency for spectrogram.")
@click.option("--f_max", default=None, type=int, help="Maximum frequency for spectrogram.")
def preprocess(
        source_dir: pathlib.Path,
        target_dir: pathlib.Path,
        val_list: pathlib.Path,
        val_num: int,
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
    audio_list = sorted(file for ext in ["*.m4a", "*.wav", "*.mp3", "*.aac"] for file in source_dir.rglob(ext))
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

    with tqdm.tqdm(audio_list) as bar:
        for audio_file in bar:
            # read audio data
            audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
            # log mel spectrogram
            mel = dynamic_range_compression_torch(mel_spec_transform(
                torch.from_numpy(audio)[None]
            ), clip_val=1e-9)[0].T.cpu().numpy()
            # save npz
            target_file = target_dir / audio_file.relative_to(source_dir).with_suffix(".npz")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            numpy.savez(target_file, spectrogram=mel)
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
    with open(target_dir / "train.txt", "w", encoding="utf8") as f:
        for i, npz_file in enumerate(npz_list):
            f.write(str(npz_file) + "\n")
            lengths.append(len_list[i])
    with open(target_dir / "valid.txt", "w", encoding="utf8") as f:
        for i in val_indices:
            f.write(npz_list[i] + "\n")
    numpy.save(target_dir / "lengths.npy", lengths)
    with open(target_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    preprocess()
