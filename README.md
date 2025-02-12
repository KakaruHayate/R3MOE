# GeneralCurveEstimator

A general tool for estimating 1D curve features from mel-spectrograms

## Installation

Install PyTorch following the official instructions: https://pytorch.org/get-started/locally/, then run

```bash
pip install -r requirements.txt
```

## Preprocessing

### 1. Mouth Opening

1. Collect data using [LipsSync](https://github.com/KCKT0112/LipsSync), export and extract data. Folder structure is like
    ```text
    2025-02-04_22-01-52/
        audio.wav
        mouth_data.csv
    2025-02-04_22-43-56/
        audio.wav
        mouth_data.csv
    ```
2. Run preprocessing command
    ```bash
    python recipes/mouth_opening/preprocessing.py <SOURCE_DATA_DIR> <TARGET_DATA_DIR>
    ```
    where `<SOURCE_DATA_DIR>` is the directory containing the raw data, and `<TARGET_DATA_DIR>` is the directory to save the preprocessed data.

## Training

Run training command

```bash
python train.py --exp_name <EXPERIMENT_NAME> --dataset <DATASET_PATH> --gpu <GPU_ID>
```
where `<EXPERIMENT_NAME>` is the name of the experiment, `<DATASET_PATH>` is the path to the preprocessed dataset, and `<GPU_ID>` is the ID of the GPU to use. Other arguments can be found by running `python train.py --help`.


## Inference

TBD

## Acknowledgements

Most of the training code is adopted from [vocal-remover](https://github.com/tsurumeso/vocal-remover). The backbone model is borrowed from [FCPE](https://github.com/CNChTu/FCPE).
