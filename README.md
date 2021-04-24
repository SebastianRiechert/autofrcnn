# autofrcnn
automated training of faster rcnn for iterative creation of segmentation datasets

## Programming language
- Python 3.7.7

## Hardware and OS
The code has been developed and tested with linux on the IBM Power9 CPU architecture (ppc64le) using NVIDIA Tesla V100 GPUs.
This version of the code requires only a single NVIDIA Tesla V100 GPU, 4 CPU cores and 6gb of RAM to run.

## Dependencies
All dependencies are listed in `environment.yml`

## Installation
We recommend using a virtual conda environment for installation. Installation takes 5 minutes:
```
conda env create -f environment.yml
```
Activate the environment using:
```
conda activate autofrcnn
```
Next, navigate to `cocoapi/PythonAPI` and run:
```
python setup.py build_ext install
```

## Data
The images required for training are released under https://www.kaggle.com/sebastianriechert/bone-marrow-slides-for-leukemia-prediction

## Training
Train a single model using default parameters. Training on 10 images for 30 epochs takes 7 minutes on the specified hardware.
```
python auto_train.py --data-path datasets/small_data.csv --imgs-path kaggle/wsi
```
- `--imgs-path` must point to the folder containing the images from the dataset.
- `--data-path` must point to the csv containing the image to label (bbox) mappings

Display configurable training parameters:
```
python auto_train.py -h
```
Hyperparametersearch:
- create experiment:
```
python create_experiment_study.py --name demo
```
- launch hyperparametersearch:
```
python hypersweep.py --data-path datasets/small_data.csv  --imgs-path kaggle/wsi --experiment-name demo
```
The hyperparameter-space to search is defined and can be changed in `hypersweep.py`

To view training metrics, launch the MLflow-UI:
```
mlflow ui
```
Training metrics also print to stdout during training