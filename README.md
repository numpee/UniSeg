# UniSeg

## Setup 

### Installation

* Create and activate a new virtualenv (virtualenvwrapper) with Python3.7
```
mkvirtualenv NAME --python=python3.7
workon NAME
```

* Install PyTorch 1.2.0 for CUDA10
```
pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Install other packages
```
pip install -r requirements.txt
```

Note that `requirements.txt` also has `torch` and `torchvision`. If you're on `conda`, you may need to install `torch` and `torchvision` separately using `conda` and install all other packages. I have not tried installing with Anaconda, but if you have `torch` and `torchvision` installed correctly, the rest should install properly as well (I think). 

### Download pretrained models

Pretrained models can be downloaded from [this Github repo by the authors of HRNet](https://github.com/HRNet/HRNet-Image-Classification). This code provides two model settings: HRNet-W18 (V2) and HRNet-W48 (V2).
Under `configs/model` there are two `.yaml` files, which define the parameter `PRETRAIN_PATH`. Set this path to point
 to the downloaded pretrained models. 

## Data Preparation
Please refer to [the dataset folder](./datasets) for details on data preparation.


## Training

`main.py` can be run for training. I used distributed training (PyTorch's `DistributedDataParallel`). Some example
 scripts are provided in `example_scripts.sh`. Below are some explanations to the settings.

### Options

I'm using Hydra to set up configs. The default configs for `main.py` are listed in `configs/config.yaml`. 

To override certain settings, you can follow the example shown below. \
For example, let's say you want to train with the following settings:
 * `hrnet_w18` (small HRNet)
 * `CIBM` dataset (City+IDD+BDD+Mapillary) with CE Loss. This corresponds to `unified` in the configs
 * `learning_rate` of 0.05
 * `batch_size` of 16
 * And name the experiment `test_training` 

```
python main.py method=unified model=hrnet_w18 lr=0.05 batch_size=16 experiment_name=test_training
```

* Note that there is no `--` as in Python's native argparse
* You can check the specific settings for `method` and `model` under the corresponding folders in `configs/`
    * Basically, calling `model=hrnet_w18` will retrieve all necessary settings for HRNet W18 model.
    * Similarly, calling `method=unified` will retrieve all necessary settings for the unified setting (current CIBM dataset) with CE Loss training.
* There are more fine-grained settings in each of the config files that you can tune. For example, some parameters such as `lr`, `batch_size`, and `num_workers` are all defined under `config.yaml`.
* Running this code will use the maximum amount of GPUs currently available and automatically find a free port. Currently, I have only tried with single-machine, multi-node setting (no multi-machine training yet). If you do not want to use all the GPU resources available, make sure to set `CUDA_VISIBLE_DEVICES=...` before running the code (or if you're on SLURM, make the correct resource settings)
* **I've provided scripts for the 3 settings: 1) Single dataset training, 2) Unified training (CE Loss), 3) BCE + Null
 setting**
* For `batch_size`, I found that using 2 samples per GPU will eat up around 9GB of VRAM (HRNetW48). So given 8 GPUs
, I can usually fit batch size of 16. For HRNetW18, I can use a batch size of 32.
 
### Logging
#### Weights and Biases
Usually, all logging is done on Weights & Biases. For this code, I have W&B off by default. However, I
 have not fully tested the functionality of logging without W&B. **If you want to use W&B with your own account, you can follow the steps below:**
1. Add a new `.yaml` file under `configs/logging`, just like the one that I've made - `configs/logging/dongwan`
2. Set the `entity` (your username) and `project_name` (your project name) to your own values
3. In `config.yaml`, change the default `logging` value to your `.yaml` file. Currently it is set to `no_wandb`

Otherwise, you can disable W&B logging by setting the `logging` parameter when running the code. I have added an
 example of a simple logging function (non-W&B) in `loggers/local_loggers.py`. Follow this format for custom loggers
 . If you want to disable W&B, you may create a new logging config and set the logging parameter to that file, or use
  the logging file I've provided under `configs/logging/no_wandb.yaml`

#### Logging Path
The logging path is defined by `experiment_dir` in the parameter files that are in `configs/logging`.

#### Model saving
Even without W&B Logging, models and checkpoints will still be saved. Currently, I save 2 types of models:
1. The best model for each type of validation dataset metric - saved under `$EXPORT_ROOT/$VALIDATION_METRIC_best.pth`
2. The most recent model - saved under `$EXPORT_ROOT/recent.pth`

Note that the export root can also be defined in your custom logging config.

## Evaluation


## Other

* PILLOW-SIMD is used instead of PIL since it's faster. 
