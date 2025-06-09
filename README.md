[![Tests](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/yurujaja/geofm-bench/actions/workflows/python-test.yml)

# CropCon: Crop-Type Contrastive Learning for Semantic Segmentation

## 📚 Introduction


For the moment, we support the following **models**:

|             | Paper | GitHub | Keywords |
|:-----------:|:-----:|:------:|:--------:|
|  [U-TAE](https://arxiv.org/abs/2107.07933) | Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks    | [link](https://github.com/VSainteuf/utae-paps) | U-Net |


And the following **datasets**:

|                     | Download | Domain | Task | Sensors | Location |
|:-------------------:|:--------:|:------:|:----:|:-------:|:--------:|
|        [PASTIS-R](https://arxiv.org/abs/2404.08351)       |    [link](https://huggingface.co/datasets/IGNF/PASTIS-HD)       |   Agriculture     |  Semantic Segmentation    |    S1, S2, SPOT-6  | France   |

The repository supports the following **tasks** using geospatial (foundation) models:
 - [Multi-Temporal Semantic Segmentation](#multi-temporal-semantic-segmentation)

## 🛠️ Setup
Clone the repository:
```
git clone https://github.com/GioCastiglioni/CropCon
cd CropCon
```

**Dependencies**

```
conda env create CropCon python=3.11
conda activate CropCon
pip install -r requirements.txt
```

## 🏋️ Training

To run experiments, please refer to `configs/train.yaml`. In it, in addition to some basic info about training (e.g. `finetune` for fine-tuning also the encoder, `limited_label_train` to train the model on a stratified subset of labels, `num_workers`, `batch_size` and so on), there are 5 different basic configs:
- `dataset`: Information of downstream datasets such as image size, band_statistics, classes etc.
- `decoder`: Downstream task decoder fine-tuning related parameters, like the type of architecture (e.g. UPerNet), which multi-temporal strategy to use, and other related hparams (e.g. nr of channels)
- `encoder`: GFM encoder related parameters. `output_layers` is used for which layers are used for Upernet decoder.  
- `preprocessing`: Both preprocessing and augmentations steps required for the dataset, such as bands adaptation, normalization, resize/crop.


Other 3 configs are used to set other training parameters:
- `criterion`: in which you can choose the loss for the training. Consider that if you want to add a custom loss, you should add to `pangaea/utils/losses.py`. Currently, we support `cross_entropy`, `weigthed_cross_entropy`, `dice` and `mae` loss functions.
- `lr_scheduler`: in which you can choose the scheduler. Consider that if you want to add a custom one, you should add to `pangaea/utils/schedulers.py`. 
- `optimizer`: in which you can choose the optimizer. Consider that if you want to add a custom one, you should add to `pangaea/utils/optimizers.py`.


We provide examples of command lines to initialize different training tasks on single GPU.

Please note:
 - The repo adopts [hydra](https://github.com/facebookresearch/hydra), so you can easily log your experiments and overwrite parameters from the command line. More examples are provided later.
 - To use more gpus or nodes, set `--nnodes` and `--nproc_per_node` correspondingly. Please refer to the [torchrun doc](https://pytorch.org/docs/stable/elastic/run.html).

#### Single Temporal Semantic Segmentation
```
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate CropCon
export PYTHONPATH=/home/<USERNAME>/CropCon:$PYTHONPATH
cd /home/<USERNAME>/CropCon
```
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py --config-name=train -m \
dataset=pastis \
dataset.multi_temporal=35 \
encoder=utae_encoder \
decoder=seg_utae \
batch_size=12 \
test_batch_size=12 \
preprocessing=seg_resize \
criterion=cross_entropy \
optimizer=adamw \
optimizer.lr=0.0015 \
ft_rate=1.0 \
task=segmentation \
finetune=True \
from_scratch=True \
lr_scheduler=multi_step_lr \
work_dir="/home/<USERNAME>/CropCon/results" \
use_wandb=True \
wandb_project="CropCon" \
num_workers=4 \
test_num_workers=4 \
limited_label_train=1.0 \
limited_label_strategy=stratified \
task.trainer.n_epochs=80
```
