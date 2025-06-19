# CropCon: Crop-Type Contrastive Learning for Semantic Segmentation

## 📚 Introduction


For the moment, we support the following **models**:

|             | Paper | GitHub | Keywords |
|:-----------:|:-----:|:------:|:--------:|
|  [U-TAE](https://arxiv.org/abs/2107.07933) | Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks    | [link](https://github.com/VSainteuf/utae-paps) | U-Net |


And the following **datasets**:

|                     | Download | Domain | Task | Sensors | Location |
|:-------------------:|:--------:|:------:|:----:|:-------:|:--------:|
|        [PASTIS-R](https://arxiv.org/abs/2107.07933)       |    [link](https://huggingface.co/datasets/IGNF/PASTIS-HD)       |   Agriculture     |  Semantic Segmentation    |    S1, S2, SPOT-6  | France   |

## 🛠️ Setup
Clone the repository:
```
git clone https://github.com/GioCastiglioni/CropCon
```

**Dependencies**

```
cd CropCon
conda env create CropCon python=3.11
conda activate CropCon
pip install -r requirements.txt
```

## 🏋️ Training

To run experiments, please refer to `configs/train.yaml`. In it, in addition to some basic info about training (e.g. `finetune` for fine-tuning the encoder, `limited_label_train` to train the model on a stratified subset of labels, `num_workers`, `batch_size` and so on), there are different configs. We provide examples of command lines to initialize a training task on a single GPU.

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
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT cropcon/run.py --config-name=train -m \
dataset=pastis \
dataset.multi_temporal=6 \
task.trainer.tau=0.1 \
task.trainer.alpha=0.0 \
encoder=utae_encoder \
decoder=seg_utae \
batch_size=8 \
test_batch_size=8 \
preprocessing=seg_resize \
criterion=cross_entropy \
optimizer=adamw \
optimizer.lr=0.001 \
ft_rate=1.0 \
task=segmentation \
finetune=True \
from_scratch=True \
lr_scheduler=multi_step_lr \
work_dir="/home/gcast/CropCon/results" \
use_wandb=True \
wandb_project="CropCon" \
num_workers=4 \
test_num_workers=4 \
limited_label_train=1.0 \
limited_label_strategy=stratified \
task.trainer.n_epochs=80
```
