# Future-conditioned Unsupervised Pretraining for Decision Transformer

This repository is the official implementation of our ICML 2023 paper [Future-conditioned Unsupervised Pretraining for Decision Transformer](https://arxiv.org/abs/2305.16683).

## Requirements

To install requirements, run:

```setup
conda env create -f env.yml
conda activate pdt
```

## Data

The D4RL datasets can be downloaded via the following commands:

```
python data/download_d4rl_gym_datasets.py
```

## Training

To pretrain a model, run this command:

```
python main.py \
    --data_dir /path/to/data \
    --max_pretrain_iters 50 \
    --num_updates_per_pretrain_iter 1000 \
    --max_online_iters 0 \
    --env hopper-medium-replay-v2 \
    --seed 0
```

To finetune a pretrained model, run:

```
python main.py \
    --data_dir /path/to/data \
    --model_path_prefix /path/to/model \
    --model_name model \
    --max_pretrain_iters 0 \
    --online_warmup_samples 10000 \
    --return_warmup_iters 5 \
    --max_online_iters 1500 \
    --num_updates_per_online_iter 300 \
    --env hopper-medium-replay-v2 \
    --seed 0
```

## Evaluation

Run the following script to evaluate a model:

```
python main.py \
    --eval_only \
    --eval_pretrained \
    --data_dir /path/to/data \
    --model_path_prefix /path/to/model \
    --model_name model \
    --env hopper-medium-replay-v2 \
    --seed 0
```

Besides, you can also monitor training with Tensorboard:

```
tensorboard --logdir /path/to/res
```

## Acknowledgement
This repository is based on [online-dt](https://github.com/facebookresearch/online-dt), which is licensed under [CC-BY-NC](https://github.com/facebookresearch/online-dt/blob/main/LICENSE.md). We have made modifications to the models, data processing, and training/evaluation scripts to fit our needs.
