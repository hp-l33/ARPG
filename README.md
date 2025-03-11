# 🎮 ARPG: Autoregressive Image Generation with Randomized Parallel Decoding

<p align="center">
<img src="assets/title.jpg" width=95%>
<p>

## News
* **[2025-03-xx]**: The paper and code are released!


## Introduction

## Getting Started

### Train
```shell
torchrun \
--nnodes=1 --nproc_per_node=8 train_c2i.py \
--gpt-model GPT-L \
--code-path YOUR_DATASET_PATH \
--epochs 400 \
--global-batch-size 1024 \
--lr 4e-4
```

### Evaluation
```shell
torchrun \
--nnodes=1 --nproc_per_node=8 sample_c2i_ddp.py \
--gpt-model GPT-XL \
--gpt-ckpt YOUR_CKPT_PATH \
--cfg-scale 6.0 \
--temperature 1.0 \
--top-k 0 \
--top-p 1.0 \
--step 64
```