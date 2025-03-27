## Getting Started
### Preparation
To accelerate the training process, we use the ImageNet dataset that has been pre-encoded into tokens, following the approach of [LlamaGen](https://github.com/FoundationVision/LlamaGen). You can directly download the pre-processed [dataset](https://huggingface.co/ziqipang/RandAR/blob/main/imagenet-llamagen-adm-256_codes.tar) provided by [RandAR](https://github.com/ziqipang/RandAR).

### Training
Taking ARPG-L as an example, the script for training using 8 A800-80GB GPUs is as follows:
```shell
torchrun \
--nnodes=1 --nproc_per_node=8 train_c2i.py \
--gpt-model ARPG-L \
--code-path YOUR_DATASET_PATH \
--epochs 400 \
--global-batch-size 1024 \
--lr 4e-4
```
Note that the learning rate is configured to be 1e-4 per 256 batch size. That is, if you set the batch size to 768, the lr should be adjusted to 3e-4.

### Evaluation
1. Prepare ADM evaluation script.
```shell
git clone https://github.com/openai/guided-diffusion.git

wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```
2. Download the [pre-trained weights](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt) of [LlamaGen](https://github.com/FoundationVision/LlamaGen)'s tokenizer.

  
3. Reproduce the experimental results of ARPG. 

```shell
# PS: arccos schedule outperforms the paper's cosine setting.
# ARPG-L. The FID should be close to 2.38.
torchrun \
--nnodes=1 --nproc_per_node=8 sample_c2i_ddp.py \
--gpt-model ARPG-L \
--gpt-ckpt arpg_300m.pt \
--vq-ckpt vq_ds16_c2i.pt \
--sample-schedule arccos \
--cfg-scale 5.0 \
--step 64
```
```shell
# ARPG-XL. The FID should be close to 2.02.
torchrun \
--nnodes=1 --nproc_per_node=8 sample_c2i_ddp.py \
--gpt-model ARPG-XL \
--gpt-ckpt arpg_700m.pt \
--vq-ckpt vq_ds16_c2i.pt \
--sample-schedule arccos \
--cfg-scale 6.0 \
--step 64
```
```shell
# ARPG-XXL. The FID should be close to 1.91.
torchrun \
--nnodes=1 --nproc_per_node=8 sample_c2i_ddp.py \
--gpt-model ARPG-XXL \
--gpt-ckpt arpg_1b.pt \
--vq-ckpt vq_ds16_c2i.pt \
--sample-schedule arccos \
--cfg-scale 7.5 \
--step 64
```
Note that the unlisted parameters (such as temperature, top-k, etc.) are all the default values set in `sample_c2i_ddp.py`.
