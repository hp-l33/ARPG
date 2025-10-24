<!-- # ARPG: Autoregressive Image Generation with Randomized Parallel Decoding
 -->
<div align ="center">
<h1>Autoregressive Image Generation with Randomized Parallel Decoding</h3>

[Haopeng Li](https://github.com/hp-l33)<sup>1</sup>, Jinyue Yang<sup>2</sup>, [Guoqi Li](https://casialiguoqi.github.io)<sup>2,📧</sup>, [Huan Wang](https://huanwang.tech)<sup>1,📧</sup>

<sup>1</sup> Westlake University,
<sup>2</sup> Institute of Automation, Chinese Academy of Sciences


[![arXiv](https://img.shields.io/badge/arXiv-2503.10568-A42C25?style=flat&logo=arXiv)](https://arxiv.org/abs/2503.10568) [![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://hp-l33.github.io/projects/arpg) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat&logo=HuggingFace)](https://huggingface.co/hp-l33/ARPG)

</div>

<p align="center">
<img src="assets/title.jpg" width=95%>
<p>

## 🔥 News
* **2025-03-27**: Add HuggingFace integration to ARPG.
* **2025-03-25**: Add the sampling arccos schedule.
* **2025-03-14**: The paper and code are released!


## Introduction
We introduce a novel autoregressive image generation framework named **ARPG**. This framework is capable of conducting **BERT-style masked modeling** by employing a **GPT-style causal architecture**. Consequently, it is able to generate images in parallel following a random token order and also provides support for the KV cache. 
* 💪 **ARPG** achieves an FID of **1.94**
* 🚀 **ARPG** delivers throughput **26 times faster** than [LlamaGen](https://github.com/FoundationVision/LlamaGen)—nearly matching [VAR](https://github.com/FoundationVision/VAR)
* ♻️ **ARPG** reducing memory consumption by over **75%** compared to [VAR](https://github.com/FoundationVision/VAR).
* 🔍 **ARPG** supports **zero-shot inference** (e.g., inpainting and outpainting).
* 🛠️ **ARPG** can be easily extended to **controllable generation**.


## Model Zoo
We provide the model weights pre-trained on ImageNet-1K 256*256.
| Model | Param | Schedule | CFG | Step | FID | IS | Weight |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ARPG-L | 320 M | cosine | 4.5 | 64 | 2.44 | 292 | [arpg_300m.pt](https://huggingface.co/hp-l33/ARPG/blob/main/arpg_300m.pt) |
| ARPG-XL | 719 M | cosine | 6.0 | 64 | 2.10 | 331 | [arpg_700m.pt](https://huggingface.co/hp-l33/ARPG/blob/main/arpg_700m.pt) |
| ARPG-XXL | 1.3 B | cosine | 7.5 | 64 | 1.94 | 340 | [arpg_1b.pt](https://huggingface.co/hp-l33/ARPG/blob/main/arpg_1b.pt) |


## 🎮 Quick Start
You can easily play ARPG using the HuggingFace ``DiffusionPipeline``.
```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("hp-l33/ARPG", custom_pipeline="hp-l33/ARPG")

class_labels = [207, 360, 388, 113, 355, 980, 323, 979]

generated_image = pipeline(
    model_type="ARPG-XL",       # choose from 'ARPG-L', 'ARPG-XL', or 'ARPG-XXL'
    seed=0,                     # set a seed for reproducibility
    num_steps=64,               # number of autoregressive steps
    class_labels=class_labels,  # provide valid ImageNet class labels
    cfg_scale=4,                # classifier-free guidance scale
    output_dir="./images",      # directory to save generated images
    cfg_schedule="constant",    # choose between 'constant' (suggested) and 'linear'
    sample_schedule="arccos",   # choose between 'arccos' (suggested) and 'cosine'
)

generated_image.show()
```
If you want to train or reproduce the results of ARPG, please refer to [Getting Started](GETTING_STARTED.md). 


## Citation
If this work is helpful for your research, please give it a star or cite it:
```bibtex
@article{li2025autoregressive,
    title={Autoregressive Image Generation with Randomized Parallel Decoding},
    author={Haopeng Li and Jinyue Yang and Guoqi Li and Huan Wang},
    journal={arXiv preprint arXiv:2503.10568},
    year={2025}
}
```

## Acknowledgement

Thanks to [LlamaGen](https://github.com/FoundationVision/LlamaGen) for its open-source codebase. Appreciate [RandAR](https://github.com/ziqipang/RandAR) and [RAR](https://github.com/bytedance/1d-tokenizer/blob/main/README_RAR.md) for inspiring this work, and also thank [ControlAR](https://github.com/hustvl/ControlAR).