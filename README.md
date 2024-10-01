# Self-Training Large Language and Vision Assistant for Medical
<em> The advancement of medical image understanding and reasoning critically depends on building high-quality visual instruction data, which is costly and labor-intensive to obtain, particularly in the medical domain. To mitigate this data-starving issue, we introduce <strong>S</strong>elf-<strong>T</strong>raining <strong>L</strong>arge <strong>L</strong>anguage <strong>a</strong>nd <strong>V</strong>ision <strong>A</strong>ssistant for <strong>Med</strong>icine (STLLaVA-Med).</em>

<strong> Self-Training Large Language and Vision Assistant for Medical Question-Answering </strong> [[paper]()]

[Guohao Sun](https://heliossun.github.io/), [Can Qin](https://canqin.tech/), [Huazhu Fu](https://hzfu.github.io/), [Linwei Wang](https://www.rit.edu/directory/lxwast-linwei-wang), [Zhiqiang Tao](https://ztao.cc/)

<p align="center">
  <img src="./images/cover.png" width="500px"> <br>
  Medical data usage and performance comparision between LLaVA-Med and our method.
</p>

<p align="center">
  <img src="./images/pipeline.png" width="500px"> <br>
  Self-training pipeline for transforming a general Vision-Language assistant to medical expert.
</p>

## 🔥 News
* **`2024.09.20`** We will release our checkpoints soon!
* **`2024.09.20`** 🌟 Our paper has been accepted by EMNLP 2024 (main conference).
* **`2024.06.10`** 🌟 Our paper and code was released!

## Contents
- [Install](#install)
- [Dataset](#data)
- [Model Zoo](./docs/MODEL_ZOO.md)
- [Train](#training)
- [Evaluation](#evaluation)

## Install

1. Install Package
```Shell
conda create -n stllava python=3.10 -y
conda activate stllava
pip install --upgrade pip  # enable PEP 660 support
cd STLLaVA-Med
pip install -e .
```

2. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```



## Data

<strong>Visual instructional data</strong>

This project utilizes vision instructional data provided by [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) `60k_inline_mention`. However, due to disabled image URL, we fillterd out the origional data to ours own version in this project.

<strong>DPO data</strong>

<p align="center">
  <img src="./images/preference_data.png" width="500px"> <br>
  DPO data example.
</p>

This project auto-generate the preference dataset using the model itself and guided by GPT-4o. We sample 10k medical images from PMC-15M. You may download the dataset via [STLLaVA-Med-DPO](https://huggingface.co/datasets/ZachSun/STLLaVA-Med-DPO).

## Traininig
Training consists of two stages: (1) visual self-questioning instruction tuning stage, teaching the model to ask questions and follow multimodal instructions; (2) preference optimization.

### Instruction tuning:
Training script with DeepSpeed ZeRO-3 and lora: [`sqllava_med.sh`](https://github.com/heliossun/STLLaVA-Med/blob/main/sqllava_med.sh).

- `--mm_projector_type cluster`: the prototype extractor & a two-layer MLP vision-language connector.
- `--mm_projector_type mlp2x_gelu`: a two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--version v1_sq`: training for visual self-questioning.
- `--vit_lora_enable`: optimize vision encoder using vit lora. 

### Preference optimization:
Training script with DeepSpeed ZeRO-3 and lora: [`dpo_finetune.sh`](https://github.com/heliossun/STLLaVA-Med/blob/main/dpo_finetune.sh).

- `--version v1`: training for visual self-questioning.

## Evaluation
Please download raw images of datasets (VQA-RAD, SLAKE, PVQA) for medical VQA tasks.

Evaluate models on a diverse set of 3 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.



## Acknowledgement
- [SQ-LLaVA](https://arxiv.org/pdf/2403.11299.pdf): the codebase we built upon.


