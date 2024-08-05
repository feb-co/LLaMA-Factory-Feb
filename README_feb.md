![# LLaMA Factory](assets/logo.png)

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Factory)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
[![Citation](https://img.shields.io/badge/citation-72-green)](#projects-using-llama-factory)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
[![Open in DSW](https://gallery.pai-ml.com/assets/open-in-dsw.svg)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
[![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
[![Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)

[![GitHub Tread](https://trendshift.io/api/badge/repositories/4535)](https://trendshift.io/repositories/4535)

👋 Join our [WeChat](assets/wechat.jpg) or [NPU user group](assets/wechat_npu.jpg).

\[ English | [中文](README_zh.md)| [FEB](README_feb.md) \]

**Fine-tuning a large language model can be easy as...**

https://github.com/user-attachments/assets/7c96b465-9df7-45f4-8053-bf03e58386d3

Choose your path:

- **Colab**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
- **PAI-DSW**: https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory
- **Local machine**: Please refer to [usage](#getting-started)

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Quick Runing](#quick-runing)
- [Data Requirements](#data-requirements)

## Features 🎉

- **Various models**: LLaMA, LLaVA, Mistral, Mixtral-MoE, Qwen, Yi, Gemma, Baichuan, ChatGLM, Phi, etc.
- **Integrated methods**: (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, etc.
- **Scalable resources**: 16-bit full-tuning, freeze-tuning, LoRA and 2/3/4/5/6/8-bit QLoRA via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ.
- **Advanced algorithms**: GaLore, BAdam, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ, PiSSA and Agent tuning.
- **Practical tricks**: FlashAttention-2, Unsloth, RoPE scaling, NEFTune and rsLoRA.
- **Experiment monitors**: LlamaBoard, TensorBoard, Wandb, MLflow, etc.
- **Faster inference**: OpenAI-style API, Gradio UI and CLI with vLLM worker.

## Getting Started 🛞

### Conda Environment Construction

> New Environment.

```bash
conda create -n feb_platform python=3.11
conda activate feb_platform
conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc
```

### Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory-Feb
pip install -e ".[torch,metrics]"
```

> DeepSpeed + Flash-attn

```bash
pip install deepspeed
pip install flash-attn --no-build-isolation
```

After you have installed deepspeed, you can check the installation results by running `ds_report`.

Extra dependencies available: torch, torch-npu, metrics, deepspeed, bitsandbytes, hqq, eetq, gptq, awq, aqlm, vllm, galore, badam, qwen, modelscope, quality

> [!TIP]
> Use `pip install --no-deps -e .` to resolve package conflicts.

## Quick Runing 🚀

### Runing by Cli

LLM training method can refer to git: xxx.

### Runing by Web

> Modify the code to replace the address of the downloaded model

```bash
python src/webui.py \
--model_name_or_path /mnt/ceph/huggingface_model_hub/Meta-Llama-3-8B-Instruct \
--template llama3 \
--infer_backend vllm \
--vllm_enforce_eager
```

## Data Requirements 📊

