## Table of Contents

- [New Features](#features)
- [Getting Started](#getting-started)
- [Quick Runing](#quick-runing)
- [Data Requirements](#data-requirements)

## New Features 🎉

- **Mixed data training**: Supporting simultaneous training with multiple data types: Pretrain, Instruction training, Conversation training.
- **Mixed peft training**: Supporting simultaneous training with multiple peft types: lora tuning, full param tuning.
- **Voice2Voice training**: Supporting end-to-end Voice2Voice training [Llama, Gemma].
- **8 bit training**: Supporting 8bit LLM training by xxx.

## Getting Started 🛞

### Conda Environment Construction

> New Environment.

```bash
conda create -n feb_platform python=3.11
conda activate feb_platform
conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc
conda config --add channels nvidia
conda install -c nvidia cuda-toolkit=12.1
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

LLM training method can refer to git: [training_scripts](https://github.com/feb-co/training_scripts).

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

### create your `dataset_info.json`

First, you need to create a `dataset_info.json` in your dataser dir, which dir includes all dataset that you want to use to traning. the following example shows the struction of `dataset_info.json`:

```json
{
    "ray": {
        "file_name": "ray_datas.jsonl",
        "stage": "conversation",
        "formatting": "sharegpt",
        "samples_ratio": 1.0,
        "columns": {
            "system": "You are Ray Dalio."
        }
    },
    "system": {
        "file_name": "system_datas.jsonl",
        "stage": "instruction",
        "formatting": "sharegpt",
        "samples_ratio": 1.0,
        "columns": {
            "system": "You are a helpful assistant."
        }
    },
    "pretrain": {
        "file_name": "pretrain_2048.jsonl",
        "stage": "pretrain",
        "formatting": "document",
        "samples_ratio": 1.0
    }
}
```

where, the key name (ray, system, pretrain) represent the data type that you want to use, the above inculde three types, every type is a isolate dataset, which will process independently by llama-factory-feb.

The meaning of each field name in dict is as follows:

- **file_name**: the `file_name` represent a releative dataset path that need to be trained, which can be a file or a directory.
- **stage**: the `stage` represent how to use this dataset to training, which will affect the method of data packing, now we support 3 type:
  - *conversation*:  `conversation` will pack multi dataset into one sequence untill it meet the max length of training for one example.
  - *instruction*: `instruction` will not pack dataset, if one data less than the max length, it will pading the `pad_token` token in tokenizer, untill it meet the max length of training for one example.
  - *pretrain*: `pretrain` will packing pretrain dataset to max length, but the difference with `conversation` is pretrain dataset will use all tokens in data to training.
- **formatting**: the `formatting` represent the data format that need to process, now we support 2 type:
  - *sharegpt*: this file should be `.jsonl` format, and every data in file should be `{"id": xxx, "conversation": [{"from": "xxx", "value": "xxx"}]}`
  - *document*: this file should be `.jsonl` format, and every data in file should be `{"prefix": xxx, "document": ["value1", "value2"]}`
- **samples_ratio**: the `samples_ratio` represent the dataset up/down sample ratio in one epoch.
- **columns**: the `columns` represent extra infomation for `sharegpt` formating, if you want to explore more detail info, you can read the code: `LLaMA-Factory-Feb/src/llamafactory/data/parser_feb.py`
