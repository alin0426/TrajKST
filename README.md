## Introduction

This repository contains the source code for the paper:

**Urban Knowledge-enhanced Spatial-Temporal LLMs for Trajectory-oriented Multitask Learning**

In this work, we propose **TrajKST-LLM**, a method for **trajectory-oriented task**.

## Framework

<img src="images/framework.png" width="700">

The original PDF version is available [here](images/framework.pdf).

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
[Download](https://drive.google.com/file/d/1irZwvrUg7O9jBc2tRCqPmzGC3YHYvfnT/view?usp=sharing) and unzip into ./dataset/ to be finished

Pretrain data preparation:
```bash
python cached_data.py   --dataset_path ./dataset/   --city porto  
python cached_token_ids.py  --dataset_path ./dataset/   --city porto 
```

Test data preparation:
```bash
python cached_data.py   --dataset_path ./dataset/   --city porto   --test_cache
python cached_token_ids_test.py  --dataset_path ./dataset/   --city porto
```

## Pretrain
```bash
python pretrain.py   --task_name pretrain   --use_gpu   --dataset_path ./dataset/   --city porto   --mask_rate 0.3   --batch_size 32   --learning_rate 2e-4   --train_epochs 20   --device 0
```

## Finetune
```bash
python finetune.py   --use_gpu   --dataset_path ./dataset/   --city porto   --batch_size 32   --learning_rate 5e-5   --train_epochs 20   --device 0
```
can add parameters:
```bash
--ckpt ./checkpoints/porto_pretrain_best.pth
--checkpoint_path  ./checkpoints/
--log_path ./log_finetune 
```

## Evaluation
```bash
python evaluate.py  --use_gpu   --dataset_path ./dataset/   --city porto  --device 0  --test_cache
```


