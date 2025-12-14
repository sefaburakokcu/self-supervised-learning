# Self-Supervised Learning Framework

This repository provides a pipeline for training and evaluating self-supervised learning models on the STL-10 dataset.
The framework supports pretraining, linear probing, and fine-tuning workflows.

## Features

* SimCLR self-supervised pretraining
* ResNet and Vision Transformer backbones
* Evaluation on STL-10 using accuracy and F1-score
* Notebook support for easy experimentation
* Unified JSON-based experiment configuration system

## Requirements

* Python 3.10+
* PyTorch 2.5.1
* CUDA 12.4 (recommended)
* GPU with at least 8 GB memory

See `requirements.txt` for all dependencies.

## Quick Start

### Installation

```bash
git clone https://github.com/sefaburakokcu/self-supervised-learning.git
cd self-supervised-learning
pip install -r requirements.txt
```

### Running Experiments

List experiments:

```bash
python experiments/run_experiment.py --list
```

Run an experiment:

```bash
python experiments/run_experiment.py --experiment baselines/baseline_resnet18 --device cuda
```

SimCLR pretraining:

```bash
python experiments/run_experiment.py --experiment ssl_pretraining/simclr_resnet18
```

Run all experiments sequentially:

For convenience and full reproducibility, all experiments can be executed sequentially using the provided 
bash script. This is useful for running the complete experimental pipeline including baselines, 
SimCLR pretraining, linear probing, and fine-tuning.

```bash
bash scripts/run_experiemens.sh
```

### Experiment Types

* Baselines (supervised training)
* SSL pretraining (SimCLR)
* Linear probing
* Full fine-tuning
* Transfer learning (ImageNet initialization)

## Jupyter Notebook

`notebooks/experiments.ipynb` supports Colab execution, automated setup, GPU training, logging, and Google Drive backups.

## Directory Structure

```
self-supervised-learning/
├── experiments/
│   └── run_experiment.py
├── scripts/
│   └── run_experiments.sh
├── models/
├── data/
├── trainers/
├── utils/
├── configs/
├── data/
├── evaluators/
├── experiments/
├── notebooks/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Configuration

Experiment files in `configs/` define model architecture, optimizer settings, training parameters, and SimCLR-specific options.

## Monitoring and Outputs

Training logs and checkpoints are stored under `./logs/` and `./checkpoints/`.
Each experiment generates:

* Model checkpoints
* Training logs
* Configuration snapshots
* Final accuracy and F1-score

## Experimental Results

We evaluate the proposed self-supervised learning framework on the **STL-10** dataset under three settings: SimCLR pretraining, full supervised training, and few-shot learning. Results demonstrate the effectiveness of SimCLR pretraining, especially when combined with fine-tuning.

### SimCLR Pretraining (Unlabeled STL-10)

| Backbone  | Epochs | Batch Size | Training Time (min) | Final Loss |
| --------- | ------ | ---------- | ------------------- | ---------- |
| ResNet-18 | 200    | 256        | 138.6               | 0.2049     |
| ResNet-50 | 200    | 256        | 481.9               | 0.1632     |

### Full Supervised Evaluation (500 images/class)

| Backbone  | Strategy                 | Accuracy   |
| --------- | ------------------------ | ---------- |
| ResNet-18 | Baseline (Scratch)       | 75.32%     |
| ResNet-18 | SimCLR + Linear Probing  | 75.08%     |
| ResNet-18 | **SimCLR + Fine-Tuning** | **78.89%** |
| ResNet-50 | Baseline (Scratch)       | 58.24%     |
| ResNet-50 | SimCLR + Linear Probing  | 75.96%     |
| ResNet-50 | **SimCLR + Fine-Tuning** | **80.59%** |

### Few-Shot Learning (50 images/class, ResNet-18)

| Strategy                 | Accuracy   |
| ------------------------ | ---------- |
| Baseline (Scratch)       | 48.63%     |
| SimCLR + Linear Probing  | 66.34%     |
| **SimCLR + Fine-Tuning** | **69.24%** |

### Key Observations

* SimCLR pretraining significantly improves downstream performance, especially in **low-label (few-shot)** settings.
* Fine-tuning consistently outperforms linear probing.
* Deeper backbones (ResNet-50) benefit more from self-supervised pretraining compared to training from scratch.
