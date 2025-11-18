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
