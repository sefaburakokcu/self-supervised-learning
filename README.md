# Self Supervised Learning

This repository contains implementations and experiments related to self-supervised learning methods for computer vision. 
It includes pretext tasks, architectures, and evaluation protocols. It also provides 
fine-tuning and linear probing experiments on STL-10 dataset.

## Experiments

### List all experiments
python experiments/run_experiment.py --list

### List all groups
python experiments/run_experiment.py --list-groups

### Run single experiment
python experiments/run_experiment.py --experiment simclr_resnet18 --use-wandb

### Run experiment group (all baselines)
python experiments/run_experiment.py --group all_baselines --use-wandb

### Run complete pipeline
python experiments/run_experiment.py --group complete_pipeline --use-wandb

### Run specific experiment with custom device
python experiments/run_experiment.py --experiment mae_vit_small --device cuda --use-wandb

### Run linear probing experiments
python experiments/run_experiment.py --group all_linear_probing

### Run all fine-tuning experiments
python experiments/run_experiment.py --group all_full_finetune
