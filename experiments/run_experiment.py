import torch
import wandb
import argparse
import sys
import json

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.stl10_dataset import STL10DataModule
from models.backbones import create_encoder
from models.ssl_methods.ssl_model_factory import SSLModelFactory
from trainers.hyperparameters_factory import OptimizerFactory, SchedulerFactory
from trainers.ssl_trainer import SSLTrainer
from trainers.finetune_trainer import FineTuneTrainer
from utils.config_manager import ConfigManager


class ExperimentRunner:
    """Simple experiment runner for single experiments"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        self.use_wandb = config.get('use_wandb', False)
        self.experiment_name = config.get('name', 'unnamed')

        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._save_config()

    def _save_config(self):
        """Save experiment configuration"""
        config_dir = self.log_dir / self.experiment_name
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved: {config_file}")

    def run_ssl_pretraining(self):
        """Run SimCLR SSL pretraining"""

        print(f"\n{'=' * 80}")
        print(f"SSL Pretraining: SimCLR + {self.config['arch']}")
        print(f"{'=' * 80}\n")

        if self.use_wandb:
            wandb.init(
                project='stl10-ssl',
                name=f"{self.experiment_name}_pretrain",
                config=self.config
            )

        data_module = STL10DataModule(
            batch_size=int(self.config['batch_size']),
            data_dir=self.config.get('data_dir', './datasets'),
            num_workers=self.config.get('num_workers', 4)
        )
        ssl_loader = data_module.get_ssl_dataloaders(ssl_method='simclr')

        encoder = create_encoder(self.config['arch'], pretrained=False)

        ssl_kwargs = {}
        for key in ['projection_dim', 'hidden_dim', 'temperature', 'momentum',
                    'decoder_dim', 'decoder_depth', 'mask_ratio']:
            if key in self.config:
                ssl_kwargs[key] = self.config[key]

        model = SSLModelFactory.create(
            self.config['ssl_method'],
            encoder,
            **ssl_kwargs
        )

        print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

        optimizer = OptimizerFactory.create(
            self.config.get('optimizer', 'adamw'),
            model.parameters(),
            self.config.get('optimizer_params', {})
        )

        scheduler_params = self.config.get('scheduler_params', {}).copy()
        if 'T_max' not in scheduler_params:
            scheduler_params['T_max'] = int(self.config['epochs'])

        scheduler = SchedulerFactory.create(
            self.config.get('scheduler', 'cosine'),
            optimizer,
            scheduler_params
        )

        experiment_dir = self.checkpoint_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        trainer = SSLTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            log_dir=str(experiment_dir),
            use_wandb=self.use_wandb
        )

        trainer.train(
            ssl_loader,
            num_epochs=int(self.config['epochs']),
            save_freq=self.config.get('save_freq', 20)
        )

        if self.use_wandb:
            wandb.finish()

        return str(experiment_dir / 'best_model.pth')

    def run_finetuning(self, pretrained_path=None):
        """Run fine-tuning"""

        print(f"\n{'=' * 80}")
        print(f"Fine-tuning: {self.config['arch']}")
        print(f"{'=' * 80}\n")

        if self.use_wandb:
            wandb.init(
                project='stl10-ssl',
                name=f"{self.experiment_name}_finetune",
                config=self.config
            )

        finetune_batch_size = int(self.config.get('batch_size', 64))
        data_module = STL10DataModule(
            batch_size=finetune_batch_size,
            data_dir=self.config.get('data_dir', './datasets'),
            num_workers=self.config.get('num_workers', 4)
        )
        train_loader, test_loader = data_module.get_supervised_dataloaders()

        if pretrained_path:
            print(f"Loading pretrained: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            encoder = create_encoder(self.config['arch'], pretrained=False)
            encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)

        elif self.config.get('use_imagenet', False):
            print("Loading ImageNet pretrained")
            encoder = create_encoder(self.config['arch'], pretrained=True)

        else:
            print("Training from scratch")
            encoder = create_encoder(self.config['arch'], pretrained=False)

        trainer = FineTuneTrainer(
            encoder=encoder,
            num_classes=10,
            freeze_encoder=self.config.get('linear_probe', False),
            device=self.device
        )

        params = list(trainer.classifier.parameters())
        if not self.config.get('linear_probe', False):
            params += list(trainer.encoder.parameters())

        optimizer = OptimizerFactory.create(
            self.config.get('optimizer', 'sgd'),
            params,
            self.config.get('optimizer_params', {})
        )

        scheduler_params = self.config.get('scheduler_params', {}).copy()
        if 'T_max' not in scheduler_params:
            scheduler_params['T_max'] = int(self.config.get('finetune_epochs', 100))

        scheduler = SchedulerFactory.create(
            self.config.get('scheduler', 'step'),
            optimizer,
            scheduler_params
        )

        best_acc = 0
        best_epoch = 0

        for epoch in range(1, int(self.config.get('finetune_epochs', 100)) + 1):
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer)
            test_metrics, _, _ = trainer.evaluate(test_loader)

            scheduler.step()

            print(f"\nEpoch {epoch}/{self.config.get('finetune_epochs', 100)}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Test Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_macro']:.4f}")

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_acc': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1_macro']
                })

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                best_epoch = epoch

        print(f"\nBest Accuracy: {best_acc:.4f} (Epoch {best_epoch})")

        if self.use_wandb:
            wandb.log({'best_accuracy': best_acc})
            wandb.finish()

        return best_acc

    def run(self):
        """Execute experiment"""

        print(f"Experiment: {self.experiment_name}")
        print(f"Mode: {self.config.get('mode', 'full')}")

        mode = self.config.get('mode', 'full')
        pretrained_path = None

        if mode in ['pretrain', 'full']:
            pretrained_path = self.run_ssl_pretraining()

        if mode in ['finetune', 'full']:
            self.run_finetuning(pretrained_path)

        print(f"\nExperiment complete: {self.experiment_name}")


def main():
    parser = argparse.ArgumentParser(
        description='STL-10 SSL Experiment Runner (Single Experiment Only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
        Examples:
          Baseline experiments:
          python experiments/run_experiment.py --experiment baselines/baseline_resnet18
          python experiments/run_experiment.py --experiment baselines/baseline_resnet50
          python experiments/run_experiment.py --experiment baselines/baseline_vit_small
        
          SimCLR pretraining:
          python experiments/run_experiment.py --experiment ssl_pretraining/simclr_resnet18
        
          Linear probing:
          python experiments/run_experiment.py --experiment linear_probing/simclr_resnet18_lp
        
          Full fine-tuning:
          python experiments/run_experiment.py --experiment full_finetune/simclr_resnet18_ft
        
          Transfer learning (ImageNet):
          python experiments/run_experiment.py --experiment transfer_learning/imagenet_resnet18
          python experiments/run_experiment.py --experiment transfer_learning/imagenet_resnet50
        
          Other:
          python experiments/run_experiment.py --list
          python experiments/run_experiment.py --info baselines/baseline_resnet18
          python experiments/run_experiment.py --experiment ssl_pretraining/simclr_resnet18 --use-wandb
        '''
    )

    parser.add_argument('--experiment', type=str, help='Experiment path (e.g., baselines/baseline_resnet18)')
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    parser.add_argument('--info', type=str, help='Show experiment details')
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()

    config_manager = ConfigManager()

    if args.list:
        config_manager.print_available_experiments()
        return

    if args.info:
        config_manager.print_experiment_info(args.info)
        return

    if args.experiment:
        try:
            exp_config = config_manager.get_experiment(args.experiment)
            exp_config['use_wandb'] = args.use_wandb
            exp_config['device'] = args.device

            runner = ExperimentRunner(exp_config)
            runner.run()

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()