import torch
import wandb
import argparse
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.stl10_dataset import STL10DataModule
from models.backbones import create_encoder
from models.ssl_methods.ssl_model_factory import SSLModelFactory
from trainers.hyperparameters_factory import OptimizerFactory, SchedulerFactory
from trainers.ssl_trainer import SSLTrainer
from trainers.finetune_trainer import FineTuneTrainer
from utils.config_manager import ConfigManager


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    """Setup logger with file and console handlers"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_dir / f'{name}.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.flush()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


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

        self.logger = setup_logger(self.experiment_name, self.log_dir)
        self.logger.info(f"Initialized ExperimentRunner: {self.experiment_name}")
        self.logger.info(f"Device: {self.device}, Use W&B: {self.use_wandb}")

        self._save_config()

    def _save_config(self):
        """Save experiment configuration"""
        config_dir = self.checkpoint_dir / self.experiment_name
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'config.json'

        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

        self.logger.info(f"Config saved: {config_file}")

    def run_ssl_pretraining(self):
        """Run SimCLR SSL pretraining"""
        self.logger.info("=" * 80)
        self.logger.info(f"SSL Pretraining: SimCLR + {self.config['arch']}")
        self.logger.info("=" * 80)

        if self.use_wandb:
            self.logger.info("Initializing W&B")
            wandb.init(
                project='stl10-ssl',
                name=f"{self.experiment_name}_pretrain",
                config=self.config
            )

        try:
            self.logger.info(f"Loading STL10 dataset, batch_size={self.config['batch_size']}")
            data_module = STL10DataModule(
                batch_size=int(self.config['batch_size']),
                data_dir=self.config.get('data_dir', './datasets'),
                num_workers=self.config.get('num_workers', 4)
            )
            ssl_loader = data_module.get_ssl_dataloaders(ssl_method='simclr')
            self.logger.info(f"SSL data loader created")

            self.logger.info(f"Creating encoder: {self.config['arch']}")
            encoder = create_encoder(self.config['arch'], pretrained=False)

            ssl_kwargs = {}
            for key in ['projection_dim', 'hidden_dim', 'temperature', 'momentum',
                        'decoder_dim', 'decoder_depth', 'mask_ratio']:
                if key in self.config:
                    ssl_kwargs[key] = self.config[key]
                    self.logger.debug(f"SSL param {key}={self.config[key]}")

            self.logger.info(f"Creating SSL model: {self.config['ssl_method']}")
            model = SSLModelFactory.create(
                self.config['ssl_method'],
                encoder,
                **ssl_kwargs
            )

            param_count = sum(p.numel() for p in model.parameters())
            self.logger.info(f"Model parameters: {param_count:,}")

            self.logger.info(f"Creating optimizer: {self.config.get('optimizer', 'adamw')}")
            optimizer = OptimizerFactory.create(
                self.config.get('optimizer', 'adamw'),
                model.parameters(),
                self.config.get('optimizer_params', {})
            )

            scheduler_params = self.config.get('scheduler_params', {}).copy()
            if 'T_max' not in scheduler_params:
                scheduler_params['T_max'] = int(self.config['epochs'])

            self.logger.info(f"Creating scheduler: {self.config.get('scheduler', 'cosine')}")
            scheduler = SchedulerFactory.create(
                self.config.get('scheduler', 'cosine'),
                optimizer,
                scheduler_params
            )

            experiment_dir = self.checkpoint_dir / self.experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Starting SSL training for {self.config['epochs']} epochs")
            trainer = SSLTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                log_dir=str(experiment_dir),
                use_wandb=self.use_wandb,
                logger=self.logger
            )

            trainer.train(
                ssl_loader,
                num_epochs=int(self.config['epochs']),
                save_freq=self.config.get('save_freq', 10)
            )

            best_model_path = str(experiment_dir / 'best_model.pth')
            self.logger.info(f"SSL pretraining completed. Best model: {best_model_path}")

            if self.use_wandb:
                wandb.finish()

            return best_model_path

        except Exception as e:
            self.logger.error(f"SSL pretraining failed: {e}", exc_info=True)
            if self.use_wandb:
                wandb.finish()
            raise

    def run_finetuning(self, pretrained_path=None):
        """Run fine-tuning"""
        self.logger.info("=" * 80)
        self.logger.info(f"Fine-tuning: {self.config['arch']}")
        self.logger.info("=" * 80)

        if self.use_wandb:
            self.logger.info("Initializing W&B for fine-tuning")
            wandb.init(
                project='stl10-ssl',
                name=f"{self.experiment_name}_finetune",
                config=self.config
            )

        try:
            finetune_batch_size = int(self.config.get('batch_size', 64))
            self.logger.info(f"Loading supervised data, batch_size={finetune_batch_size}")

            data_module = STL10DataModule(
                batch_size=finetune_batch_size,
                data_dir=self.config.get('data_dir', './datasets'),
                num_workers=self.config.get('num_workers', 4)
            )
            train_loader, test_loader = data_module.get_supervised_dataloaders()
            self.logger.info("Supervised data loaders created")

            if pretrained_path:
                self.logger.info(f"Loading pretrained weights from {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location=self.device)

                encoder = create_encoder(self.config['arch'], pretrained=False)
                encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.logger.info("Pretrained weights loaded successfully")

            elif self.config.get('use_imagenet', False):
                self.logger.info("Using ImageNet pretrained weights")
                encoder = create_encoder(self.config['arch'], pretrained=True)

            else:
                self.logger.info("Training from scratch (no pretrained weights)")
                encoder = create_encoder(self.config['arch'], pretrained=False)

            linear_probe = self.config.get('linear_probe', False)
            self.logger.info(f"Creating fine-tuner, linear_probe={linear_probe}")

            experiment_dir = self.checkpoint_dir / self.experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)

            trainer = FineTuneTrainer(
                encoder=encoder,
                num_classes=10,
                freeze_encoder=linear_probe,
                device=self.device,
                logger=self.logger,
                log_dir=str(experiment_dir),
                use_wandb=self.use_wandb
            )

            params = list(trainer.classifier.parameters())
            if not linear_probe:
                params += list(trainer.encoder.parameters())

            trainable_params = sum(p.numel() for p in params)
            self.logger.info(f"Trainable parameters: {trainable_params:,}")

            self.logger.info(f"Creating optimizer: {self.config.get('optimizer', 'sgd')}")
            optimizer = OptimizerFactory.create(
                self.config.get('optimizer', 'sgd'),
                params,
                self.config.get('optimizer_params', {})
            )

            scheduler_params = self.config.get('scheduler_params', {}).copy()
            if self.config.get('scheduler', 'step') in ['cosine', 'warmup_cosine']:
                if 'T_max' not in scheduler_params:
                    scheduler_params['T_max'] = int(self.config.get('finetune_epochs', 100))

            self.logger.info(f"Creating scheduler: {self.config.get('scheduler', 'step')}")
            scheduler = SchedulerFactory.create(
                self.config.get('scheduler', 'step'),
                optimizer,
                scheduler_params
            )

            finetune_epochs = int(self.config.get('finetune_epochs', 100))
            self.logger.info(f"Starting fine-tuning for {finetune_epochs} epochs")

            trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=finetune_epochs,
                save_freq=self.config.get('save_freq', 10)
            )

            best_acc = trainer.best_acc
            self.logger.info(f"Fine-tuning complete. Best Accuracy: {best_acc:.4f}")

            if self.use_wandb:
                wandb.log({'best_accuracy': best_acc})
                wandb.finish()

            return best_acc

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}", exc_info=True)
            if self.use_wandb:
                wandb.finish()
            raise

    def run(self):
        """Execute experiment"""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Mode: {self.config.get('mode', 'full')}")

        mode = self.config.get('mode', 'full')
        pretrained_path = None

        try:
            if mode in ['pretrain', 'full']:
                self.logger.info("Executing pretraining phase")
                pretrained_path = self.run_ssl_pretraining()

            if mode in ['finetune', 'full']:
                self.logger.info("Executing fine-tuning phase")
                self.run_finetuning(pretrained_path)

            self.logger.info(f"Experiment completed successfully: {self.experiment_name}")

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


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
            logger = setup_logger('main', Path('./logs'))
            logger.info(f"Experiment requested: {args.experiment}")

            exp_config = config_manager.get_experiment(args.experiment)
            exp_config['use_wandb'] = args.use_wandb
            exp_config['device'] = args.device

            runner = ExperimentRunner(exp_config)
            runner.run()

            logger.info("Main execution completed successfully")

        except FileNotFoundError as e:
            logger.error(f"Config not found: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()