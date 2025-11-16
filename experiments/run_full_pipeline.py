import torch
import torch.nn as nn
import torch.optim as optim
import timm
import wandb
import argparse
import sys

from pathlib import Path
from torchvision.models import resnet18, resnet50

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.stl10_dataset import STL10DataModule
from models.ssl_methods.simclr import SimCLR
from models.ssl_methods.moco import MoCoV3
from models.ssl_methods.byol import BYOL
from models.ssl_methods.mae import MAE
from trainers.ssl_trainer import SSLTrainer
from trainers.finetune_trainer import FineTuneTrainer


def create_encoder(arch='resnet18', pretrained=False):
    """Factory function for encoders"""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained)
    elif arch == 'vit_small':
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model


def run_ssl_pretraining(args):
    """Phase 1: Self-supervised pretraining"""
    print(f"\n{'=' * 60}")
    print(f"SSL Pretraining: {args.ssl_method} + {args.arch}")
    print(f"{'=' * 60}\n")

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project='stl10-ssl',
            name=f'{args.ssl_method}_{args.arch}_pretrain',
            config=vars(args)
        )

    data_module = STL10DataModule(batch_size=args.batch_size)
    ssl_loader = data_module.get_ssl_dataloaders(ssl_method=args.ssl_method)

    encoder = create_encoder(args.arch, pretrained=False)

    if args.ssl_method == 'simclr':
        model = SimCLR(encoder, projection_dim=128, temperature=0.5)
    elif args.ssl_method == 'moco':
        model = MoCoV3(encoder, projection_dim=256, temperature=0.2)
    elif args.ssl_method == 'byol':
        model = BYOL(encoder, projection_dim=256, hidden_dim=4096)
    elif args.ssl_method == 'mae':
        model = MAE(encoder, decoder_dim=512, mask_ratio=0.75)
    else:
        raise ValueError(f"Unknown SSL method: {args.ssl_method}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    trainer = SSLTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        log_dir=f'./logs/{args.ssl_method}_{args.arch}',
        use_wandb=args.use_wandb
    )

    # Train
    trainer.train(ssl_loader, num_epochs=args.epochs, save_freq=20)

    if args.use_wandb:
        wandb.finish()

    return f'./logs/{args.ssl_method}_{args.arch}/best_model.pth'


def run_finetuning(args, pretrained_path=None):
    """Phase 2: Supervised fine-tuning"""
    print(f"\n{'=' * 60}")
    print(f"Fine-tuning: {args.arch}")
    print(f"{'=' * 60}\n")

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project='stl10-ssl',
            name=f'{args.arch}_finetune',
            config=vars(args)
        )

    data_module = STL10DataModule(batch_size=64)
    train_loader, test_loader = data_module.get_supervised_dataloaders()

    if pretrained_path:
        print(f"Loading pretrained encoder from {pretrained_path}")
        checkpoint = torch.load(pretrained_path)
        encoder = create_encoder(args.arch, pretrained=False)

        if args.ssl_method in ['simclr', 'moco', 'byol']:
            encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif args.ssl_method == 'mae':
            encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif args.use_imagenet:
        print("Loading ImageNet pretrained encoder")
        encoder = create_encoder(args.arch, pretrained=True)
    else:
        print("Training from scratch")
        encoder = create_encoder(args.arch, pretrained=False)

    trainer = FineTuneTrainer(
        encoder=encoder,
        num_classes=10,
        freeze_encoder=args.linear_probe,
        device=args.device
    )

    params = list(trainer.classifier.parameters())
    if not args.linear_probe:
        params += list(trainer.encoder.parameters())

    optimizer = optim.SGD(
        params,
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer)
        test_metrics, _, _ = trainer.evaluate(test_loader)

        scheduler.step()

        print(f"Epoch {epoch}/{args.finetune_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_macro']:.4f}")

        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_metrics['accuracy'],
                'test_f1_macro': test_metrics['f1_macro']
            })

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']

    if args.use_wandb:
        wandb.finish()

    return best_acc


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--mode', type=str, default='full', choices=['pretrain', 'finetune', 'full'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vit_small'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true')

    # SSL pretraining
    parser.add_argument('--ssl_method', type=str, default='simclr',
                        choices=['simclr', 'moco', 'byol', 'mae'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Fine-tuning
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--use_imagenet', action='store_true')

    args = parser.parse_args()

    if args.mode in ['pretrain', 'full']:
        pretrained_path = run_ssl_pretraining(args)
    else:
        pretrained_path = None

    if args.mode in ['finetune', 'full']:
        run_finetuning(args, pretrained_path)


if __name__ == '__main__':
    main()