import torch
import torch.nn as nn
import wandb

from tqdm import tqdm
from pathlib import Path


class SSLTrainer:
    """Unified trainer for self-supervised learning methods"""

    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            device,
            log_dir='./logs',
            use_wandb=True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.global_step = 0
        self.best_loss = float('inf')

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, (images, _) in enumerate(pbar):
            if isinstance(images, list):
                x1, x2 = images[0].to(self.device), images[1].to(self.device)
            else:
                x1 = images.to(self.device)
                x2 = None

            if x2 is not None:
                loss = self.model(x1, x2)
            else:
                loss = self.model(x1)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': self.global_step
                })

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def train(self, dataloader, num_epochs, save_freq=10):
        print(f"Starting SSL pretraining for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

            if epoch % save_freq == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, avg_loss)

        print("SSL pretraining completed!")

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }

        path = self.log_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.log_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model updated!")