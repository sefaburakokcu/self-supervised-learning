import torch
import wandb

from tqdm import tqdm
from pathlib import Path


class SSLTrainer:
    """
    Trainer for self-supervised learning (SSL) methods.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        log_dir='./logs',
        use_wandb=True,
        logger=None
    ):
        """
        Initialize SSLTrainer.

        Args:
            model: Self-supervised model implementing forward(x1, x2?) returning loss.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler or None.
            device: Torch device.
            log_dir: Directory where checkpoints will be stored.
            use_wandb: Enable or disable Weights & Biases logging.
            logger: Python logging.Logger instance for structured logging.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.logger = logger

        self.global_step = 0
        self.best_loss = float('inf')

    def train_epoch(self, dataloader, epoch):
        """
        Train for a single epoch.

        Args:
            dataloader: SSL dataloader yielding augmented image pairs or single views.
            epoch: Current epoch number.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, (images, _) in enumerate(pbar):
            if isinstance(images, list):
                x1, x2 = images[0].to(self.device), images[1].to(self.device)
            else:
                x1 = images.to(self.device)
                x2 = None

            loss = self.model(x1, x2) if x2 is not None else self.model(x1)

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
        """
        Full SSL training loop.

        Args:
            dataloader: SSL data loader.
            num_epochs: Number of epochs to train.
            save_freq: Save checkpoint every N epochs.
        """
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.4f}"
                )

            if epoch % save_freq == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, avg_loss)

        if self.logger:
            self.logger.info("SSL pretraining completed")

    def save_checkpoint(self, epoch, loss):
        """
        Save a checkpoint of model and optimizer state.

        Args:
            epoch: Epoch number.
            loss: Loss at this checkpoint (for best model tracking).
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }

        path = self.log_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)

        if self.logger:
            self.logger.info(f"Checkpoint saved: {path}")

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.log_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

            if self.logger:
                self.logger.info(
                    f"Best model updated at epoch {epoch} with loss {loss:.4f}"
                )
