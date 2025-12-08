import torch
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report


class FineTuneTrainer:
    """
    Fine-tuning trainer for supervised learning on top of a pretrained encoder.
    """

    def __init__(
        self,
        encoder,
        num_classes=10,
        freeze_encoder=False,
        device='cuda',
        logger=None,
        log_dir='./ft_logs',
        use_wandb=True
    ):
        """
        Initialize FineTuneTrainer.

        Args:
            encoder: Pretrained encoder model producing feature vectors.
            num_classes: Number of output classes.
            freeze_encoder: If True, encoder weights are frozen.
            device: Torch device.
            logger: Python logging.Logger instance for structured logging.
        """
        self.encoder = encoder.to(device)
        self.device = device
        self.num_classes = num_classes
        self.logger = logger
        self.use_wandb = use_wandb
        self.freeze_encoder = freeze_encoder

        feature_dim = self._get_feature_dim()
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()

        # Logging + checkpoint directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_acc = 0.0

    def _get_feature_dim(self):
        dummy_input = torch.randn(1, 3, 96, 96).to(self.device)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        return features.shape[1]

    def train_epoch(self, dataloader, optimizer, epoch):
        """
        Run one supervised training epoch.

        Args:
            dataloader: Training dataloader with (image, label) batches.
            optimizer: Optimizer for classifier + optionally encoder.
            epoch: Current training epoch.

        Returns:
            avg_loss, accuracy
        """
        if self.freeze_encoder:
            # Linear probe: do NOT update BN stats / dropout
            self.encoder.eval()
        else:
            # Full fine-tuning
            self.encoder.train()
        self.classifier.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.encoder(images)
            logits = self.classifier(features)
            loss = self.criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            self.global_step += 1

            if self.use_wandb and batch_idx % 10 == 0:
                import wandb
                wandb.log({
                    'ft_train_loss': loss.item(),
                    'epoch': epoch,
                    'step': self.global_step
                })

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.encoder.eval()
        self.classifier.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.encoder(images)
            logits = self.classifier(features)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

        if self.logger:
            self.logger.info(
                "Evaluation metrics: "
                f"Loss={avg_loss:.4f}, "
                f"Accuracy={accuracy:.4f}, "
                f"F1_macro={f1_macro:.4f}, "
                f"F1_weighted={f1_weighted:.4f}"
            )

        return metrics, all_preds, all_labels

    def train(self, train_loader, val_loader, optimizer, scheduler=None,
              num_epochs=20, save_freq=5):
        """
        Full fine-tuning training loop.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            optimizer: Optimizer for classifier (and encoder if unfrozen).
            scheduler: Optional LR scheduler.
            num_epochs: Number of epochs.
            save_freq: Save checkpoint every N epochs.
        """
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch)

            if scheduler is not None:
                scheduler.step()

            metrics, _, _ = self.evaluate(val_loader)
            val_acc = metrics['accuracy']

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {metrics['loss']:.4f}, Val Acc: {val_acc:.4f}"
                )

            if self.use_wandb:
                import wandb
                wandb.log({
                    'ft_val_loss': metrics['loss'],
                    'ft_val_accuracy': val_acc,
                    'epoch': epoch
                })

            if epoch % save_freq == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, val_acc)

        if self.logger:
            self.logger.info("Fine-tuning completed")

    def save_checkpoint(self, epoch, val_acc):
        """
        Save model + classifier checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'val_accuracy': val_acc,
            'global_step': self.global_step
        }

        path = self.log_dir / f'latest_ft_model.pth'
        torch.save(checkpoint, path)

        if self.logger:
            self.logger.info(f"Checkpoint saved: {path}")

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            best_path = self.log_dir / 'best_ft_model.pth'
            torch.save(checkpoint, best_path)

            if self.logger:
                self.logger.info(
                    f"Best finetuned model updated at epoch {epoch} "
                    f"with accuracy {val_acc:.4f}"
                )
