import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report


class FineTuneTrainer:
    """Fine-tune pretrained encoder on labeled data"""

    def __init__(
            self,
            encoder,
            num_classes=10,
            freeze_encoder=False,
            device='cuda'
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.num_classes = num_classes

        feature_dim = self._get_feature_dim()
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()

    def _get_feature_dim(self):
        dummy_input = torch.randn(1, 3, 96, 96).to(self.device)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        return features.shape[1]

    def train_epoch(self, dataloader, optimizer):
        self.encoder.train()
        self.classifier.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc='Training'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.encoder(images)
            logits = self.classifier(features)
            loss = self.criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.encoder.eval()
        self.classifier.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

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

        print(f"\nEvaluation Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (Macro): {f1_macro:.4f}")
        print(f"  F1 (Weighted): {f1_weighted:.4f}")

        return metrics, all_preds, all_labels