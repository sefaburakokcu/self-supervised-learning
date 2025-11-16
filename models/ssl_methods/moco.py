import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


class MoCoV3(nn.Module):
    """MoCo v3: Momentum Contrast with Vision Transformers"""

    def __init__(self, encoder, projection_dim=256, temperature=0.2, momentum=0.99):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum

        self.encoder_q = encoder
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.projection_q = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

        self.encoder_k = deepcopy(self.encoder_q)
        self.projection_k = deepcopy(self.projection_q)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False
        for param_k in self.projection_k.parameters():
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoder"""
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

        for param_q, param_k in zip(
                self.projection_q.parameters(), self.projection_k.parameters()
        ):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    def forward(self, x1, x2):
        # Query
        q = self.projection_q(self.encoder_q(x1))
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = self.projection_k(self.encoder_k(x2))
            k = F.normalize(k, dim=1)

        logits = torch.mm(q, k.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss