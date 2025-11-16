import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """SimCLR: A Simple Framework for Contrastive Learning"""

    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x1, x2):
        """
        x1, x2: Two augmented views [B, C, H, W]
        Returns: loss value
        """
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        loss = self.nt_xent_loss(z1, z2)

        return loss

    def nt_xent_loss(self, z1, z2):
        """Normalized Temperature-scaled Cross Entropy Loss"""
        batch_size = z1.shape[0]

        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        sim_matrix = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        pos_sim = torch.exp(torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ]))

        all_sim = torch.exp(sim_matrix).sum(dim=1)

        loss = -torch.log(pos_sim / all_sim).mean()

        return loss