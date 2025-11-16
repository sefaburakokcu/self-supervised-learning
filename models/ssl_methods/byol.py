import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


class BYOL(nn.Module):
    """Bootstrap Your Own Latent: Non-contrastive SSL"""

    def __init__(self, encoder, projection_dim=256, hidden_dim=4096, momentum=0.996):
        super().__init__()
        self.momentum = momentum

        self.online_encoder = encoder
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.online_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = deepcopy(self.online_projector)

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self):
        for param_o, param_t in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_o.data

        for param_o, param_t in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_o.data

    def forward(self, x1, x2):
        online_proj_1 = self.online_projector(self.online_encoder(x1))
        online_proj_2 = self.online_projector(self.online_encoder(x2))

        online_pred_1 = self.predictor(online_proj_1)
        online_pred_2 = self.predictor(online_proj_2)

        with torch.no_grad():
            self._update_target_network()
            target_proj_1 = self.target_projector(self.target_encoder(x1))
            target_proj_2 = self.target_projector(self.target_encoder(x2))

        loss = (
                       self._regression_loss(online_pred_1, target_proj_2) +
                       self._regression_loss(online_pred_2, target_proj_1)
               ) / 2

        return loss

    def _regression_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=1).mean()