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
            nn.Linear(feature_dim, projection_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim*4, projection_dim)
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
        """
        Numerically optimized NT-Xent loss (SimCLR).
        Uses explicit log-sum-exp stabilization.

        z1, z2: [B, D] L2-normalized embeddings
        Returns: scalar loss
        """

        device = z1.device
        B = z1.size(0)
        N = 2 * B

        # ------------------------------------------------------------
        # 1. Combine embeddings into [2B, D]
        # ------------------------------------------------------------
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # ------------------------------------------------------------
        # 2. Pairwise similarity matrix (cosine since z normalized)
        # ------------------------------------------------------------
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        # Mask diagonal (self-similarity)
        mask = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, -1e9)  # large negative constant → stable softmax ignore

        # ------------------------------------------------------------
        # 3. Identify positives (correct matching index)
        #    For each i, positive = i + B (mod 2B)
        # ------------------------------------------------------------
        pos_index = (torch.arange(N, device=device) + B) % N

        # Extract positive logit for each i
        pos_logits = sim[torch.arange(N, device=device), pos_index]  # [2B]

        # ------------------------------------------------------------
        # 4. Compute logsumexp for each row (denominator)
        #    log( sum_k exp(sim[i,k]) )
        # ------------------------------------------------------------
        denom_logsumexp = torch.logsumexp(sim, dim=1)  # [2B]

        # ------------------------------------------------------------
        # 5. Final NT-Xent loss: -log( exp(pos) / sum exp(all) )
        # ------------------------------------------------------------
        loss = - (pos_logits - denom_logsumexp)
        return loss.mean()
