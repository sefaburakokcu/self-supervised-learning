import torch
import torch.nn as nn
import numpy as np


class MAE(nn.Module):
    """Masked Autoencoder for Vision Transformers"""

    def __init__(self, encoder, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder  # ViT encoder
        self.mask_ratio = mask_ratio

        encoder_dim = encoder.embed_dim

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=8,
                dim_feedforward=decoder_dim * 4,
                batch_first=True
            )
            for _ in range(4)
        ])

        patch_size = 16
        self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * 3)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: loss, predicted, mask
        """
        patches, mask, ids_restore = self._random_masking(x)

        latent = self.encoder.forward_features(patches)

        pred = self._forward_decoder(latent, ids_restore)

        loss = self._compute_loss(x, pred, mask)

        return loss

    def _random_masking(self, x):
        """Random masking of image patches"""
        B, C, H, W = x.shape
        patch_size = 16
        num_patches = (H // patch_size) * (W // patch_size)

        len_keep = int(num_patches * (1 - self.mask_ratio))

        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones(B, num_patches, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def _forward_decoder(self, latent, ids_restore):
        x = self.decoder_embed(latent)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = self.decoder_blocks(x)

        pred = self.decoder_pred(x)

        return pred

    def _compute_loss(self, imgs, pred, mask):
        """MSE loss on masked patches"""
        target = self._patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        return loss

    def _patchify(self, imgs):
        """Convert image to patches"""
        B, C, H, W = imgs.shape
        patch_size = 16
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        patches = imgs.reshape(
            B, C, num_patches_h, patch_size, num_patches_w, patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            B, num_patches_h * num_patches_w, C * patch_size * patch_size
        )
        return patches