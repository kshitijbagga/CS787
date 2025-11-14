# models/diffusion/ddpm_adapter.py
import math
import torch
import torch.nn as nn
from typing import Optional

class DDPMGeneratorAdapter(nn.Module):
    """
    A wrapper that exposes:
        .sample(cond, batch_size, device, guidance_scale)
    and can convert image output → sequence if needed.
    """

    def __init__(self, ddpm_model,
                 target_format="image",
                 seq_len=300,
                 out_dim=5,
                 H=32,
                 W=92):
        super().__init__()
        self.ddpm = ddpm_model
        self.target_format = target_format
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.H = 32
        self.W = 92

    @torch.no_grad()
    def sample(self, cond: Optional[torch.Tensor],
               batch_size: int,
               device,
               guidance_scale: float = 1.0,
               steps: Optional[int] = None,
               **kwargs):

        if isinstance(device, str):
            device = torch.device(device)

        # Normalize cond shape
        if cond is not None and cond.shape[0] != batch_size:
            cond = cond.repeat(batch_size, 1)

        imgs = self.ddpm.sample(
            shape=(batch_size, 1, self.H, self.W),
            cond=cond,
            device=device,
            steps=steps,
            guidance_scale=guidance_scale
        )

        if self.target_format == "image":
            return imgs.float()

        # If output is sequence — flatten
        if self.target_format == "sequence":
            B, C, H, W = imgs.shape
            flat = imgs.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

            # pad or truncate to seq_len
            if flat.shape[1] < self.seq_len:
                pad = torch.zeros(B, self.seq_len - flat.shape[1], flat.shape[2], device=imgs.device)
                flat = torch.cat([flat, pad], dim=1)

            flat = flat[:, :self.seq_len, :self.out_dim]  # trim channels
            return flat.float()

        raise RuntimeError("Unknown output format.")
