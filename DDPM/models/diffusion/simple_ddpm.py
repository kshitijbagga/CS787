# models/diffusion/simple_ddpm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from typing import Optional

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class EMA:
    def __init__(self, model, decay=0.9999):
        self.shadow = {}
        self.decay = decay
        self.model = model
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().cpu().clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                new = p.detach().cpu()
                self.shadow[name].mul_(self.decay).add_(new, alpha=1.0 - self.decay)

    def store(self):
        self._store = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._store[n] = p.detach().cpu().clone()

    def copy_to(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n].to(p.device))

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._store[n].to(p.device))
        self._store = None

class SimpleDDPM(nn.Module):
    """
    Simple DDPM wrapper:
    - unet: network predicting noise given noisy image and cond vector
    - timesteps: diffusion timesteps
    - cond_drop_prob: during training, randomly zero cond vector to teach unconditional behavior
    """
    def __init__(self, unet, timesteps=1000, device='cpu', cond_drop_prob=0.1):
        super().__init__()
        self.unet = unet
        self.device = device
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.cond_drop_prob = cond_drop_prob

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1,1).to(x_start.device)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1).to(x_start.device)
        return a * x_start + b * noise

    def forward(self, x_start, cond):
        """
        Training step:
        x_start: [B, C, H, W] in [-1,1]
        cond: [B, cond_dim]
        Returns: mse loss
        """
        B = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.timesteps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        # classifier-free: randomly drop conditioning
        if self.cond_drop_prob > 0.0 and self.training:
            mask = (torch.rand(B, device=device) >= self.cond_drop_prob).float().view(B,1)
            cond_in = cond * mask
        else:
            cond_in = cond
        pred_noise = self.unet(x_noisy, cond_in)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, steps=100, guidance_scale=1.0):
        """
        Basic reverse sampling.
        shape: (B, C, H, W)
        cond: (B, cond_dim)
        guidance: classifier-free guidance scale
        Returns: images (B,C,H,W)
        """
        device = next(self.unet.parameters()).device
        B = shape[0]
        img = torch.randn(shape, device=device)

        # prepare unconditional cond (zero) for guidance
        cond_uncond = torch.zeros_like(cond)
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            # predict noise for cond and uncond (model trained with random drop so zero vector works)
            eps_cond = self.unet(img, cond)
            if guidance_scale != 1.0:
                eps_uncond = self.unet(img, cond_uncond)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond
            alpha = self.alphas_cumprod[i].to(device)
            alpha_prev = self.alphas_cumprod[i-1].to(device) if i > 0 else torch.tensor(1.0, device=device)
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha)
            img = coef1 * (img - coef2 * eps)
            if i > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - self.betas[i])).to(device)
                img = img + sigma * noise
        return img

    @torch.no_grad()
    def sample(self, batch_size, cond, image_size=(1,32,192), steps=None, guidance_scale=1.0, use_ema=False, ema=None):
        """
        High-level sampling API:
        - batch_size: integer
        - cond: (batch_size, cond_dim) or (1, cond_dim) (will repeat)
        - image_size: (C,H,W)
        - steps: sampling steps, defaults to self.timesteps if None
        - guidance_scale: float
        - use_ema + ema: if use_ema True, call ema.copy_to() before sampling
        """
        device = next(self.unet.parameters()).device
        if cond is None:
            cond = torch.zeros(batch_size, getattr(self.unet, "cond_dim", cond.shape[-1]), device=device)
        if cond.shape[0] != batch_size:
            cond = cond.repeat(batch_size, 1)
        if steps is None:
            steps = min(200, self.timesteps)
        shape = (batch_size, image_size[0], image_size[1], image_size[2])
        if use_ema and ema is not None:
            ema.store()
            ema.copy_to()
            out = self.p_sample_loop(shape, cond.to(device), steps=steps, guidance_scale=guidance_scale)
            ema.restore()
        else:
            out = self.p_sample_loop(shape, cond.to(device), steps=steps, guidance_scale=guidance_scale)
        return out
