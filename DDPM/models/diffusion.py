
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Simple CNN style encoder to produce a conditioning vector
class StyleEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)
    def forward(self, x):
        # x: (B,1,H,W)
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)

# Small UNet for image-to-image diffusion
class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, cond_dim=256):
        super().__init__()
        self.inc = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, padding=1, stride=2), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1, stride=2), nn.ReLU())
        self.mid = nn.Sequential(nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1), nn.ReLU())
        self.outc = nn.Conv2d(base_ch, in_ch, 1)
        # condition proj
        self.cond_proj = nn.Linear(cond_dim, base_ch*4)
    def forward(self, x, cond):
        # x: (B,1,H,W), cond: (B,cond_dim)
        x1 = F.relu(self.inc(x))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        m = self.mid(x3)
        # add cond as bias
        b = self.cond_proj(cond).view(cond.size(0), -1, 1, 1)
        m = m + b
        u2 = self.up2(m)
        u1 = self.up1(u2 + x2)  # skip connection
        out = self.outc(u1 + x1)
        return out

# Simple DDPM utilities and wrapper
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class SimpleDDPM(nn.Module):
    def __init__(self, unet, timesteps=1000, device='cpu'):
        super().__init__()
        self.unet = unet
        self.device = device
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    def forward(self, x_start, cond):
        # training objective: predict noise
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        # predict noise with unet conditioned on cond
        pred_noise = self.unet(x_noisy, cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, shape, cond, steps=None):
        if steps is None:
            steps = self.timesteps
        device = next(self.parameters()).device
        img = torch.randn(shape, device=device)
        for i in reversed(range(steps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            betas_t = self.betas[i]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alphas_t = 1.0 / torch.sqrt(1.0 - self.betas[i])  # approx
            # predict noise
            pred_noise = self.unet(img, cond)
            # simple DDPM step (nice to keep it small for prototype)
            alpha = self.alphas_cumprod[i]
            alpha_prev = self.alphas_cumprod[i-1] if i>0 else torch.tensor(1.0, device=device)
            coef1 = 1/torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha)
            img = coef1 * (img - coef2 * pred_noise)
            if i>0:
                img = img + torch.sqrt((1 - alpha_prev)) * torch.randn_like(img)
        return img
