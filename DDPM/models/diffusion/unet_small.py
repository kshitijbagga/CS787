# models/diffusion/unet_small.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

# Down / Up
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
            nn.AvgPool2d(2)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff))
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNetSmall(nn.Module):
    """
    UNet with conditioning vector `cond` projected and added as bias to mid features.
    - in_ch: input channels (1)
    - base_ch: base channels (use 128+ for good capacity)
    - cond_dim: conditioning embedding dim
    """
    def __init__(self, in_ch=1, base_ch=128, cond_dim=256):
        super().__init__()
        self.inc = ConvBlock(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.mid1 = ConvBlock(base_ch*4, base_ch*4)
        self.mid2 = ConvBlock(base_ch*4, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up1 = Up(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        # conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, base_ch*4),
            nn.SiLU(),
            nn.Linear(base_ch*4, base_ch*4)
        )

    def forward(self, x, cond):
        """
        x: (B, C, H, W)
        cond: (B, cond_dim)
        """
        # encoder
        x1 = self.inc(x)       # (B, base_ch, H, W)
        x2 = self.down1(x1)    # (B, base_ch*2, H/2, W/2)
        x3 = self.down2(x2)    # (B, base_ch*4, H/4, W/4)
        m = self.mid1(x3)
        # add conditioning as bias across spatial dims
        b = self.cond_proj(cond).view(cond.size(0), -1, 1, 1)
        m = m + b
        m = self.mid2(m)
        u2 = self.up2(m, x2)
        u1 = self.up1(u2, x1)
        out = self.outc(u1)
        return out
