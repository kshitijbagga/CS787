# train_diffusion_finetune.py
"""
Powerful training script (Option B):
- trains DDPM on style exemplars (simg)
- classifier-free guidance via cond_drop_prob
- EMA of weights (ema_decay)
- paragraph sampling using adapter
- saves checkpoints and sample pages
"""
import os
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# local imports - adjust based on your repo structure
from data.dataset import TextDataset
from models.diffusion.style_encoder import StyleEncoder
from models.diffusion.unet_small import UNetSmall
from models.diffusion.simple_ddpm import SimpleDDPM, EMA
from util.page_renderer import to_numpy_img, assemble_page_from_word_images

# helper
def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(x, device) for x in obj)
    return obj

def sample_paragraph(ddpm, style_enc, paragraph, style_images, device, out_path,
                     guidance=1.5, steps=100, batch_sample=8, max_line_width=1024):
    """
    Generate one page from paragraph using style_images (B,N,C,H,W) or (N,C,H,W).
    Sample one image per word.
    """
    ddpm.eval()
    style_enc.eval()
    if torch.is_tensor(style_images) and style_images.ndim==5:
        s = style_images[0].to(device)
    elif torch.is_tensor(style_images) and style_images.ndim==4:
        s = style_images.to(device)
    else:
        raise ValueError("style_images must be torch tensor with dims (B,N,C,H,W) or (N,C,H,W)")
    with torch.no_grad():
        emb = style_enc(s)                    # (N, cond_dim)
        cond_vec = emb.mean(dim=0, keepdim=True)  # (1, cond_dim)

    words = [w for w in paragraph.strip().split() if len(w)>0]
    if len(words)==0:
        words = [""]

    imgs = []
    idx = 0
    while idx < len(words):
        bs = min(batch_sample, len(words) - idx)
        cond_batch = cond_vec.repeat(bs,1).to(device)
        samp = ddpm.sample(batch_size=bs, cond=cond_batch, image_size=(1, s.shape[1], s.shape[2]), steps=steps, guidance_scale=guidance)
        # convert each to numpy (H,W)
        for i in range(samp.shape[0]):
            arr = to_numpy_img(samp[i])
            imgs.append(arr)
        idx += bs

    page = assemble_page_from_word_images(imgs, max_line_width=max_line_width)
    Image.fromarray(page).save(out_path)
    ddpm.train()
    style_enc.train()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="files/IAM-32.pickle")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--cond_drop_prob", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--sample_interval", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--samples_dir", type=str, default="samples")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--base_ch", type=int, default=128)
    return p.parse_args()

def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.samples_dir).mkdir(parents=True, exist_ok=True)

    ds = TextDataset(base_path=args.data, num_examples=15)
    collate_fn = ds.collate_fn
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    # models
    cond_dim = 256
    style_enc = StyleEncoder(out_dim=cond_dim).to(device)
    unet = UNetSmall(in_ch=1, base_ch=args.base_ch, cond_dim=cond_dim).to(device)
    # store cond dim on unet for sample helper
    unet.cond_dim = cond_dim
    ddpm = SimpleDDPM(unet, timesteps=args.timesteps, device=device, cond_drop_prob=args.cond_drop_prob).to(device)

    # EMA
    ema = EMA(ddpm, decay=args.ema_decay)

    opt = optim.Adam(list(ddpm.unet.parameters()) + list(style_enc.parameters()), lr=args.lr, betas=(0.9, 0.999))

    global_step = 0
    start_time = time.time()

    ddpm.train()
    style_enc.train()

    ckpt_latest = os.path.join(args.save_dir, "latest.pth")
    if os.path.exists(ckpt_latest):
        ck = torch.load(ckpt_latest, map_location=device)
        ddpm.load_state_dict(ck.get("ddpm", {}))
        style_enc.load_state_dict(ck.get("style_enc", {}))
        opt.load_state_dict(ck.get("opt", {}))
        global_step = ck.get("global_step", global_step)
        print("Resumed checkpoint at step", global_step)

    print("Starting training loop...")
    for ep in range(args.epochs):
        for batch in loader:
            global_step += 1
            batch = to_device(batch, device)
            simg = batch["simg"].float()   # (B, N, C, H, W)
            # ensure simg dims consistent
            if simg.ndim != 5:
                raise RuntimeError("simg must have shape (B,N,C,H,W)")
            B,N,C,H,W = simg.shape
            # flatten style images for encoding
            styles_flat = simg.reshape(B*N, C, H, W)   # (B*N,C,H,W)
            # encode
            emb = style_enc(styles_flat)                # (B*N, cond_dim)
            cond = emb.reshape(B, N, -1).mean(dim=1)    # (B, cond_dim)

            # prepare training batch: use style exemplars as targets
            x_real = styles_flat.to(device)             # (B*N, C, H, W)
            # cond vector per style image (repeat cond per N)
            cond_repeat = cond.unsqueeze(1).repeat(1, N, 1).reshape(B*N, -1)   # (B*N, cond_dim)

            opt.zero_grad()
            loss = ddpm(x_real, cond_repeat)
            loss.backward()
            opt.step()
            # update EMA of the entire ddpm module (unet + buffers)
            ema.update(ddpm)

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"[step {global_step}] ep={ep} loss={loss.item():.6f} elapsed={elapsed:.1f}s")

            # sampling (use device writer 0 from this batch)
            if global_step % args.sample_interval == 0:
                try:
                    paragraph = open("mytext.txt", "r", encoding="utf-8").read()
                except:
                    paragraph = "This is a test paragraph for sampling handwriting generation."

                # use first writer's simg set to get cond/shape
                style_for_sampling = simg[0].to(device)   # (N, C, H, W)
                # use ema weights for sampling
                ddpm_eval = ddpm
                # sample with EMA
                out_img = ddpm.sample(batch_size=1, cond=cond[0:1].to(device), image_size=(C,H,W),
                                      steps=args.sample_steps, guidance_scale=args.guidance, use_ema=True, ema=ema)
                # Save one generated tile as check
                sample_tile = to_numpy_img(out_img[0])
                Image.fromarray(sample_tile).save(os.path.join(args.samples_dir, f"sample_tile_{global_step:06d}.png"))

                # Generate page by sampling one image per word using EMA sampling
                sample_path = os.path.join(args.samples_dir, f"page_{global_step:06d}.png")
                # Use sampling helper that repeats internal sampling by word count (keeps GPU CPU memory reasonable)
                # We'll call ddpm.sample inside helper (use use_ema=True)
                # reuse sample_paragraph helper
                try:
                    sample_paragraph(ddpm, style_enc, paragraph, simg, device, sample_path,
                                     guidance=args.guidance, steps=args.sample_steps, batch_sample=8, max_line_width=1024)
                    print("Saved page sample ->", sample_path)
                except Exception as e:
                    print("Page sampling failed:", e)

            if global_step % args.save_interval == 0:
                ck = {
                    "ddpm": ddpm.state_dict(),
                    "style_enc": style_enc.state_dict(),
                    "opt": opt.state_dict(),
                    "global_step": global_step
                }
                torch.save(ck, os.path.join(args.save_dir, f"ckpt_{global_step:06d}.pth"))
                torch.save(ck, os.path.join(args.save_dir, "latest.pth"))
                print("Saved checkpoint at step", global_step)

            if args.max_iter is not None and global_step >= args.max_iter:
                print("Reached max_iter -> stop")
                return

    print("Training finished.")

if __name__ == "__main__":
    args = parse_args()
    train(args)
