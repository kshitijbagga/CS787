import os
import time
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Try to import repository's diffusion components; otherwise the user should add them.
try:
    from models.diffusion import StyleEncoder, UNetSmall, SimpleDDPM
except Exception as e:
    # If models.diffusion is not present, raise a clear error
    raise ImportError("models.diffusion module not found in repo. Please add your diffusion model (StyleEncoder, UNetSmall, SimpleDDPM). Original error: " + str(e))

# Try to import dataset; adapt if your dataset module has different name.
try:
    from data.dataset import TextDataset
except Exception as e:
    raise ImportError("data.dataset.TextDataset not found. Ensure your dataset loader exists. Original error: " + str(e))

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    ds = TextDataset(root='data', mode='train')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=lambda b: b)

    style_enc = StyleEncoder(out_dim=256).to(device)
    unet = UNetSmall(in_ch=1, base_ch=32, cond_dim=256).to(device)
    ddpm = SimpleDDPM(unet, timesteps=args.timesteps, device=device).to(device)

    optimizer = torch.optim.Adam(list(style_enc.parameters()) + list(unet.parameters()), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        running_loss = 0.0
        for i, item in enumerate(loader):
            # item is a list of dataset entries; adapt to your dataset's output structure
            # Expected that dataset returns dict with 'img' (BCHW) and 'simg' style imgs
            batch_imgs = torch.stack([it['img'] for it in item]).to(device).float() / 255.0
            batch_simgs = torch.stack([it['simg'] for it in item]).to(device).float() / 255.0

            # compute conditioning vector
            try:
                cond = style_enc(batch_simgs)
            except Exception:
                # fallback: if style_enc expects single images
                cond = style_enc(batch_simgs)

            # sample random timesteps for each sample
            b = batch_imgs.size(0)
            t = torch.randint(0, ddpm.timesteps, (b,), device=device).long()

            # compute ddpm loss (SimpleDDPM.forward should return mse loss)
            loss = ddpm(batch_imgs, cond)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('Loss/ddpm', loss.item(), global_step)
            global_step += 1

            if (i+1) % args.log_interval == 0:
                avg = running_loss / (i+1)
                print(f"Epoch {epoch} Iter {i+1} AvgLoss {avg:.6f}")

        print(f"Epoch {epoch} completed in {time.time()-t0:.1f}s avg_loss={running_loss/len(loader):.6f}")

        # sample and save a few images at epoch end (use ddpm.sample if available)
        with torch.no_grad():
            sample_shape = (min(4, args.batch_size), 1, args.img_h, args.img_w)
            # prepare cond for sampling: take first N style images
            cond_sample = cond[:sample_shape[0]].to(device)
            try:
                samples = ddpm.sample(sample_shape, cond_sample, steps=args.sample_steps)
            except Exception as e:
                print("ddpm.sample failed:", e)
                samples = None

            if samples is not None:
                # normalize and save a grid
                import torchvision.utils as vutils
                grid = vutils.make_grid((samples.clamp(-1,1)+1)/2.0, nrow=2)
                writer.add_image('Samples', grid, epoch)
                # also save checkpoint image
                os.makedirs(args.sample_dir, exist_ok=True)
                vutils.save_image(grid, os.path.join(args.sample_dir, f'epoch_{epoch}.png'))

        # checkpoint
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save({'style_enc': style_enc.state_dict(), 'unet': unet.state_dict()}, os.path.join(args.ckpt_dir, f'ddpm_epoch_{epoch}.pt'))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_dir', type=str, default='runs/ddpm')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_ddpm')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--img_h', type=int, default=32)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--log_interval', type=int, default=50)
    args = parser.parse_args()
    train(args)
