
import os, torch, argparse
from models.diffusion import UNetSmall, SimpleDDPM
from models.model import TRGAN
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

def load_image(path):
    img = Image.open(path).convert('L')
    t = T.ToTensor()
    return t(img).unsqueeze(0)  # (1,1,H,W)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize TRGAN and load its encoder
    trgan = TRGAN().to(device)
    trgan.eval()

    # Initialize UNet and DDPM (must match training config)
    unet = UNetSmall(in_ch=1, base_ch=32, cond_dim=256).to(device)
    ddpm = SimpleDDPM(unet, timesteps=200, device=device).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    unet.load_state_dict(ckpt['unet'])
    print(f"Loaded checkpoint from {args.ckpt}")

    # Load and process style image
    style = load_image(args.style).to(device).float() / 255.0
    cond = trgan.encode_style(style)  # use TRGAN encoder directly

    # Generate handwriting sample
    print("Sampling diffusion model...")
    out = ddpm.sample((1,1,args.height,args.width), cond, steps=200)
    out = (out.clamp(-1,1) + 1) / 2  # normalize to [0,1]

    # Save output image
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "generated_sample.png")
    save_image(out, out_path)
    print(f"✅ Saved generated image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to trained DDPM checkpoint (.pt)')
    parser.add_argument('--style', required=True, help='Path to a style image (handwriting example)')
    parser.add_argument('--out_dir', default='samples', help='Output directory')
    parser.add_argument('--height', type=int, default=64, help='Height of generated image')
    parser.add_argument('--width', type=int, default=192, help='Width of generated image')
    args = parser.parse_args()
    main(args)
