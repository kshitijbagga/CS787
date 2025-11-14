
import os, torch, argparse
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.text import char_error_rate, word_error_rate
from models.OCR_network import CRNN
from models.model import TRGAN
from models.diffusion import UNetSmall, SimpleDDPM
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

def load_images_from_folder(folder, limit=None):
    paths = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            paths.append(os.path.join(folder, fname))
    if limit:
        paths = paths[:limit]
    return paths

def load_image(path):
    img = Image.open(path).convert('L')
    t = T.ToTensor()
    return t(img).unsqueeze(0)

def evaluate_ocr_accuracy(ocr_model, gen_folder, gt_folder):
    """Compare OCR outputs of generated vs. ground truth images"""
    gen_paths = sorted(load_images_from_folder(gen_folder))
    gt_paths = sorted(load_images_from_folder(gt_folder))
    total_cer, total_wer, n = 0, 0, 0
    for g, t in zip(gen_paths, gt_paths):
        g_img = load_image(g)
        t_img = load_image(t)
        with torch.no_grad():
            pred_text = ocr_model.predict(g_img)
            gt_text = ocr_model.predict(t_img)
        cer = char_error_rate(pred_text, gt_text)
        wer = word_error_rate(pred_text, gt_text)
        total_cer += cer
        total_wer += wer
        n += 1
    return total_cer / n, total_wer / n

def compute_fid(real_folder, fake_folder):
    fid = FrechetInceptionDistance(normalize=True)
    transform = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor()])
    def load_tensor_folder(folder):
        imgs = []
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(folder, f)).convert('RGB')
                imgs.append(transform(img))
        return torch.stack(imgs)
    real_imgs = load_tensor_folder(real_folder)
    fake_imgs = load_tensor_folder(fake_folder)
    fid.update(real_imgs, real=True)
    fid.update(fake_imgs, real=False)
    return fid.compute().item()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load TRGAN encoder and diffusion model
    trgan = TRGAN().to(device)
    trgan.eval()
    unet = UNetSmall(in_ch=1, base_ch=32, cond_dim=256).to(device)
    ddpm = SimpleDDPM(unet, timesteps=200, device=device).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    unet.load_state_dict(ckpt['unet'])
    print(f"Loaded checkpoint from {args.ckpt}")

    # Load style image and generate samples
    os.makedirs(args.gen_folder, exist_ok=True)
    style_paths = load_images_from_folder(args.style_folder, limit=args.num_samples)
    for i, sp in enumerate(style_paths):
        style = load_image(sp).to(device).float() / 255.0
        cond = trgan.encode_style(style)
        out = ddpm.sample((1,1,args.height,args.width), cond, steps=200)
        out = (out.clamp(-1,1)+1)/2
        save_image(out, os.path.join(args.gen_folder, f"gen_{i}.png"))
    print(f"Generated {len(style_paths)} samples in {args.gen_folder}")

    # FID computation
    fid_score = compute_fid(args.real_folder, args.gen_folder)
    print(f"FID Score: {fid_score:.4f}")

    # OCR evaluation (optional, if OCR_network has predict() function)
    try:
        ocr_model = CRNN().to(device)
        ocr_model.eval()
        cer, wer = evaluate_ocr_accuracy(ocr_model, args.gen_folder, args.real_folder)
        print(f"OCR Evaluation - CER: {cer:.4f}, WER: {wer:.4f}")
    except Exception as e:
        print("Skipping OCR evaluation due to error:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Trained diffusion checkpoint path')
    parser.add_argument('--style_folder', required=True, help='Folder of style images')
    parser.add_argument('--real_folder', required=True, help='Folder of real handwritten images for FID')
    parser.add_argument('--gen_folder', default='generated_eval', help='Folder to save generated images')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=192)
    args = parser.parse_args()
    main(args)
