import os
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from pytorch_msssim import ssim
from data.dataset import TextDatasetval
from models.model import TRGAN
from params import *

# -----------------------------
# Config
# -----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'saved_models/IAM-339-15-E3D3-LR0.00005-bs8/model100.pth'
NUM_SAMPLES = 50  # number of images to evaluate
BATCH_SIZE = 4

os.makedirs('generated_images', exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
print("Loading TRGAN model...")
model = TRGAN().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()
print("✅ Model loaded successfully.")

# -----------------------------
# Load Validation Dataset
# -----------------------------
TextDatasetObjval = TextDatasetval(num_examples=NUM_EXAMPLES)
datasetval = DataLoader(
    TextDatasetObjval, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TextDatasetObjval.collate_fn
)

# -----------------------------
# Helper function: convert to 4D tensor
# -----------------------------
def to_4d_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if x.dim() == 2:
        # Flattened image: [H*W, ?] → [1,1,H,W]
        H = int(np.sqrt(x.size(0)))
        W = H
        x = x.view(1, 1, H, W)
    elif x.dim() == 3:
        # [C,H,W] → add batch dim
        x = x.unsqueeze(0)
    return x.float()

# -----------------------------
# Generate Images
# -----------------------------
generated_images = []
real_images = []

for i, data_val in enumerate(tqdm(datasetval)):
    if len(generated_images) >= NUM_SAMPLES:
        break

    style_image = data_val['simg'].to(DEVICE)
    swids = data_val['swids'].to(DEVICE)

    with torch.no_grad():
        fake_page = model._generate_page(style_image, swids)

    fake_page = to_4d_tensor(fake_page).to(DEVICE)
    style_image_norm = to_4d_tensor(style_image).to(DEVICE)

    # Normalize to [0,1]
    fake_page = ((fake_page + 1) / 2).clamp(0,1)
    style_image_norm = ((style_image_norm + 1) / 2).clamp(0,1)

    # Collect samples
    for j in range(fake_page.size(0)):
        generated_images.append(fake_page[j].cpu())
        real_images.append(style_image_norm[j].cpu())
        if len(generated_images) >= NUM_SAMPLES:
            break

# Stack tensors
generated_images = torch.stack(generated_images)
real_images = torch.stack(real_images)

# -----------------------------
# Compute SSIM
# -----------------------------
ssim_value = ssim(generated_images.to(DEVICE), real_images.to(DEVICE), data_range=1.0)
print(f"SSIM Score: {ssim_value.item():.4f}")

# -----------------------------
# Compute L2 / MSE
# -----------------------------
mse_value = F.mse_loss(generated_images.to(DEVICE), real_images.to(DEVICE))
print(f"MSE (L2) Score: {mse_value.item():.6f}")

# -----------------------------
# Save a few generated images
# -----------------------------
for idx in range(min(10, NUM_SAMPLES)):
    torchvision.utils.save_image(generated_images[idx], f'generated_images/sample_{idx}.png')

print("✅ Sample images saved to generated_images/")
