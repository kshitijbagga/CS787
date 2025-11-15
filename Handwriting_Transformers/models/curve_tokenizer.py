import cv2
import numpy as np
from skimage.morphology import skeletonize
import torch
import torch.nn as nn

def extract_curves_from_word(img, max_curves=100):
    """
    Takes a grayscale handwriting word image.
    Returns list of fitted curve segments as normalized 8D vectors.
    """
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # --- Adaptive or Otsu binarization
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, img_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # small dilation to connect broken ink pixels
    kernel = np.ones((1,1), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)

    # optional erosion to remove small noise
    img_bin = cv2.erode(img_bin, kernel, iterations=5)
    img_bin = (img_bin > 0).astype(np.uint8)
    skel = skeletonize(img_bin).astype(np.uint8)

    contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curves = []

    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim < 2 or len(pts) < 4:
            continue

        for i in range(0, len(pts)-3, 3):
            if len(curves) >= max_curves:
                break
            p = pts[i:i+4].flatten().astype(np.float32)
            # normalize to [0,1]
            p[0::2] /= img.shape[1]
            p[1::2] /= img.shape[0]
            curves.append(p)

    if len(curves) == 0:
        curves.append(np.zeros(8, dtype=np.float32))
    return np.stack(curves)
    

class CurveTokenEncoder(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)  # [B, n_curves, out_dim]


class CurveSequenceEncoder(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
    def forward(self, tokens):
        x = tokens.permute(1,0,2)
        z = self.encoder(x)
        return z.mean(dim=0)  # [B, embed_dim]


class WordCurveTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = CurveTokenEncoder()
        self.sequence_encoder = CurveSequenceEncoder()

    def forward(self, curve_points):
        tokens = self.token_encoder(curve_points)
        word_embed = self.sequence_encoder(tokens)
        return word_embed
