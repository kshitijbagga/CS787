# Copyright Amazon...
# SPDX-License-Identifier: MIT

import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import glob
from .params1 import *
from .image_utils1 import *
import torch.nn.functional as F

def ensure_pil(x):
    """Convert any Torch/Numpy/PIL input into a clean grayscale PIL Image."""
    if isinstance(x, Image.Image):
        return x

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
        if x.ndim == 3:
            x = x[0]
        x = (x * 255).astype("uint8")

    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            x = x[:, :, 0]
        return Image.fromarray(x.astype("uint8"))

    raise TypeError(f"Unexpected image type: {type(x)}")


def get_transform(grayscale=True):
    tf = []
    if grayscale:
        tf.append(transforms.Grayscale(1))
    tf += [transforms.ToTensor()]
    tf += [transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(tf)


class TextDataset(Dataset):
    def __init__(self, base_path=DATASET_PATHS, num_examples=15):
        self.NUM_EXAMPLES = num_examples

        with open(base_path, "rb") as f:
            self.IMG_DATA = pickle.load(f)["train"]

        if "None" in self.IMG_DATA:
            del self.IMG_DATA["None"]

        self.author_id = list(self.IMG_DATA.keys())
        self.transform = get_transform(True)
        self.collate_fn = TextCollator()

    def __len__(self):
        return len(self.author_id)

    def __getitem__(self, index):

        NUM_SAMPLES = self.NUM_EXAMPLES
        author_id = self.author_id[index]
        data = self.IMG_DATA[author_id]

        # ------------ Pick real image -----------------
        rid = np.random.randint(0, len(data))
        raw = data[rid]["img"]
        pil = ensure_pil(raw).convert("L")
        real_img = self.transform(pil)  # (1,32,W)
        real_label = data[rid]["label"].encode()

        # ------------ Pick style samples --------------
        idxs = np.random.choice(len(data), NUM_SAMPLES, replace=True)
        imgs_pad = []
        widths = []

        max_width = 192

        for i in idxs:
            raw = data[i]["img"]
            pil = ensure_pil(raw).convert("L")
            arr = np.array(pil)

            arr = 255 - arr
            H, W = arr.shape
            canvas = np.zeros((H, max_width), dtype="float32")
            canvas[:, :W] = arr[:, :max_width]
            canvas = 255 - canvas

            pil2 = Image.fromarray(canvas.astype("uint8"))
            imgs_pad.append(self.transform(pil2))
            widths.append(W)

        imgs_pad = torch.stack(imgs_pad, dim=0)  # (N, 1, 32, 192)

        return {
            "simg": imgs_pad,
            "swids": widths,
            "img": real_img,
            "label": real_label,
            "img_path": "",
            "idx": "",
            "wcl": index
        }


class TextDatasetval(TextDataset):
    def __init__(self, base_path=DATASET_PATHS, num_examples=15):
        self.NUM_EXAMPLES = num_examples

        with open(base_path, "rb") as f:
            self.IMG_DATA = pickle.load(f)["test"]

        if "None" in self.IMG_DATA:
            del self.IMG_DATA["None"]

        self.author_id = list(self.IMG_DATA.keys())
        self.transform = get_transform(True)
        self.collate_fn = TextCollator()


class TextCollator(object):
    """
    Collates a batch of items produced by TextDataset.__getitem__.
    Each item expected keys:
      - 'simg' : Tensor (N, C, H, W_i)   # N = NUM_EXAMPLES (per-author samples)
      - 'swids': list of widths length N  (or Tensor)
      - 'img'  : Tensor (C, H, W_r)      # single real img per author
      - 'label': list/bytes (optional)
      - 'wcl'  : int (writer/class id)
    This collator pads simg's last dim and img's last dim to the per-batch maxima.
    Returns dict with:
      'simg': (B, N, C, H, W_s) tensor
      'swids': (B, N) tensor (float)
      'img': (B, C, H, W_i) tensor
      'label': list (unchanged)
      'wcl': LongTensor (B,)
    """
    def __init__(self, resolution_h=32):
        self.resolution_h = resolution_h

    def pad_to(self, t, target_w):
        # t: (C, H, W) or (N, C, H, W) expected contiguous
        if t.dim() == 3:
            c,h,w = t.shape
            if w == target_w:
                return t
            pad = target_w - w
            # pad: (left, right) along last dim -> pad=(0, pad, 0, 0) for F.pad
            return F.pad(t, (0, pad, 0, 0))
        elif t.dim() == 4:
            n,c,h,w = t.shape
            if w == target_w:
                return t
            pad = target_w - w
            return F.pad(t, (0, pad, 0, 0))
        else:
            raise ValueError("Unexpected tensor dim in pad_to: %s" % (t.dim(),))

    def __call__(self, batch):
        # batch: list of dicts
        B = len(batch)

        # --- handle simg (list per sample) ---
        simg_list = [b['simg'] for b in batch]         # each: (N, C, H, W_s_i)
        N = simg_list[0].shape[0]
        C = simg_list[0].shape[1]
        H = simg_list[0].shape[2]

        # compute max simg width across batch
        simg_widths = [s.shape[3] for s in simg_list]
        max_simg_w = max(simg_widths)

        # pad each simg sample to max_simg_w and stack -> (B, N, C, H, W)
        simg_padded = []
        for s in simg_list:
            s_pad = self.pad_to(s, max_simg_w)        # (N, C, H, W)
            simg_padded.append(s_pad.unsqueeze(0))    # (1, N, C, H, W)
        simg_batch = torch.cat(simg_padded, dim=0)     # (B, N, C, H, W)

        # --- handle single real imgs ---
        real_imgs = [b['img'] for b in batch]         # each: (C, H, W_r)
        real_widths = [img.shape[2] for img in real_imgs]
        max_real_w = max(real_widths)

        imgs_padded = []
        for img in real_imgs:
            img_pad = self.pad_to(img, max_real_w)    # (C, H, max_real_w)
            imgs_padded.append(img_pad.unsqueeze(0))  # (1, C, H, W)
        imgs_batch = torch.cat(imgs_padded, dim=0)    # (B, C, H, W)

        # --- swids to (B, N) tensor ---
        swids_list = []
        for b in batch:
            sw = b.get('swids', None)
            if isinstance(sw, list) or isinstance(sw, tuple):
                swids_list.append(torch.tensor(sw, dtype=torch.float32))
            elif torch.is_tensor(sw):
                swids_list.append(sw.float())
            else:
                # fallback: zeros
                swids_list.append(torch.zeros(N, dtype=torch.float32))
        swids_padded = torch.stack(swids_list, dim=0)  # (B, N)

        # --- labels and wcl ---
        labels = [b.get('label', None) for b in batch]
        wcl = torch.tensor([b.get('wcl', -1) for b in batch], dtype=torch.long)

        out = {
            'simg': simg_batch,   # (B,N,C,H,W)
            'swids': swids_padded, # (B,N)
            'img': imgs_batch,     # (B,C,H,W)
            'label': labels,
            'wcl': wcl
        }

        # pass through other keys if present
        extra_keys = set().union(*[set(b.keys()) for b in batch]) - set(out.keys())
        for k in extra_keys:
            out[k] = [b.get(k, None) for b in batch]

        return out