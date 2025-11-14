# util/page_renderer.py
import numpy as np
from PIL import Image

def to_numpy_img(tensor):
    """
    tensor: torch.Tensor or numpy array in [0,1] or [-1,1], shape (C,H,W) or (H,W) or (B,C,H,W)
    Return numpy float32 image in [0,255] shape (H,W) grayscale (uint8).
    """
    import torch
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)
    # if batch, take first
    if arr.ndim == 4:
        arr = arr[0]
    # shape (C,H,W) -> (H,W)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            # rgb -> convert to grayscale
            arr = np.mean(arr, axis=0)
    # now arr is (H,W)
    # if in [-1,1], scale
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr

def pad_horiz(img_arr, width, fill=255):
    H, W = img_arr.shape
    if W >= width:
        return img_arr[:, :width]
    pad = np.ones((H, width - W), dtype=img_arr.dtype) * fill
    return np.concatenate([img_arr, pad], axis=1)

def assemble_page_from_word_images(word_images, max_line_width=1024, gap_w=16, gap_h=16, bg=255):
    """
    word_images: list of numpy arrays (H,W) or list-of-lists per line. We'll pack them greedily.
    Returns: final page numpy array (H_page, W_page) uint8 grayscale, background=bg (255 white).
    Behavior:
      - Packs words left to right with gap_w between words until max_line_width reached, then new line.
      - Keeps constant row height = max height among words in that line.
    """
    # Flatten inputs if nested
    if len(word_images) == 0:
        return np.ones((32, max_line_width), dtype=np.uint8) * bg

    # Ensure array format
    imgs = [np.array(im) for im in word_images]

    lines = []
    cur_line = []
    cur_width = 0
    cur_max_h = 0

    for im in imgs:
        h, w = im.shape
        # If this single image larger than max_line_width, trim/pad
        if w > max_line_width:
            # slice horizontally
            im = im[:, :max_line_width]
            h, w = im.shape
        # if adding this word exceeds width -> start new line
        if cur_width == 0:
            new_width = w
        else:
            new_width = cur_width + gap_w + w
        if new_width <= max_line_width:
            cur_line.append(im)
            cur_width = new_width
            cur_max_h = max(cur_max_h, h)
        else:
            # finalize current line
            if len(cur_line) == 0:
                # single oversized image: put it alone (trimmed earlier)
                lines.append(( [im], max(h, cur_max_h) ))
                cur_line = []
                cur_width = 0
                cur_max_h = 0
            else:
                lines.append((cur_line, cur_max_h))
                cur_line = [im]
                cur_width = w
                cur_max_h = h
    if len(cur_line) > 0:
        lines.append((cur_line, cur_max_h))

    # Build each line into an image
    line_images = []
    actual_width = 0
    for line_imgs, line_h in lines:
        # normalize heights by padding vertically centered
        padded_imgs = []
        widths = []
        for im in line_imgs:
            h,w = im.shape
            if h < line_h:
                pad_top = (line_h - h) // 2
                pad_bot = line_h - h - pad_top
                top = np.ones((pad_top, w), dtype=im.dtype) * bg
                bot = np.ones((pad_bot, w), dtype=im.dtype) * bg
                im2 = np.concatenate([top, im, bot], axis=0)
            else:
                im2 = im
            padded_imgs.append(im2)
            widths.append(im2.shape[1])
        # join with gap
        if len(padded_imgs) == 0:
            continue
        row = padded_imgs[0]
        for idx in range(1, len(padded_imgs)):
            gap = np.ones((line_h, gap_w), dtype=row.dtype) * bg
            row = np.concatenate([row, gap, padded_imgs[idx]], axis=1)
        actual_width = max(actual_width, row.shape[1])
        line_images.append(row)

    # make final page width = max_line_width but we can also shrink to actual_width
    page_width = max_line_width
    # pad lines to same width
    padded_lines = []
    for li in line_images:
        if li.shape[1] < page_width:
            pad = np.ones((li.shape[0], page_width - li.shape[1]), dtype=li.dtype) * bg
            li2 = np.concatenate([li, pad], axis=1)
        else:
            li2 = li[:, :page_width]
        padded_lines.append(li2)

    # join lines vertically with gap_h
    if len(padded_lines) == 0:
        H = 32
        return np.ones((H, page_width), dtype=np.uint8) * bg

    page = padded_lines[0]
    gap_h_arr = np.ones((gap_h, page_width), dtype=np.uint8) * bg
    for idx in range(1, len(padded_lines)):
        page = np.concatenate([page, gap_h_arr, padded_lines[idx]], axis=0)

    return page