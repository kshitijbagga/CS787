import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure, transform, img_as_ubyte
import io
import os

def imread_gray(path):
    """Read image as grayscale numpy array (H,W) uint8"""
    img = Image.open(path).convert("L")
    return np.array(img)

def imdecode_gray(imageBuf):
    """Decode image bytes buffer to grayscale numpy array"""
    img = Image.open(io.BytesIO(imageBuf)).convert("L")
    return np.array(img)

def imwrite(path, img_array):
    """Write numpy array or PIL image to path. Expects uint8 0-255."""
    if isinstance(img_array, np.ndarray):
        if img_array.dtype != 'uint8':
            img_array = img_as_ubyte(img_array)
        img = Image.fromarray(img_array)
    else:
        img = img_array
    # ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def resize(img_array_or_pil, dsize=None, fx=None, fy=None):
    """Resize given numpy array or PIL Image. If dsize None, use fx,fy scaling factors"""
    if isinstance(img_array_or_pil, np.ndarray):
        pil = Image.fromarray(img_array_or_pil)
    else:
        pil = img_array_or_pil
    if dsize is not None:
        w,h = dsize
        out = pil.resize((w,h), Image.LANCZOS)
    else:
        w = int(pil.width * fx) if fx is not None else pil.width
        h = int(pil.height * fy) if fy is not None else pil.height
        out = pil.resize((w,h), Image.LANCZOS)
    return np.array(out)

def addWeighted(img1, alpha, img2, beta, gamma=0):
    """Blend two numpy images (H,W,C) or (H,W)"""
    a = np.array(img1).astype(np.float32)
    b = np.array(img2).astype(np.float32)
    res = a * alpha + b * beta + gamma
    res = np.clip(res, 0, 255).astype('uint8')
    return res

def getTextSize(text, font_path=None, font_scale=1, thickness=1):
    """Approximate cv2.getTextSize using PIL ImageFont"""
    # font_scale corresponds to pixel size; default to 20*font_scale
    size = int(20 * font_scale)
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new("L", (1,1))
    draw = ImageDraw.Draw(dummy)
    textsize = draw.textsize(text, font=font)
    return textsize, None  # cv2 returns (width,height), baseline

def putText(img_array, text, org, font_path=None, font_scale=1, color=(255,255,255), thickness=1):
    """Draw text onto numpy image (H,W) gray or (H,W,3) color. org is (x,y)"""
    if isinstance(img_array, np.ndarray):
        mode = 'RGB' if img_array.ndim==3 else 'L'
        pil = Image.fromarray(img_array) if img_array.dtype=='uint8' else Image.fromarray(img_as_ubyte(img_array))
        if mode=='L':
            pil = pil.convert("RGB")
    else:
        pil = img_array.convert("RGB")
    try:
        size = int(20 * font_scale)
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(pil)
    draw.text(org, text, fill=color, font=font)
    return np.array(pil)

def connectedComponents(seg_mask, connectivity=4):
    """Mimic cv2.connectedComponents. seg_mask is a binary mask (H,W)"""
    # skimage.measure.label uses connectivity: 1 or 2 for 2D; set background=0
    labeled = measure.label(seg_mask, connectivity=1)  # connectivity=1 corresponds to 4-connectivity
    # labels are 0..N; number of labels equals max+1 (including background 0)
    num = int(labeled.max()) + 1
    return num, labeled.astype('int32')
