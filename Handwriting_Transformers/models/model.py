# model.py (full, aligned to WordCurveTokenizer expecting [B, n_curves, 8])
import os
import sys
import time
import cv2
import math
import shutil
import random
import unicodedata
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import CTCLoss, MSELoss, L1Loss
from torch.nn.utils import clip_grad_norm_

from params import *
from util.util import (
    toggle_grad, loss_hinge_dis, loss_hinge_gen, ortho, default_ortho,
    prepare_z_y, make_one_hot, to_device, multiple_replace, random_word
)
from .OCR_network import *
from .BigGAN_networks import *
from models.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from models.transformer import *
from models.inception import InceptionV3, calculate_frechet_distance
from data.dataset import TextDataset, TextDatasetval

# === NEW: curve tokenizer bits (your file)
from models.curve_tokenizer import extract_curves_from_word, WordCurveTokenizer

import matplotlib.pyplot as plt

def get_rgb(x):
    R = 255 - int(int(x > 0.5) * 255 * (x - 0.5) / 0.5)
    G = 0
    B = 255 + int(int(x < 0.5) * 255 * (x - 0.5) / 0.5)
    return R, G, B

def get_page_from_words(word_lists, MAX_IMG_WIDTH=800):
    line_all, line_t = [], []
    width_t = 0
    for i in word_lists:
        width_t = width_t + i.shape[1] + 16
        if width_t > MAX_IMG_WIDTH:
            line_all.append(np.concatenate(line_t, 1))
            line_t = []
            width_t = i.shape[1] + 16
        line_t.append(i)
        line_t.append(np.ones((i.shape[0], 16)))
    if len(line_all) == 0:
        line_all.append(np.concatenate(line_t, 1))
    max_lin_widths = MAX_IMG_WIDTH
    gap_h = np.ones([16, max_lin_widths])
    page_ = []
    for l in line_all:
        pad_ = np.ones([l.shape[0], max_lin_widths - l.shape[1]])
        page_.append(np.concatenate([l, pad_], 1))
        page_.append(gap_h)
    page = np.concatenate(page_, 0)
    return page * 255


class FCNDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super().__init__()
        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for _ in range(ups):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type),
            ]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        INP_CHANNEL = NUM_EXAMPLES
        if IS_SEQ:
            INP_CHANNEL = 1

        enc_layer = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT, "relu", True)
        enc_norm = nn.LayerNorm(TN_HIDDEN_DIM)
        self.encoder = TransformerEncoder(enc_layer, TN_ENC_LAYERS, enc_norm)

        dec_layer = TransformerDecoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT, "relu", True)
        dec_norm = nn.LayerNorm(TN_HIDDEN_DIM)
        self.decoder = TransformerDecoder(dec_layer, TN_DEC_LAYERS, dec_norm, return_intermediate=True)

        # feature encoder (resnet18 backbone)
        self.Feat_Encoder = nn.Sequential(
            *([nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)]
              + list(models.resnet18(pretrained=True).children())[1:-2])
        )

        self.query_embed = nn.Embedding(VOCAB_SIZE, TN_HIDDEN_DIM)
        self.linear_q = nn.Linear(TN_DIM_FEEDFORWARD, TN_DIM_FEEDFORWARD * 8)
        self.DEC = FCNDecoder(res_norm='in')

        self._muE = nn.Linear(512, 512)
        self._logvarE = nn.Linear(512, 512)
        self._muD = nn.Linear(512, 512)
        self._logvarD = nn.Linear(512, 512)

        self.l1loss = nn.L1Loss()
        self.noise = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))

        # === curve tokenizer & fusion ===
        self.curve_tokenizer = WordCurveTokenizer().to(DEVICE)
        self.curve_tokenizer.eval()
        self.curve_proj = nn.Linear(128, 512)  # project word-curve emb to 512
        self.fuse_conv = nn.Conv2d(512 + 512, 512, kernel_size=1, stride=1, padding=0)

        self.last_curve_emb = None
        self.L_curve = torch.tensor(0.).to(DEVICE)
        self.curve_loss_fn = nn.MSELoss()

    # ---------- helpers (robust + match tokenizer contract) ----------
    @staticmethod
    def _to_gray_uint8(img_tensor: torch.Tensor) -> np.ndarray:
        """
        img_tensor: [1,H,W] or [H,W] (float), possibly in [-1,1] or [0,1]
        returns: np.uint8 [H,W] in [0,255]
        """
        if img_tensor.dim() == 3:
            img = img_tensor[0]
        else:
            img = img_tensor
        x = img.detach().cpu().numpy()
        if x.max() <= 1.0 and x.min() >= -1.0:
            x = ((x + 1.0) / 2.0) * 255.0
        elif x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    def compute_curve_emb_batch(self, ST, max_curves=100):
        """
        ST: [B,1,H,W] or [B,C,H,W]; returns [B,128] via WordCurveTokenizer.
        Feeds **[B, n_curves, 8]** to tokenizer (variable n_curves <= max_curves). No [*,32,192] reshapes.
        """
        device = ST.device
        if ST.dim() == 4 and ST.shape[1] != 1:
            ST_use = ST[:, :1, :, :]
        else:
            ST_use = ST

        B = ST_use.shape[0]
        embeds = []
        for b in range(B):
            try:
                img_u8 = self._to_gray_uint8(ST_use[b])  # [H,W] uint8
                curves = extract_curves_from_word(img_u8, max_curves=max_curves)  # -> [n_curves, 8] float32
                if curves.ndim != 2 or curves.shape[1] != 8:
                    curves = np.zeros((1, 8), dtype=np.float32)
                curves_t = torch.from_numpy(curves).unsqueeze(0).to(device)  # [1, n_curves, 8]
                with torch.no_grad():
                    emb = self.curve_tokenizer(curves_t)  # [1,128]
                embeds.append(emb)
            except Exception as e:
                print(f"[Warning] curve extraction failed for batch idx {b}: {e}")
                embeds.append(torch.zeros(1, 128, device=device))
        return torch.cat(embeds, dim=0)  # [B,128]
    # -----------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        mu = torch.unbind(mu, 1)
        logvar = torch.unbind(logvar, 1)
        outs = []
        for m, l in zip(mu, logvar):
            sigma = torch.exp(l)
            eps = torch.cuda.FloatTensor(l.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            outs.append(m + sigma * eps)
        return torch.stack(outs, 1)

    def Eval(self, ST, QRS):
        batch_size = ST.shape[0]
        if IS_SEQ:
            B, N, R, C = ST.shape
            FEAT_ST = self.Feat_Encoder(ST.view(B * N, 1, R, C))
            FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
        else:
            FEAT_ST = self.Feat_Encoder(ST)  # [B,512,H,W]

        # curves -> [B,128]
        try:
            curve_emb = self.compute_curve_emb_batch(ST)
        except Exception as e:
            print(f"[Warning] compute_curve_emb_batch failed in Eval: {e}")
            curve_emb = torch.zeros(FEAT_ST.shape[0], 128, device=FEAT_ST.device)
        self.last_curve_emb = curve_emb

        # fuse curve embedding spatially
        curve_proj = self.curve_proj(curve_emb).unsqueeze(-1).unsqueeze(-1)  # [B,512,1,1]
        curve_proj = curve_proj.expand(-1, -1, FEAT_ST.size(2), FEAT_ST.size(3))
        FEAT_ST = self.fuse_conv(torch.cat([FEAT_ST, curve_proj], dim=1))

        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)  # [HW,B,512]
        memory = self.encoder(FEAT_ST_ENC)

        if IS_KLD:
            Ex = memory.permute(1, 0, 2)
            memory_mu = self._muE(Ex)
            memory_logvar = self._logvarE(Ex)
            memory = self.reparameterize(memory_mu, memory_logvar).permute(1, 0, 2)

        OUT_IMGS = []
        for i in range(QRS.shape[1]):
            QR = QRS[:, i, :]
            if ALL_CHARS:
                QR_EMB = self.query_embed.weight.repeat(ST.shape[0], 1, 1).permute(1, 0, 2)
            else:
                QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)
            tgt = torch.zeros_like(QR_EMB)
            hs = self.decoder(tgt, memory, query_pos=QR_EMB)

            if IS_KLD:
                Dx = hs[0].permute(1, 0, 2)
                hs_mu = self._muD(Dx)
                hs_logvar = self._logvarD(Dx)
                hs = self.reparameterize(hs_mu, hs_logvar).permute(1, 0, 2).unsqueeze(0)

            h = hs.transpose(1, 2)[-1]
            if ADD_NOISE:
                h = h + self.noise.sample(h.size()).squeeze(-1).to(DEVICE)
            h = self.linear_q(h).contiguous()
            if ALL_CHARS:
                h = torch.stack([h[i][QR[i]] for i in range(batch_size)], 0)
            h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
            h = h.permute(0, 3, 2, 1)
            h = self.DEC(h)
            OUT_IMGS.append(h.detach())

        return OUT_IMGS

    def forward(self, ST, QR, QRs=None, mode='train'):
        # --- attention capture hooks
        enc_attn_weights, dec_attn_weights = [], []
        self.hooks = [
            self.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self_, inp, out: enc_attn_weights.append(out[1])
            ),
            self.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self_, inp, out: dec_attn_weights.append(out[1])
            ),
        ]

        B, N, R, C = ST.shape
        FEAT_ST = self.Feat_Encoder(ST.view(B * N, 1, R, C))
        FEAT_ST = FEAT_ST.view(B, 512, 1, -1)

        # curves -> [B,128]
        try:
            curve_emb = self.compute_curve_emb_batch(ST)
        except Exception as e:
            print(f"[Warning] compute_curve_emb_batch failed in forward: {e}")
            curve_emb = torch.zeros(B, 128, device=FEAT_ST.device)
        self.last_curve_emb = curve_emb

        # fuse curve embedding
        curve_proj = self.curve_proj(curve_emb).unsqueeze(-1).unsqueeze(-1)  # [B,512,1,1]
        curve_proj = curve_proj.expand(-1, -1, FEAT_ST.size(2), FEAT_ST.size(3))
        FEAT_ST = self.fuse_conv(torch.cat([FEAT_ST, curve_proj], dim=1))

        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)
        memory = self.encoder(FEAT_ST_ENC)

        QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)
        tgt = torch.zeros_like(QR_EMB)
        hs = self.decoder(tgt, memory, query_pos=QR_EMB)

        h = hs.transpose(1, 2)[-1]
        if ADD_NOISE:
            h = h + self.noise.sample(h.size()).squeeze(-1).to(DEVICE)
        h = self.linear_q(h).contiguous()
        h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
        h = h.permute(0, 3, 2, 1)
        h = self.DEC(h)

        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()

        # === curve-consistency loss (optional)
        try:
            gen_curve_embs = []
            for b in range(h.size(0)):
                gen_img = h[b, 0].detach().cpu().numpy()
                if gen_img.max() <= 1.0 and gen_img.min() >= -1.0:
                    gen_vis = ((gen_img + 1.0) / 2.0) * 255.0
                else:
                    gen_vis = gen_img * 255.0
                gen_vis = np.clip(gen_vis, 0, 255).astype(np.uint8)
                try:
                    curves = extract_curves_from_word(gen_vis, max_curves=100)  # -> [n,8]
                    if curves.ndim != 2 or curves.shape[1] != 8:
                        curves = np.zeros((1, 8), dtype=np.float32)
                    curves_t = torch.from_numpy(curves).unsqueeze(0).to(h.device)  # [1,n,8]
                    with torch.no_grad():
                        emb_fake = self.curve_tokenizer(curves_t)  # [1,128]
                    gen_curve_embs.append(emb_fake)
                except Exception:
                    gen_curve_embs.append(torch.zeros(1, 128, device=h.device))
            if len(gen_curve_embs) == 0:
                self.L_curve = torch.tensor(0.).to(h.device)
            else:
                gen_curve_embs = torch.cat(gen_curve_embs, dim=0).to(h.device)  # [B,128]
                if self.last_curve_emb is not None:
                    try:
                        self.L_curve = self.curve_loss_fn(gen_curve_embs, self.last_curve_emb.detach())
                    except Exception as e:
                        print(f"[Warning] curve loss computation failed: {e}")
                        self.L_curve = torch.tensor(0.).to(h.device)
                else:
                    self.L_curve = torch.tensor(0.).to(h.device)
        except Exception as e:
            print(f"[Warning] overall gen curve extraction failed: {e}")
            self.L_curve = torch.tensor(0.).to(h.device)

        for hook in self.hooks:
            hook.remove()
        return h


class TRGAN(nn.Module):
    def __init__(self, batch_size=batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.epsilon = 1e-7

        self.netG = Generator().to(DEVICE)
        self.netD = nn.DataParallel(Discriminator()).to(DEVICE)
        self.netW = nn.DataParallel(WDiscriminator()).to(DEVICE)
        self.netconverter = strLabelConverter(ALPHABET)
        self.netOCR = CRNN().to(DEVICE)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction='none')

        self.lambda_curve = 0.1  # weight for curve-consistency loss

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(DEVICE)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=G_LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)
        self.optimizer_OCR = torch.optim.Adam(self.netOCR.parameters(), lr=OCR_LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=D_LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)
        self.optimizer_wl = torch.optim.Adam(self.netW.parameters(), lr=W_LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizers = [self.optimizer_G, self.optimizer_OCR, self.optimizer_D, self.optimizer_wl]
        for opt in self.optimizers:
            opt.zero_grad()

        # loss bookkeeping
        self.loss_G = 0
        self.loss_D = 0
        self.loss_Dfake = 0
        self.loss_Dreal = 0
        self.loss_OCR_fake = 0
        self.loss_OCR_real = 0
        self.loss_w_fake = 0
        self.loss_w_real = 0
        self.Lcycle1 = 0
        self.Lcycle2 = 0
        self.lda1 = 0
        self.lda2 = 0
        self.KLD = 0

        with open(ENGLISH_WORDS_PATH, 'rb') as f:
            self.lex = f.read().splitlines()
        lex = []
        for word in self.lex:
            try:
                word = word.decode("utf-8")
            except:
                continue
            if len(word) < 20:
                lex.append(word)
        self.lex = lex

        f = open('mytext.txt', 'r')
        self.text = [j.encode() for j in sum([i.split(' ') for i in f.readlines()], [])]
        self.eval_text_encode, self.eval_len_text = self.netconverter.encode(self.text)
        self.eval_text_encode = self.eval_text_encode.to(DEVICE).repeat(self.batch_size, 1, 1)

    # ------------- utility: page/FID saving (unchanged) -------------
    def save_images_for_fid_calculation(self, dataloader, epoch, mode='train'):
        self.real_base = os.path.join('saved_images', EXP_NAME, 'Real')
        self.fake_base = os.path.join('saved_images', EXP_NAME, 'Fake')
        if os.path.isdir(self.real_base): shutil.rmtree(self.real_base)
        if os.path.isdir(self.fake_base): shutil.rmtree(self.fake_base)
        os.mkdir(self.real_base); os.mkdir(self.fake_base)

        for step, data in enumerate(dataloader):
            ST = data['simg'].cuda()
            self.fakes = self.netG.Eval(ST, self.eval_text_encode)
            fake_images = torch.cat(self.fakes, 1).detach().cpu().numpy()
            for i in range(fake_images.shape[0]):
                for j in range(fake_images.shape[1]):
                    cv2.imwrite(os.path.join(self.fake_base, f"{step*self.batch_size + i}_{j}.png"), 255 * (fake_images[i, j]))

        if mode == 'train':
            TextDatasetObj = TextDataset(num_examples=self.eval_text_encode.shape[1])
            dataset_real = torch.utils.data.DataLoader(
                TextDatasetObj, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObj.collate_fn
            )
        else:
            TextDatasetObjval = TextDatasetval(num_examples=self.eval_text_encode.shape[1])
            dataset_real = torch.utils.data.DataLoader(
                TextDatasetObjval, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObjval.collate_fn
            )

        for step, data in enumerate(dataset_real):
            real_images = data['simg'].numpy()
            for i in range(real_images.shape[0]):
                for j in range(real_images.shape[1]):
                    cv2.imwrite(os.path.join(self.real_base, f"{step*self.batch_size + i}_{j}.png"), 255 * (real_images[i, j]))
        return self.real_base, self.fake_base
    # ---------------------------------------------------------------

    # ------------- (the rest of your training loop methods) -------------
    # All methods below match your original file, except we add curve loss where noted.

    def _generate_page(self, ST, SLEN, eval_text_encode=None, eval_len_text=None):
        if eval_text_encode is None: eval_text_encode = self.eval_text_encode
        if eval_len_text is None: eval_len_text = self.eval_len_text
        self.fakes = self.netG.Eval(ST, eval_text_encode)
        page1s, page2s = [], []
        for batch_idx in range(self.batch_size):
            word_t, word_l, line_wids = [], [], []
            gap = np.ones([IMG_HEIGHT, 16])
            for idx, fake_ in enumerate(self.fakes):
                word_t.append((fake_[batch_idx, 0, :, :eval_len_text[idx] * resolution].cpu().numpy() + 1) / 2)
                word_t.append(gap)
                if len(word_t) == 16 or idx == len(self.fakes) - 1:
                    line_ = np.concatenate(word_t, -1)
                    word_l.append(line_); line_wids.append(line_.shape[1]); word_t = []
            gap_h = np.ones([16, max(line_wids)])
            page_ = []
            for l in word_l:
                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])
                page_.append(np.concatenate([l, pad_], 1)); page_.append(gap_h)
            page1 = np.concatenate(page_, 0)

            word_t, word_l, line_wids = [], [], []
            sdata_ = [i.unsqueeze(1) for i in torch.unbind(ST, 1)]
            for idx, st in enumerate(sdata_):
                word_t.append((st[batch_idx, 0, :, :int(SLEN.cpu().numpy()[batch_idx][idx])].cpu().numpy() + 1) / 2)
                word_t.append(gap)
                if len(word_t) == 16 or idx == len(sdata_) - 1:
                    line_ = np.concatenate(word_t, -1)
                    word_l.append(line_); line_wids.append(line_.shape[1]); word_t = []
            gap_h = np.ones([16, max(line_wids)])
            page_ = []
            for l in word_l:
                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])
                page_.append(np.concatenate([l, pad_], 1)); page_.append(gap_h)
            page2 = np.concatenate(page_, 0)
            merge_w_size = max(page1.shape[0], page2.shape[0])
            if page1.shape[0] != merge_w_size:
                page1 = np.concatenate([page1, np.ones([merge_w_size - page1.shape[0], page1.shape[1]])], 0)
            if page2.shape[0] != merge_w_size:
                page2 = np.concatenate([page2, np.ones([merge_w_size - page2.shape[0], page2.shape[1]])], 0)
            page1s.append(page1); page2s.append(page2)
        page1s_ = np.concatenate(page1s, 0)
        max_wid = max([i.shape[1] for i in page2s])
        padded_page2s = [np.concatenate([para, np.ones([para.shape[0], max_wid - para.shape[1]])], 1) for para in page2s]
        padded_page2s_ = np.concatenate(padded_page2s, 0)
        return np.concatenate([padded_page2s_, page1s_], 1)

    def get_current_losses(self):
        return {
            'G': self.loss_G, 'D': self.loss_D, 'Dfake': self.loss_Dfake, 'Dreal': self.loss_Dreal,
            'OCR_fake': self.loss_OCR_fake, 'OCR_real': self.loss_OCR_real,
            'w_fake': self.loss_w_fake, 'w_real': self.loss_w_real,
            'cycle1': self.Lcycle1, 'cycle2': self.Lcycle2, 'lda1': self.lda1, 'lda2': self.lda2, 'KLD': self.KLD
        }

    # ---------- the rest of your optimization methods (unchanged) ----------
    # NOTE: only modifications: we add + self.lambda_curve * self.netG.L_curve into generator losses.

    def load_networks(self, epoch):
        BaseModel.load_networks(self, epoch)
        if self.opt.single_writer:
            load_filename = '%s_z.pkl' % (epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            self.z = torch.load(load_path)

    def _set_input(self, input):
        self.input = input

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for p in net.parameters():
                    p.requires_grad = requires_grad

    def forward(self):
        self.real = self.input['img'].to(DEVICE)
        self.label = self.input['label']
        self.sdata = self.input['simg'].to(DEVICE)
        self.ST_LEN = self.input['swids']
        self.text_encode, self.len_text = self.netconverter.encode(self.label)
        self.one_hot_real = make_one_hot(self.text_encode, self.len_text, VOCAB_SIZE).to(DEVICE).detach()
        self.text_encode = self.text_encode.to(DEVICE).detach()
        self.len_text = self.len_text.detach()

        self.words = [word.encode('utf-8') for word in np.random.choice(self.lex, self.batch_size)]
        self.text_encode_fake, self.len_text_fake = self.netconverter.encode(self.words)
        self.text_encode_fake = self.text_encode_fake.to(DEVICE)
        self.one_hot_fake = make_one_hot(self.text_encode_fake, self.len_text_fake, VOCAB_SIZE).to(DEVICE)

        self.text_encode_fake_js = []
        for _ in range(NUM_WORDS - 1):
            self.words_j = [word.encode('utf-8') for word in np.random.choice(self.lex, self.batch_size)]
            self.text_encode_fake_j, self.len_text_fake_j = self.netconverter.encode(self.words_j)
            self.text_encode_fake_j = self.text_encode_fake_j.to(DEVICE)
            self.text_encode_fake_js.append(self.text_encode_fake_j)

        self.fake = self.netG(self.sdata, self.text_encode_fake, self.text_encode_fake_js)

    def backward_D_OCR(self):
        pred_real = self.netD(self.real.detach())
        pred_fake = self.netD(**{'x': self.fake.detach()})
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), True)
        self.loss_D = self.loss_Dreal + self.loss_Dfake

        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        loss_total = self.loss_D + self.loss_OCR_real
        loss_total.backward()
        for p in self.netOCR.parameters():
            if p.grad is not None:
                p.grad[p.grad != p.grad] = 0
                p.grad[torch.isnan(p.grad)] = 0
                p.grad[torch.isinf(p.grad)] = 0
        return loss_total

    def backward_D_WL(self):
        pred_real = self.netD(self.real.detach())
        pred_fake = self.netD(**{'x': self.fake.detach()})
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), True)
        self.loss_D = self.loss_Dreal + self.loss_Dfake
        self.loss_w_real = self.netW(self.real.detach(), self.input['wcl'].to(DEVICE)).mean()
        loss_total = self.loss_D + self.loss_w_real
        loss_total.backward()
        return loss_total

    def optimize_D_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], True)
        self.optimizer_D.zero_grad(); self.optimizer_wl.zero_grad()
        self.backward_D_WL()

    def backward_D_OCR_WL(self):
        if getattr(self, "real_z_mean", None) is None:
            pred_real = self.netD(self.real.detach())
        else:
            pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
        pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
        self.loss_D = self.loss_Dreal + self.loss_Dfake
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])
        self.loss_w_real = self.netW(self.real.detach(), self.wcl)
        loss_total = self.loss_D + self.loss_OCR_real + self.loss_w_real
        loss_total.backward()
        for p in self.netOCR.parameters():
            if p.grad is not None:
                p.grad[p.grad != p.grad] = 0
                p.grad[torch.isnan(p.grad)] = 0
                p.grad[torch.isinf(p.grad)] = 0
        return loss_total

    def optimize_D_WL_step(self):
        self.optimizer_D.step(); self.optimizer_wl.step()
        self.optimizer_D.zero_grad(); self.optimizer_wl.zero_grad()

    def backward_OCR(self):
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])
        self.loss_OCR_real.backward()
        for p in self.netOCR.parameters():
            if p.grad is not None:
                p.grad[p.grad != p.grad] = 0
                p.grad[torch.isnan(p.grad)] = 0
                p.grad[torch.isinf(p.grad)] = 0
        return self.loss_OCR_real

    def backward_D(self):
        if getattr(self, "real_z_mean", None) is None:
            pred_real = self.netD(self.real.detach())
        else:
            pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
        pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
        self.loss_D = self.loss_Dreal + self.loss_Dfake
        self.loss_D.backward()
        return self.loss_D

    def backward_G_only(self):
        self.gb_alpha = 0.7
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()
        pred_fake_OCR = self.netOCR(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size, self.len_text_fake.detach())
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])
        self.loss_G = self.loss_G + self.Lcycle1 + self.Lcycle2 + self.lda1 + self.lda2 - self.KLD
        # add curve-consistency
        self.loss_G = self.loss_G + self.lambda_curve * getattr(self.netG, 'L_curve', torch.tensor(0.).to(DEVICE))
        self.loss_T = self.loss_G + self.loss_OCR_fake

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_OCR = 10**6 * torch.mean(grad_fake_OCR**2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10**6 * torch.mean(grad_fake_adv**2)

        self.loss_T.backward(retain_graph=True)
        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
        a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR))
        if a is None:
            print(self.loss_OCR_fake, self.loss_G, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
        if a > 1000 or a < 0.0001:
            print(a)
        self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
        self.loss_T = self.loss_G + self.loss_OCR_fake
        self.loss_T.backward(retain_graph=True)
        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
        with torch.no_grad():
            self.loss_T.backward()
        if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G):
            print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
            sys.exit()

    def backward_G_WL(self):
        self.gb_alpha = 0.7
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()
        self.loss_w_fake = self.netW(self.fake, self.input['wcl'].to(DEVICE)).mean()
        self.loss_G = self.loss_G + self.Lcycle1 + self.Lcycle2 + self.lda1 + self.lda2 - self.KLD
        # add curve-consistency
        self.loss_G = self.loss_G + self.lambda_curve * getattr(self.netG, 'L_curve', torch.tensor(0.).to(DEVICE))
        self.loss_T = self.loss_G + self.loss_w_fake

        self.loss_T.backward(retain_graph=True)
        grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
        a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_WL))
        if a is None:
            print(self.loss_w_fake, self.loss_G, torch.std(grad_fake_adv), torch.std(grad_fake_WL))
        if a > 1000 or a < 0.0001:
            print(a)
        self.loss_w_fake = a.detach() * self.loss_w_fake
        self.loss_T = self.loss_G + self.loss_w_fake
        self.loss_T.backward(retain_graph=True)
        grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=False, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_WL = 10 ** 6 * torch.mean(grad_fake_WL ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
        with torch.no_grad():
            self.loss_T.backward()

    def backward_G(self):
        self.opt.gb_alpha = 0.7
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake, 'z': self.z}), self.len_text_fake.detach(), self.opt.mask_loss)
        pred_fake_OCR = self.netOCR(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.opt.batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size, self.len_text_fake.detach())
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])
        self.loss_w_fake = self.netW(self.fake, self.wcl)
        self.loss_G_ = 10 * self.loss_G + self.loss_w_fake
        # add curve-consistency
        self.loss_G_ = self.loss_G_ + self.lambda_curve * getattr(self.netG, 'L_curve', torch.tensor(0.).to(DEVICE))
        self.loss_T = self.loss_G_ + self.loss_OCR_fake

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_OCR = 10**6 * torch.mean(grad_fake_OCR**2)
        grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10**6 * torch.mean(grad_fake_adv**2)

        if not False:
            self.loss_T.backward(retain_graph=True)
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=True, retain_graph=True)[0]
            a = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR))
            if a is None:
                print(self.loss_OCR_fake, self.loss_G_, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
            if a > 1000 or a < 0.0001:
                print(a)
            self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
            self.loss_T = (1 - 1 * self.opt.onlyOCR) * self.loss_G_ + self.loss_OCR_fake
            self.loss_T.backward(retain_graph=True)
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
            self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            with torch.no_grad():
                self.loss_T.backward()
        else:
            self.loss_T.backward()

        if self.opt.clip_grad > 0:
            clip_grad_norm_(self.netG.parameters(), self.opt.clip_grad)
        if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G_):
            print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
            sys.exit()

    def optimize_D_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.backward_D_OCR()

    def optimize_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], True)
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_OCR()

    def optimize_D(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.backward_D()

    def optimize_D_OCR_step(self):
        self.optimizer_D.step()
        self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()

    def optimize_D_OCR_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.set_requires_grad([self.netW], True)
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_D_OCR_WL()

    def optimize_D_OCR_WL_step(self):
        self.optimizer_D.step()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.step()
        self.optimizer_wl.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_wl.zero_grad()

    def optimize_D_step(self):
        self.optimizer_D.step()
        if any(torch.isnan(self.netD.infer_img.blocks[0][0].conv1.bias)):
            print('D is nan')
            sys.exit()
        self.optimizer_D.zero_grad()

    def optimize_G(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G()

    def optimize_G_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_WL()

    def optimize_G_only(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_only()

    def optimize_G_step(self):
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

    def optimize_ocr(self):
        self.set_requires_grad([self.netOCR], True)
        pred_real_OCR = self.netOCR(self.real)
        preds_size = torch.IntTensor([pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        self.loss_OCR_real = self.OCR_criterion(pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real.backward()
        self.optimizer_OCR.step()

    def optimize_z(self):
        self.set_requires_grad([self.z], True)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def test(self):
        self.visual_names = ['fake']
        self.netG.eval()
        with torch.no_grad():
            self.forward()

    def train_GD(self):
        self.netG.train(); self.netD.train()
        self.optimizer_G.zero_grad(); self.optimizer_D.zero_grad()
        x = torch.split(self.real, self.opt.batch_size)
        y = torch.split(self.label, self.opt.batch_size)
        counter = 0

        if self.opt.toggle_grads:
            toggle_grad(self.netD, True)
            toggle_grad(self.netG, False)

        for _ in range(self.opt.num_critic_train):
            self.optimizer_D.zero_grad()
            with torch.set_grad_enabled(False):
                self.forward()
            D_input = torch.cat([self.fake, x[counter]], 0) if x is not None else self.fake
            D_class = torch.cat([self.label_fake, y[counter]], 0) if y[counter] is not None else y[counter]
            D_out = self.netD(D_input, D_class)
            if x is not None:
                pred_fake, pred_real = torch.split(D_out, [self.fake.shape[0], x[counter].shape[0]])
            else:
                pred_fake = D_out
            self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
            self.loss_D = self.loss_Dreal + self.loss_Dfake
            self.loss_D.backward()
            counter += 1
            self.optimizer_D.step()

        if self.opt.toggle_grads:
            toggle_grad(self.netD, False)
            toggle_grad(self.netG, True)

        self.optimizer_G.zero_grad()
        self.forward()
        self.loss_G = loss_hinge_gen(self.netD(self.fake, self.label_fake), self.len_text_fake.detach(), self.opt.mask_loss)
        self.loss_G.backward()
        self.optimizer_G.step()

    def save_networks(self, epoch, save_dir):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if len(self.gpu_ids) > 1:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
