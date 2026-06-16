import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .heatmap import normalize_map


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def mask_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def prepare_refiner_input(image_path, heatmap, img_size=384, input_mode="rgb_heat"):
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0
    h0, w0 = image_np.shape[:2]
    heat = cv2.resize(normalize_map(heatmap), (w0, h0), interpolation=cv2.INTER_CUBIC)
    image_np = cv2.resize(image_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    heat = cv2.resize(heat, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    image_t = torch.from_numpy(image_np).permute(2, 0, 1).float()
    heat_t = torch.from_numpy(heat[None]).float()
    if input_mode == "rgb_heat":
        x = torch.cat([image_t, heat_t], dim=0)
    elif input_mode == "heat_only":
        x = heat_t
    elif input_mode == "rgb_only":
        x = image_t
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return x.unsqueeze(0), image.size


@torch.no_grad()
def predict_mask(model, image_path, heatmap, img_size=384, input_mode="rgb_heat", device="cuda"):
    x, (w0, h0) = prepare_refiner_input(image_path, heatmap, img_size, input_mode)
    model = model.to(device).eval()
    prob = torch.sigmoid(model(x.to(device)))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_CUBIC)
