"""Original SmallUNet topology with a fixed revision base width of 16."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class FirstFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = DoubleConv(3, 8)
        self.heat = DoubleConv(1, 8)

    def forward(self, x):
        return torch.cat([self.rgb(x[:, :3]), self.heat(x[:, 3:])], dim=1)


class SmallUNet(nn.Module):
    def __init__(self, mode="rgb_heat"):
        super().__init__()
        in_ch = {"rgb_heat": 4, "rgb_only": 3, "heat_only": 1,
                 "rgb_zero": 4, "rgb_shuffle": 4, "feature_fusion": 4}[mode]
        base = 16
        self.enc1 = FirstFusion() if mode == "feature_fusion" else DoubleConv(in_ch, base)
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
        return self.out(self.dec1(torch.cat([self.up1(d2), e1], dim=1)))


def total_loss(logits, target, kind="bce_dice", pos_weight=None):
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()
    if kind == "focal_dice":
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = prob * target + (1 - prob) * (1 - target)
        alpha = 0.25 * target + 0.75 * (1 - target)
        return (alpha * (1 - pt).pow(2) * ce).mean() + dice
    if kind not in ("bce_dice", "weighted_bce_dice"):
        raise ValueError(kind)
    weight = None
    if kind == "weighted_bce_dice":
        if pos_weight is None or pos_weight <= 0:
            raise ValueError("Training-only positive weight required")
        weight = logits.new_tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight) + dice
