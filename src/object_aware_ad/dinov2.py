from pathlib import Path
import hashlib

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_dinov2(ckpt_path, img_size, device):
    model = timm.create_model(
        "vit_base_patch14_dinov2",
        pretrained=False,
        img_size=img_size,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

    cleaned = {}
    for k, v in ckpt.items():
        k = k.replace("module.", "")
        k = k.replace("backbone.", "")
        k = k.replace("teacher.", "")
        cleaned[k] = v

    msg = model.load_state_dict(cleaned, strict=False)
    print("[Load DINOv2]", msg)
    model.to(device)
    model.eval()
    return model


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


@torch.no_grad()
def extract_patch_tokens(model, img_path, transform, device, img_size, patch_size):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    feats = model.forward_features(x)

    if isinstance(feats, dict):
        if "x_norm_patchtokens" in feats:
            patch = feats["x_norm_patchtokens"]
        elif "x" in feats:
            patch = feats["x"][:, 1:, :]
        else:
            raise RuntimeError(f"Unknown DINOv2 feature keys: {list(feats.keys())}")
    else:
        patch = feats[:, 1:, :]

    patch = F.normalize(patch.squeeze(0), dim=-1)
    grid_hw = (img_size // patch_size, img_size // patch_size)
    return patch.cpu(), grid_hw


def cache_key_for_path(path):
    p = Path(path)
    digest = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:16]
    return f"{p.stem}_{digest}.pt"


def get_or_extract_patch_tokens(record, model, transform, config, device):
    cache_dir = Path(config["feature_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / cache_key_for_path(record["resolved_crop_path"])
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["patch"], tuple(payload["grid_hw"])

    patch, grid_hw = extract_patch_tokens(
        model=model,
        img_path=record["resolved_crop_path"],
        transform=transform,
        device=device,
        img_size=int(config["img_size"]),
        patch_size=int(config["patch_size"]),
    )
    torch.save({"patch": patch, "grid_hw": grid_hw}, cache_path)
    return patch, grid_hw
