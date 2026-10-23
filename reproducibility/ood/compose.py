"""Paste an isolated RGBA object into a user-supplied background/component box."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background', required=True)
    parser.add_argument('--asset', required=True)
    parser.add_argument('--part-box', type=float, nargs=4, required=True,
                        metavar=('X1', 'Y1', 'X2', 'Y2'))
    parser.add_argument('--area-fraction', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1400)
    parser.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args()
    image = Image.open(args.background).convert('RGB')
    asset = Image.open(args.asset).convert('RGBA')
    x1, y1, x2, y2 = args.part_box
    pw, ph = x2-x1, y2-y1
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        parser.error('Component box must lie inside the background')
    if not (0 < args.area_fraction <= 1) or pw < 4 or ph < 4:
        parser.error('Invalid area fraction or component size')
    if not asset.getchannel('A').getbbox():
        parser.error('Asset alpha is empty')
    rng = np.random.default_rng(args.seed)
    angle = float(rng.uniform(-25, 25))
    asset = asset.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    asset = asset.crop(asset.getchannel('A').getbbox())
    scale = min(np.sqrt(pw*ph*args.area_fraction/(asset.width*asset.height)),
                pw*.75/asset.width, ph*.75/asset.height)
    asset = asset.resize((max(2, round(asset.width*scale)),
                          max(2, round(asset.height*scale))), Image.Resampling.LANCZOS)
    cx = x1+pw*(.5+float(rng.uniform(-.13, .13)))
    cy = y1+ph*(.5+float(rng.uniform(-.13, .13)))
    x = max(int(np.ceil(x1)), min(int(cx-asset.width/2), int(np.floor(x2))-asset.width))
    y = max(int(np.ceil(y1)), min(int(cy-asset.height/2), int(np.floor(y2))-asset.height))
    arr = np.asarray(asset).copy()
    foreground = arr[:, :, 3] > 0
    patch = np.asarray(image)[y:y+asset.height, x:x+asset.width]
    gain = float(np.clip(np.mean(patch)/max(1, np.mean(arr[:, :, :3][foreground])), .65, 1.05))
    arr[:, :, :3] = np.clip(arr[:, :, :3].astype(float)*gain, 0, 255).astype(np.uint8)
    asset = Image.fromarray(arr)
    canvas = image.copy()
    canvas.paste(asset, (x, y), asset.getchannel('A'))
    mask = Image.new('L', image.size)
    mask.paste(asset.getchannel('A').point(lambda v: 255 if v else 0), (x, y))
    box = list(mask.getbbox())
    outside = np.asarray(mask) == 0
    assert np.array_equal(np.asarray(canvas)[outside], np.asarray(image)[outside])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    canvas.save(out/'composite.png')
    mask.save(out/'mask.png')
    w, h = image.size
    values = [(box[0]+box[2])/2/w, (box[1]+box[3])/2/h,
              (box[2]-box[0])/w, (box[3]-box[1])/h]
    (out/'label.txt').write_text('0 '+' '.join(f'{v:.10f}' for v in values)+'\n')
    metadata = dict(asset_id=Path(args.asset).stem, seed=args.seed,
                    part_bbox_crop_xyxy=args.part_box, target_box_part_fraction=args.area_fraction,
                    rotation_degrees=angle, brightness_gain=gain, position_xy=[x, y],
                    resized_wh=list(asset.size), gt_boxes=[box], width=w, height=h)
    (out/'record.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(out.resolve())


if __name__ == '__main__':
    main()
