import argparse
import json
import os
from PIL import Image, ImageDraw
from tqdm import tqdm

def load_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Most common: list of {"name": "...jpg", "labels": [...]}
    if isinstance(data, list):
        m = {}
        for item in data:
            name = item.get("name")
            if name is None:
                continue
            m[name] = item.get("labels", [])
        return m

    # If dict, try common keys
    if isinstance(data, dict):
        for key in ("frames", "images", "data"):
            if key in data and isinstance(data[key], list):
                m = {}
                for item in data[key]:
                    name = item.get("name")
                    if name is None:
                        continue
                    m[name] = item.get("labels", [])
                return m

    raise RuntimeError(f"Unknown label JSON format in {json_path}")

def draw_drivable_mask(size, labels):
    # 0 bg, 1 direct, 2 alternative
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for lab in labels:
        cat = (lab.get("category") or "").lower()
        poly2d = lab.get("poly2d")
        if not poly2d:
            continue

        if cat == "direct":
            val = 1
        elif cat == "alternative":
            val = 2
        else:
            continue

        for p in poly2d:
            verts = p.get("vertices", [])
            if len(verts) >= 3:
                draw.polygon([tuple(v) for v in verts], fill=val)
    return mask

def draw_lane_mask(size, labels, width=6):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for lab in labels:
        poly2d = lab.get("poly2d")
        if not poly2d:
            continue
        for p in poly2d:
            verts = p.get("vertices", [])
            if len(verts) >= 2:
                draw.line([tuple(v) for v in verts], fill=1, width=width)
    return mask

def prepare_split(split, img_dir, drv_labels, lane_labels, out_root, out_hw, lane_width, skip_existing):
    H, W = out_hw
    out_img_dir = os.path.join(out_root, "images", split)
    out_drv_dir = os.path.join(out_root, "masks_drivable", split)
    out_lane_dir = os.path.join(out_root, "masks_lane", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_drv_dir, exist_ok=True)
    os.makedirs(out_lane_dir, exist_ok=True)

    names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])

    for name in tqdm(names, desc=f"prepare {split}"):
        out_img = os.path.join(out_img_dir, name)
        out_drv = os.path.join(out_drv_dir, name.replace(".jpg", ".png"))
        out_lane = os.path.join(out_lane_dir, name.replace(".jpg", ".png"))

        if skip_existing and os.path.exists(out_img) and os.path.exists(out_drv) and os.path.exists(out_lane):
            continue

        img = Image.open(os.path.join(img_dir, name)).convert("RGB")
        w0, h0 = img.size

        d_labs = drv_labels.get(name, [])
        l_labs = lane_labels.get(name, [])

        drv = draw_drivable_mask((w0, h0), d_labs)
        lane = draw_lane_mask((w0, h0), l_labs, width=lane_width)

        img_r = img.resize((W, H), Image.BILINEAR)
        drv_r = drv.resize((W, H), Image.NEAREST)
        lane_r = lane.resize((W, H), Image.NEAREST)

        img_r.save(out_img, quality=90)
        drv_r.save(out_drv)
        lane_r.save(out_lane)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--img_h", type=int, default=360)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--lane_width", type=int, default=6)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--drivable_train_json", required=True)
    ap.add_argument("--drivable_val_json", required=True)
    ap.add_argument("--lane_train_json", required=True)
    ap.add_argument("--lane_val_json", required=True)
    args = ap.parse_args()

    img_train = os.path.join(args.raw_root, "images", "100k", "train")
    img_val = os.path.join(args.raw_root, "images", "100k", "val")
    if not os.path.isdir(img_train) or not os.path.isdir(img_val):
        raise RuntimeError("Expected images at raw_root/images/100k/{train,val}")

    drv_train = load_labels(args.drivable_train_json)
    drv_val = load_labels(args.drivable_val_json)
    lane_train = load_labels(args.lane_train_json)
    lane_val = load_labels(args.lane_val_json)

    prepare_split("train", img_train, drv_train, lane_train, args.out_root,
                  (args.img_h, args.img_w), args.lane_width, args.skip_existing)
    prepare_split("val", img_val, drv_val, lane_val, args.out_root,
                  (args.img_h, args.img_w), args.lane_width, args.skip_existing)

    os.makedirs(os.path.join(args.out_root, "meta"), exist_ok=True)
    with open(os.path.join(args.out_root, "meta", "version.txt"), "w") as f:
        f.write(f"prepared_{args.img_h}x{args.img_w}\n")

if __name__ == "__main__":
    main()