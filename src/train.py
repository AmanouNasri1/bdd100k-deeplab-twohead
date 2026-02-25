import argparse
import os
import csv
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.data.prepared_bdd100k import PreparedBDD100K
from src.models.deeplab_two_head import DeepLabV3TwoHead
from src.eval import evaluate

def save_ckpt(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)

def load_ckpt(path, device):
    return torch.load(path, map_location=device)

def main(cfg, run_dir, resume: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(cfg["seed"])

    H, W = cfg["data"]["img_size"]
    prep = cfg["data"]["prepared_root"]

    ds_train = PreparedBDD100K(prep, "train", (H, W))
    ds_val   = PreparedBDD100K(prep, "val", (H, W))

    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = DeepLabV3TwoHead(
        pretrained=cfg["model"]["pretrained"],
        num_drivable=cfg["model"]["num_classes_drivable"],
        num_lane=cfg["model"]["num_classes_lane"],
    ).to(device)

    drv_w = torch.tensor(cfg["loss"]["drivable_ce_weight"], dtype=torch.float32, device=device)
    loss_drv = nn.CrossEntropyLoss(weight=drv_w)

    lane_w = torch.tensor([1.0, float(cfg["loss"]["lane_pos_weight"])], dtype=torch.float32, device=device)
    loss_lane = nn.CrossEntropyLoss(weight=lane_w)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    start_epoch = 0
    global_step = 0
    best_metric = -1.0

    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")
    metrics_csv = os.path.join(run_dir, "metrics.csv")

    if resume and os.path.exists(last_path):
        ckpt = load_ckpt(last_path, device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_metric = ckpt.get("best_metric", best_metric)
        print(f"[resume] epoch={start_epoch} global_step={global_step} best={best_metric:.4f}")

    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_iou_drivable", "val_iou_lane", "val_mean_iou", "best_so_far"])

    save_every = int(cfg["checkpoint"]["save_every_steps"])
    keep_epoch_ckpts = bool(cfg["checkpoint"]["keep_epoch_ckpts"])

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        n = 0

        optim.zero_grad(set_to_none=True)
        pbar = tqdm(dl_train, desc=f"epoch {epoch}", leave=False)

        for imgs, drv_gt, lane_gt, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            drv_gt = drv_gt.to(device, non_blocking=True)
            lane_gt = lane_gt.to(device, non_blocking=True)

            with autocast(enabled=cfg["train"]["amp"]):
                out = model(imgs)
                l1 = loss_drv(out["drivable"], drv_gt)
                l2 = loss_lane(out["lane"], lane_gt)
                loss = l1 + l2

            scaler.scale(loss).backward()

            if (global_step + 1) % cfg["train"]["grad_accum_steps"] == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
            global_step += 1
            pbar.set_postfix(loss=running / max(n, 1))

            if global_step % save_every == 0:
                save_ckpt(last_path, {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metric": best_metric,
                    "cfg": cfg,
                })

        train_loss = running / max(n, 1)

        val = evaluate(model, dl_val, device,
                       cfg["model"]["num_classes_drivable"],
                       cfg["model"]["num_classes_lane"])

        if val["val_mean_iou"] > best_metric:
            best_metric = val["val_mean_iou"]
            save_ckpt(best_path, {
                "model": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_metric,
                "cfg": cfg,
            })

        save_ckpt(last_path, {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "cfg": cfg,
        })

        if keep_epoch_ckpts:
            save_ckpt(os.path.join(run_dir, f"epoch_{epoch:03d}.pt"), {
                "model": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_metric,
                "cfg": cfg,
            })

        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, val["val_iou_drivable"], val["val_iou_lane"], val["val_mean_iou"], best_metric])

        print(f"[epoch {epoch}] loss={train_loss:.4f} val_mean_iou={val['val_mean_iou']:.4f} best={best_metric:.4f}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.run_dir, args.resume)