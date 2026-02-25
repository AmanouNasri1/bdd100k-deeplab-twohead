import torch

@torch.no_grad()
def compute_iou(pred, target, num_classes: int):
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0

@torch.no_grad()
def evaluate(model, loader, device, num_drivable, num_lane):
    model.eval()
    drv_ious = []
    lane_ious = []

    for imgs, drv_gt, lane_gt, _ in loader:
        imgs = imgs.to(device)
        drv_gt = drv_gt.to(device)
        lane_gt = lane_gt.to(device)

        out = model(imgs)
        drv_pred = out["drivable"].argmax(dim=1)
        lane_pred = out["lane"].argmax(dim=1)

        for i in range(imgs.size(0)):
            drv_ious.append(compute_iou(drv_pred[i], drv_gt[i], num_drivable))
            lane_ious.append(compute_iou(lane_pred[i], lane_gt[i], num_lane))

    iou_drv = sum(drv_ious) / len(drv_ious) if drv_ious else 0.0
    iou_lane = sum(lane_ious) / len(lane_ious) if lane_ious else 0.0
    mean_iou = 0.5 * (iou_drv + iou_lane)

    return {"val_iou_drivable": iou_drv, "val_iou_lane": iou_lane, "val_mean_iou": mean_iou}