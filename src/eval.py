import torch
from src.metrics import confusion_matrix, miou_from_cm

@torch.no_grad()
def evaluate(model, loader, device, num_sem, num_drv, ignore_index=255):
    model.eval()
    cm_sem = torch.zeros((num_sem, num_sem), dtype=torch.int64, device=device)
    cm_drv = torch.zeros((num_drv, num_drv), dtype=torch.int64, device=device)

    for imgs, sem_gt, drv_gt, _ in loader:
        imgs = imgs.to(device)
        sem_gt = sem_gt.to(device)
        drv_gt = drv_gt.to(device)

        out = model(imgs)
        sem_pred = out["semantic"].argmax(1)
        drv_pred = out["drivable"].argmax(1)

        for i in range(imgs.size(0)):
            cm_sem += confusion_matrix(sem_pred[i], sem_gt[i], num_sem, ignore_index)
            cm_drv += confusion_matrix(drv_pred[i], drv_gt[i], num_drv, ignore_index)

    miou_sem, _ = miou_from_cm(cm_sem)

    # drivable: focus on class 1 IoU (drivable pixels)
    inter = torch.diag(cm_drv).float()
    union = cm_drv.sum(0).float() + cm_drv.sum(1).float() - inter
    iou_drv1 = (inter[1] / union[1]).item() if union[1] > 0 else 0.0

    val_score = 0.5 * (miou_sem + iou_drv1)
    return {"val_miou_sem": miou_sem, "val_iou_drv1": iou_drv1, "val_score": val_score}