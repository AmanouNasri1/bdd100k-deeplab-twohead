import torch


@torch.no_grad()
def confusion_matrix(pred, target, num_classes, ignore_index=255):
    mask = target != ignore_index
    pred = pred[mask].view(-1)
    target = target[mask].view(-1)
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)

    k = (target * num_classes + pred).to(torch.int64)
    cm = torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def miou_from_cm(cm):
    inter = torch.diag(cm).float()
    union = cm.sum(0).float() + cm.sum(1).float() - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou.mean().item(), iou