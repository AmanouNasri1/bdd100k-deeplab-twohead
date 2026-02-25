import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabV3TwoHead(nn.Module):
    def __init__(self, pretrained: bool, num_semantic: int, num_drivable: int):
        super().__init__()
        base = deeplabv3_mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        self.backbone = base.backbone
        in_ch = 960
        self.head_sem = DeepLabHead(in_ch, num_semantic)
        self.head_drv = DeepLabHead(in_ch, num_drivable)

    def forward(self, x):
        input_shape = x.shape[-2:]
        feats = self.backbone(x)["out"]
        sem = self.head_sem(feats)
        drv = self.head_drv(feats)
        sem = F.interpolate(sem, size=input_shape, mode="bilinear", align_corners=False)
        drv = F.interpolate(drv, size=input_shape, mode="bilinear", align_corners=False)
        return {"semantic": sem, "drivable": drv}