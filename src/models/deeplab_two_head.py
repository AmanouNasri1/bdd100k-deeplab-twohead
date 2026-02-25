import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabV3TwoHead(nn.Module):
    def __init__(self, pretrained: bool, num_drivable: int, num_lane: int):
        super().__init__()
        base = deeplabv3_mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        self.backbone = base.backbone  # returns dict {"out": tensor}

        # MobileNetV3 DeepLab uses 960 channels at out
        in_ch = 960
        self.head_drivable = DeepLabHead(in_ch, num_drivable)
        self.head_lane = DeepLabHead(in_ch, num_lane)

    def forward(self, x):
        input_shape = x.shape[-2:]
        feats = self.backbone(x)["out"]

        out_d = self.head_drivable(feats)
        out_l = self.head_lane(feats)

        out_d = F.interpolate(out_d, size=input_shape, mode="bilinear", align_corners=False)
        out_l = F.interpolate(out_l, size=input_shape, mode="bilinear", align_corners=False)

        return {"drivable": out_d, "lane": out_l}