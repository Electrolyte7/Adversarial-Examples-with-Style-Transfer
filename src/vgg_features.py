import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

VGG_LAYERS_MAP = {
    "0": "conv1_1", "2": "conv1_2",
    "5": "conv2_1", "7": "conv2_2",
    "10": "conv3_1","12":"conv3_2","14":"conv3_3","16":"conv3_4",
    "19":"conv4_1","21":"conv4_2","23":"conv4_3","25":"conv4_4",
    "28":"conv5_1","30":"conv5_2","32":"conv5_3","34":"conv5_4",
}

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers_to_return):
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        # 有些版本的 weights.meta 里没有 mean/std，就用常见的 ImageNet 预处理数值兜底
        meta = getattr(weights, "meta", {}) or {}
        self.mean = meta.get("mean", [0.485, 0.456, 0.406])
        self.std = meta.get("std", [0.229, 0.224, 0.225])
        self.categories = meta.get("categories", None)

        cnn = vgg19(weights=weights)
        self.features = cnn.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.layers_to_return = layers_to_return

    def forward(self, x):
        # 期望 x ∈ [0,1]
        x = (x - x.new_tensor(self.mean)[None, :, None, None]) / \
            x.new_tensor(self.std)[None, :, None, None]
        feats = {}
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            key = str(i)
            if key in self.layers_to_return:
                feats[self.layers_to_return[key]] = h
        return feats
