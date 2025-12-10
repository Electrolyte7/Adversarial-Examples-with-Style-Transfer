import torch
import torch.nn as nn

def gram_matrix(feat: torch.Tensor):
    # feat: [B, C, H, W]
    B, C, H, W = feat.shape
    F = feat.view(B, C, H * W)
    G = torch.bmm(F, F.transpose(1, 2)) / (C * H * W)
    return G

class StyleContentAdvLoss(nn.Module):
    def __init__(self, style_layers, content_layers, layer_weights=None):
        super().__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.layer_weights = layer_weights or {}
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def style_loss(self, adv_feats, style_targets):
        loss = 0.0
        for l in self.style_layers:
            w = self.layer_weights.get(l, 1.0)
            G_adv = gram_matrix(adv_feats[l])
            loss += w * self.mse(G_adv, style_targets[l])
        return loss

    def content_loss(self, adv_feats, content_targets):
        loss = 0.0
        for l in self.content_layers:
            w = self.layer_weights.get(l, 1.0)
            loss += w * self.mse(adv_feats[l], content_targets[l])
        return loss

    def adv_loss(self, logits, target_idx):
        # 定向对抗：让logits预测为目标类
        B = logits.size(0)
        target = torch.full((B,), target_idx, device=logits.device, dtype=torch.long)
        return self.ce(logits, target)
