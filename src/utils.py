import os, random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

IM_SIZE = 224

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_image(path):
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize(IM_SIZE),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),  # [0,1]
    ])
    return tfm(img)

def save_image(tensor, path):
    # tensor: [3,H,W] in [0,1]
    arr = (tensor.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(path)

def find_target_index(categories, keywords=("movie theater", "cinema")):
    # 在weights.meta["categories"]中模糊搜索
    cand = []
    for i, name in enumerate(categories):
        low = name.lower()
        if any(k in low for k in [k.lower() for k in keywords]):
            cand.append((i, name))
    # 以首个为默认；若没搜到，让用户在命令行传入 --target_idx 手工指定
    return cand[0][0] if cand else None

def save_pred_fig(img_tensor, label_str, prob, out_path):
    # 生成“带文字”的截图
    plt.figure(figsize=(3.6,3.6), dpi=100)
    plt.axis('off')
    arr = img_tensor.clamp(0,1).permute(1,2,0).cpu().numpy()
    plt.imshow(arr)
    plt.title(f"{label_str} ({prob:.2%})")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
