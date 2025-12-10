import argparse, os, glob, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import vgg19, VGG19_Weights

from src.losses import StyleContentAdvLoss, gram_matrix
from src.vgg_features import VGGFeatureExtractor, VGG_LAYERS_MAP
from src.utils import set_seed, load_image, save_image, save_pred_fig, find_target_index

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", type=str, default="data/content")
    ap.add_argument("--style_path", type=str, default="data/style/style.jpg")
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lambda_adv", type=float, default=50.0)    # 目标强度
    ap.add_argument("--alpha", type=float, default=1e4)          # 风格权重
    ap.add_argument("--beta", type=float, default=1.0)           # 内容权重
    ap.add_argument("--target_idx", type=int, default=None)      # 可手动指定
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.join(args.output_dir, "adv_images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "preds"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu")

    # 1) 分类器（用于对抗损失与最终预测）
    weights = VGG19_Weights.IMAGENET1K_V1
    classifier = vgg19(weights=weights).eval().to(device)
    for p in classifier.parameters():
        p.requires_grad_(False)

    # 有些版本的 weights.meta 没有 mean/std/categories，这里统一兜底
    meta = getattr(weights, "meta", {}) or {}
    categories = meta.get("categories", [f"class_{i}" for i in range(1000)])
    img_mean = torch.tensor(meta.get("mean", [0.485, 0.456, 0.406]),
                            device=device).view(1, 3, 1, 1)
    img_std = torch.tensor(meta.get("std", [0.229, 0.224, 0.225]),
                           device=device).view(1, 3, 1, 1)


    # 自动寻找“movie theater / cinema”索引；找不到再用 --target_idx 指定
    target_idx = find_target_index(categories)
    if target_idx is None:
        if args.target_idx is None:
            raise ValueError("未在类别列表中找到 'movie theater/cinema'，请通过 --target_idx 手动指定。")
        target_idx = args.target_idx

    # 2) 特征提取器（用于风格/内容损失）
    style_layers_idx = {"0":"conv1_1","5":"conv2_1","10":"conv3_1","19":"conv4_1","28":"conv5_1"}
    content_layers_idx = {"21":"conv4_2"}
    feat_model = VGGFeatureExtractor({**style_layers_idx, **content_layers_idx}).to(device).eval()

    # 3) 加载风格图并预计算风格目标（Gram）
    style = load_image(args.style_path).unsqueeze(0).to(device)
    with torch.no_grad():
        style_feats = feat_model(style)
        style_targets = {l: gram_matrix(style_feats[l]) for l in style_layers_idx.values()}

    # 4) 构造损失
    # 可对各层设权重（默认1.0）；如需强调浅层纹理，可加大conv1_1/2_1权重
    layer_weights = {"conv1_1":1.0,"conv2_1":1.0,"conv3_1":1.0,"conv4_1":1.0,"conv5_1":1.0,"conv4_2":1.0}
    crit = StyleContentAdvLoss(
        style_layers=list(style_layers_idx.values()),
        content_layers=list(content_layers_idx.values()),
        layer_weights=layer_weights
    ).to(device)

    # 5) 读取10张内容图
    content_paths = sorted(glob.glob(os.path.join(args.content_dir, "*")))
    if len(content_paths) < 10:
        print(f"[警告] 仅发现 {len(content_paths)} 张图，将对其全部攻击。建议放置≥10张。")

    for img_id, cpath in enumerate(content_paths[:10], 1):
        content = load_image(cpath).unsqueeze(0).to(device)
        with torch.no_grad():
            content_targets = feat_model(content)

        # 以内容图为初值优化（也可用内容+微噪声）
        x_adv = content.clone().requires_grad_(True)
        opt = torch.optim.Adam([x_adv], lr=args.lr)

        pbar = tqdm(range(args.iters), desc=f"[{img_id:02d}] {os.path.basename(cpath)}")
        best_pred, best_prob, best_img = None, -1.0, None

        for it in pbar:
            opt.zero_grad()

            # 5.1 风格/内容特征
            adv_feats = feat_model(x_adv)

            # 5.2 分类器前向（用于对抗损失）
            logits = classifier((x_adv - img_mean) / img_std)

            # 5.3 计算三类损失
            s_loss = crit.style_loss(adv_feats, style_targets)
            c_loss = crit.content_loss(adv_feats, content_targets)
            a_loss = crit.adv_loss(logits, target_idx)

            total = args.lambda_adv * a_loss + args.alpha * s_loss + args.beta * c_loss
            total.backward()
            opt.step()

            # 像素域裁剪保持[0,1]
            with torch.no_grad():
                x_adv.clamp_(0.0, 1.0)

            # 记录当前预测与概率
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                top_prob, top_idx = probs.max(dim=1)
                pred_name = categories[top_idx.item()]
                pbar.set_postfix({
                    "L": f"{total.item():.2f}",
                    "adv": f"{a_loss.item():.2f}",
                    "sty": f"{s_loss.item():.2f}",
                    "con": f"{c_loss.item():.2f}",
                    "pred": f"{pred_name[:14]}",
                    "p": f"{top_prob.item():.2f}"
                })
                if top_prob.item() > best_prob:
                    best_prob, best_pred = top_prob.item(), pred_name
                    best_img = x_adv.detach().clone()

        # 保存最终结果（用最后一步；也可换best_img）
        adv_img = x_adv.detach().squeeze(0).cpu()
        base = os.path.splitext(os.path.basename(cpath))[0]
        out_img = os.path.join(args.output_dir, "adv_images", f"{img_id:02d}_{base}_adv.png")
        save_image(adv_img, out_img)

        # 保存“截图”图（图片+标题显示预测与概率）
        # 以最终一步预测来生成
        with torch.no_grad():
            logits = classifier((x_adv - img_mean) / img_std)
            probs = F.softmax(logits, dim=1)
            
            top_prob, top_idx = probs.max(dim=1)
            label_str = categories[top_idx.item()]
            out_fig = os.path.join(args.output_dir, "preds", f"{img_id:02d}_{base}_pred.png")
            save_pred_fig(adv_img, label_str, top_prob.item(), out_fig)

        # 简要日志
        print(f"[DONE] {base}: top-1 = {label_str} ({top_prob.item():.2%}); saved to {out_img}")

if __name__ == "__main__":
    main()
