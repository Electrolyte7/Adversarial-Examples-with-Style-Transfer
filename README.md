# Adversarial Examples with Style Transfer

This project implements an adversarial attack combined with neural style transfer to generate natural-looking adversarial images that fool a pre-trained VGG19 classifier into misclassifying them as "movie theater, cinema". The approach is inspired by the paper ["Adversarial Camouflage: Hiding Physical-World Attacks with Natural Styles"](https://arxiv.org/abs/2003.08757) (Duan et al., CVPR 2020).

The adversarial examples are created by optimizing a combined loss function that includes:
- **Adversarial loss**: Cross-entropy to target the desired class.
- **Style loss**: Gram matrix matching to apply the style from a reference image.
- **Content loss**: Feature map preservation to keep the original content.

The optimization is performed using Adam, with hyperparameters controlling the balance between these losses.

## Project Structure

- `data/`
  - `content/`: Directory for input content images (e.g., 10 ImageNet images).
  - `style/`: Directory for the style reference image (e.g., `style.jpg`).
- `outputs_style/`: Output directory (configurable).
  - `adv_images/`: Generated adversarial images.
  - `logs/`: (Currently unused, but reserved for logs).
  - `preds/`: Screenshots of adversarial images with predicted labels and probabilities.
- `src/`: Source code.
  - `attack.py`: Main script for running the attack.
  - `losses.py`: Defines style, content, and adversarial loss functions.
  - `utils.py`: Utility functions for loading/saving images, setting seeds, etc.
  - `vgg_features.py`: VGG feature extractor for style and content losses.
- `requirements.txt`: Python dependencies.
- `run_attack.sh`: Bash script to run the attack with configurable parameters.
- `README.md`: This file.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended for GPU acceleration)
- Other dependencies listed in `requirements.txt`:
  ```
  torch
  torchvision
  tqdm
  matplotlib
  pillow
  numpy
  ```

Install dependencies with:
```
pip install -r requirements.txt
```

## Setup

1. Prepare input images:
   - Place at least 10 content images (e.g., from ImageNet) in `data/content/`. You can use examples from [this GitHub repo](https://github.com/RjDuan/AdvCam-Hide-Adv-with-Natural-Styles/).
   - Place a style reference image (e.g., an artistic image for natural perturbations) in `data/style/style.jpg`.

2. Ensure a GPU is available for faster computation (falls back to CPU if not).

## Usage

### Running via Bash Script (Recommended)

The `run_attack.sh` script provides an easy way to configure and run the attack. Edit the script or pass arguments as needed.

Example:
```
bash run_attack.sh --content_dir data/content --style_path data/style/style.jpg --output_dir outputs_style --iters 1000 --lr 0.05 --lambda_adv 25 --alpha 30000 --beta 0.2 --device cuda
```

Key parameters:
- `--content_dir`: Path to content images directory.
- `--style_path`: Path to style image.
- `--output_dir`: Output directory.
- `--iters`: Number of optimization iterations (default: 1000).
- `--lr`: Learning rate for Adam optimizer (default: 0.05).
- `--lambda_adv`: Weight for adversarial loss (default: 25).
- `--alpha`: Weight for style loss (default: 30000).
- `--beta`: Weight for content loss (default: 0.2).
- `--device`: Device to use (`cuda` or `cpu`).

### Running via Python

Run `attack.py` directly:
```
python -m src.attack --content_dir data/content --style_path data/style/style.jpg --output_dir outputs_style --iters 1000 --lr 0.05 --lambda_adv 25 --alpha 30000 --beta 0.2 --device cuda
```

The script processes up to 10 images from the content directory, generates adversarial versions, saves them in `adv_images/`, and creates prediction screenshots in `preds/`.

## Method Overview

1. **Load Models**:
   - Pre-trained VGG19 classifier from `torchvision` for adversarial loss and predictions.
   - VGG feature extractor for style (layers: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1) and content (layer: conv4_2) losses.

2. **Precompute Targets**:
   - Gram matrices from style image for style loss.
   - Feature maps from content image for content loss.

3. **Optimization**:
   - Initialize adversarial image as the content image.
   - Optimize using Adam to minimize the total loss: \(\lambda_{adv} \cdot \mathcal{L}_{adv} + \alpha \cdot \mathcal{L}_{style} + \beta \cdot \mathcal{L}_{content}\).
   - Clip pixel values to [0, 1] after each step.

4. **Output**:
   - Adversarial images and prediction figures (image + title with predicted label and probability).

Tune hyperparameters (`lambda_adv`, `alpha`, `beta`) to balance attack success, style application, and content preservation. The target class is automatically detected as "movie theater" or similar from ImageNet categories (index can be manually specified via `--target_idx` if needed).

## Example Results

After running, check `outputs_style/adv_images/` for adversarial images and `outputs_style/preds/` for visualizations. Successful attacks should show the VGG19 predicting "movie theater, cinema, cineplex" with high probability while the image visually resembles the style-applied content.

## Notes

- This implementation focuses on digital-domain attacks; physical-world robustness (e.g., EOT) is not included.
- For reference, see [this Zhihu article](https://zhuanlan.zhihu.com/p/409662511) for similar style transfer implementations.
- If issues arise (e.g., target class not found), specify `--target_idx` manually (ImageNet class index for "movie theater" is typically 652).

Good luck with your experiments! If you encounter issues, feel free to open an issue on the GitHub repo.