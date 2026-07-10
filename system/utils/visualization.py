import os
import numpy as np
from PIL import Image

COLORS = np.array([
    [0, 0, 0],          # 0 Background
    [0, 255, 0],        # 1 No Damage
    [255, 255, 0],      # 2 Minor
    [255, 165, 0],      # 3 Major
    [255, 0, 0],        # 4 Destroyed
], dtype=np.uint8)


def tensor_to_rgb(img):
    """
    img : (3,H,W) torch tensor in [0,1]
    """

    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(img)


def mask_to_rgb(mask):
    """
    mask : (H,W)
    """

    mask = mask.detach().cpu().numpy().astype(np.uint8)

    rgb = COLORS[mask]

    return Image.fromarray(rgb)


def save_prediction_visualization(
    pre,
    post,
    gt,
    pred,
    save_dir,
    sample_idx,
):
    """
    Saves:

    pre.png
    post.png
    gt.png
    pred.png

    as ONE comparison image.
    """

    os.makedirs(save_dir, exist_ok=True)

    pre = tensor_to_rgb(pre)
    post = tensor_to_rgb(post)

    gt = mask_to_rgb(gt)
    pred = mask_to_rgb(pred)

    w, h = pre.size

    canvas = Image.new("RGB", (2 * w, 2 * h))

    canvas.paste(pre, (0, 0))
    canvas.paste(post, (w, 0))
    canvas.paste(gt, (0, h))
    canvas.paste(pred, (w, h))

    canvas.save(
        os.path.join(
            save_dir,
            f"sample_{sample_idx:02d}.png"
        )
    )