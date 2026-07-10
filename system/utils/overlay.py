import numpy as np
from PIL import Image

COLORS = np.array([
    [0,0,0],
    [0,255,0],
    [255,255,0],
    [255,165,0],
    [255,0,0],
], dtype=np.uint8)


def overlay_mask(image, mask, alpha=0.4):
    """
    image : torch tensor (3,H,W)
    mask  : torch tensor (H,W)
    """

    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1,2,0))
    image = (image*255).clip(0,255).astype(np.uint8)

    mask = mask.detach().cpu().numpy().astype(np.uint8)
    color = COLORS[mask]

    overlay = (
        image*(1-alpha)
        + color*alpha
    ).astype(np.uint8)

    return Image.fromarray(overlay)