import os
import random
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_DIR = "data/processed/train/patches/images"
MASK_DIR = "data/processed/train/patches/masks"

random.seed(42)

files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png")
]

selected = random.sample(files, 6)

fig, axes = plt.subplots(6, 2, figsize=(8, 20))

for i, filename in enumerate(selected):

    image = Image.open(
        os.path.join(IMAGE_DIR, filename)
    )

    mask = Image.open(
        os.path.join(MASK_DIR, filename)
    )

    axes[i, 0].imshow(image)
    axes[i, 0].set_title("H&E Patch")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/figures/sample_patches.png",
    dpi=150,
    bbox_inches="tight"
)
