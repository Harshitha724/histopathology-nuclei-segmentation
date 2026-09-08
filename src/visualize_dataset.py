import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = (
    "data/raw/MoNuSeg 2018 Training Data/"
    "Tissue Images/TCGA-18-5592-01Z-00-DX1.tif"
)

MASK_PATH = (
    "data/processed/masks/"
    "TCGA-18-5592-01Z-00-DX1.png"
)


image = Image.open(IMAGE_PATH)
mask = Image.open(MASK_PATH)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("H&E Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Nuclei Mask")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/figures/image_and_mask.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
