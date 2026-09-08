import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import find_contours


# =========================
# Paths
# =========================
IMAGE_DIR = "data/processed/val/normalized/images"
MASK_DIR = "data/processed/val/normalized/masks"
OUTPUT_DIR = "outputs/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Parameters
# =========================
MIN_DISTANCE = 8


# =========================
# Select one validation patch
# =========================
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png")
])

# Pick a representative patch
filename = image_files[100]

image_path = os.path.join(IMAGE_DIR, filename)
mask_path = os.path.join(MASK_DIR, filename)


# =========================
# Load image and mask
# =========================
image = np.array(
    Image.open(image_path).convert("RGB")
)

mask = np.array(
    Image.open(mask_path).convert("L")
)

binary = mask > 127


# =========================
# Distance transform
# =========================
distance = ndimage.distance_transform_edt(binary)


# =========================
# Find nuclei centers
# =========================
coordinates = peak_local_max(
    distance,
    min_distance=MIN_DISTANCE,
    labels=binary
)


# =========================
# Create markers
# =========================
markers = np.zeros_like(binary, dtype=np.int32)

for i, (y, x) in enumerate(coordinates, start=1):
    markers[y, x] = i


# =========================
# Watershed
# =========================
labels = watershed(
    -distance,
    markers,
    mask=binary
)


# =========================
# Create visualization
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original H&E
axes[0].imshow(image)
axes[0].set_title("Normalized H&E")
axes[0].axis("off")


# Binary mask
axes[1].imshow(binary, cmap="gray")
axes[1].set_title("U-Net Mask")
axes[1].axis("off")


# Watershed boundaries
axes[2].imshow(image)

contours = find_contours(labels, 0.5)

for contour in contours:
    axes[2].plot(
        contour[:, 1],
        contour[:, 0],
        linewidth=0.8
    )

axes[2].set_title("Watershed-separated Nuclei")
axes[2].axis("off")


plt.tight_layout()


# =========================
# Save
# =========================
output_path = os.path.join(
    OUTPUT_DIR,
    "watershed_visualization.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("=" * 50)
print("WATERSHED VISUALIZATION")
print("=" * 50)
print(f"Patch: {filename}")
print(f"Watershed regions: {len(np.unique(labels)) - 1}")
print(f"Saved: {output_path}")
print("=" * 50)
