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
PREDICTION_DIR = "outputs/predictions"
OUTPUT_DIR = "outputs/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Parameters
# =========================
MIN_DISTANCE = 8

# Pick one representative validation patch
FILENAME = sorted([
    f for f in os.listdir(PREDICTION_DIR)
    if f.endswith(".png")
])[100]


# =========================
# Load image
# =========================
image_path = os.path.join(
    IMAGE_DIR,
    FILENAME
)

prediction_path = os.path.join(
    PREDICTION_DIR,
    FILENAME
)

image = np.array(
    Image.open(image_path).convert("RGB")
)

prediction = np.array(
    Image.open(prediction_path).convert("L")
)

binary = prediction > 127


# =========================
# Clean prediction
# =========================
binary = ndimage.binary_opening(binary)


# =========================
# Distance transform
# =========================
distance = ndimage.distance_transform_edt(
    binary
)


# =========================
# Find nucleus centers
# =========================
coordinates = peak_local_max(
    distance,
    min_distance=MIN_DISTANCE,
    labels=binary
)


# =========================
# Create markers
# =========================
markers = np.zeros_like(
    binary,
    dtype=np.int32
)

for i, (y, x) in enumerate(
    coordinates,
    start=1
):
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
# Visualization
# =========================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# H&E image
axes[0].imshow(image)
axes[0].set_title(
    "Normalized H&E"
)
axes[0].axis("off")


# U-Net prediction
axes[1].imshow(
    binary,
    cmap="gray"
)
axes[1].set_title(
    "U-Net Prediction"
)
axes[1].axis("off")


# Watershed boundaries
axes[2].imshow(image)

contours = find_contours(
    labels,
    0.5
)

for contour in contours:

    axes[2].plot(
        contour[:, 1],
        contour[:, 0],
        linewidth=0.8
    )

axes[2].set_title(
    "U-Net + Watershed"
)
axes[2].axis("off")


plt.tight_layout()


# =========================
# Save
# =========================
output_path = os.path.join(
    OUTPUT_DIR,
    "final_prediction_watershed.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================
# Print information
# =========================
print("=" * 50)
print("FINAL U-NET + WATERSHED VISUALIZATION")
print("=" * 50)

print("Patch:", FILENAME)
print(
    "Watershed nuclei:",
    len(np.unique(labels)) - 1
)

print("Saved:", output_path)

print("=" * 50)
