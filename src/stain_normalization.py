import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchstain


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PATCH_DIR = "data/processed/train/patches/images"
OUTPUT_DIR = "outputs/figures"

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Select reference and source patches
# --------------------------------------------------

random.seed(SEED)

patch_files = sorted(
    f for f in os.listdir(PATCH_DIR)
    if f.endswith(".png")
)

if len(patch_files) < 2:
    raise RuntimeError("Not enough patches found.")


reference_file = patch_files[0]
source_file = patch_files[1]

reference_path = os.path.join(
    PATCH_DIR,
    reference_file
)

source_path = os.path.join(
    PATCH_DIR,
    source_file
)


# --------------------------------------------------
# Load images
# --------------------------------------------------

reference = np.array(
    Image.open(reference_path).convert("RGB")
)

source = np.array(
    Image.open(source_path).convert("RGB")
)


# --------------------------------------------------
# Create Macenko normalizer
# --------------------------------------------------

normalizer = torchstain.normalizers.MacenkoNormalizer(
    backend="numpy"
)


# Fit the normalizer using the reference image
normalizer.fit(reference)


# --------------------------------------------------
# Normalize source image
# --------------------------------------------------

normalized, _, _ = normalizer.normalize(
    I=source
)


# Make sure the result is uint8
normalized = np.asarray(normalized).astype(np.uint8)


# --------------------------------------------------
# Save normalized patch
# --------------------------------------------------

normalized_path = os.path.join(
    OUTPUT_DIR,
    "macenko_normalized_patch.png"
)

Image.fromarray(normalized).save(
    normalized_path
)


# --------------------------------------------------
# Visual comparison
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(source)
plt.title("Original H&E")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalized)
plt.title("Macenko Normalized")
plt.axis("off")

plt.tight_layout()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "stain_normalization.png"
)

plt.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("Macenko normalization complete.")
print(f"Reference : {reference_file}")
print(f"Source    : {source_file}")
print(f"Saved     : {comparison_path}")
