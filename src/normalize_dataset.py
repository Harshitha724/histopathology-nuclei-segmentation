from pathlib import Path
import numpy as np
from PIL import Image
import torchstain
from tqdm import tqdm


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path("data/processed")

TRAIN_IMAGES = BASE_DIR / "train/patches/images"
TRAIN_MASKS = BASE_DIR / "train/patches/masks"

VAL_IMAGES = BASE_DIR / "val/patches/images"
VAL_MASKS = BASE_DIR / "val/patches/masks"

TRAIN_NORM_IMAGES = BASE_DIR / "train/normalized/images"
TRAIN_NORM_MASKS = BASE_DIR / "train/normalized/masks"

VAL_NORM_IMAGES = BASE_DIR / "val/normalized/images"
VAL_NORM_MASKS = BASE_DIR / "val/normalized/masks"


# Create output directories
for directory in [
    TRAIN_NORM_IMAGES,
    TRAIN_NORM_MASKS,
    VAL_NORM_IMAGES,
    VAL_NORM_MASKS,
]:
    directory.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Select reference image
# -----------------------------
reference_path = sorted(TRAIN_IMAGES.glob("*.png"))[0]

reference = np.array(
    Image.open(reference_path).convert("RGB")
)

print("Using reference:")
print(reference_path.name)


# -----------------------------
# Fit Macenko normalizer
# -----------------------------
normalizer = torchstain.normalizers.MacenkoNormalizer(
    backend="numpy"
)

normalizer.fit(reference)


# -----------------------------
# Normalize images
# -----------------------------
def normalize_images(input_dir, output_dir):
    image_paths = sorted(input_dir.glob("*.png"))

    for image_path in tqdm(image_paths, desc=f"Normalizing {input_dir}"):
        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        normalized, _, _ = normalizer.normalize(I=image)

        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        output_path = output_dir / image_path.name

        Image.fromarray(normalized).save(output_path)


# -----------------------------
# Copy masks unchanged
# -----------------------------
def copy_masks(input_dir, output_dir):
    mask_paths = sorted(input_dir.glob("*.png"))

    for mask_path in tqdm(mask_paths, desc=f"Copying masks {input_dir}"):
        output_path = output_dir / mask_path.name

        Image.open(mask_path).save(output_path)


# -----------------------------
# Run
# -----------------------------
print("\nNormalizing training images...")
normalize_images(TRAIN_IMAGES, TRAIN_NORM_IMAGES)

print("\nCopying training masks...")
copy_masks(TRAIN_MASKS, TRAIN_NORM_MASKS)

print("\nNormalizing validation images...")
normalize_images(VAL_IMAGES, VAL_NORM_IMAGES)

print("\nCopying validation masks...")
copy_masks(VAL_MASKS, VAL_NORM_MASKS)

print("\nDone!")
