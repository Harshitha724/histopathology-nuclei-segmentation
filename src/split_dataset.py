import os
import random
import shutil


IMAGE_DIR = "data/raw/MoNuSeg 2018 Training Data/Tissue Images"
MASK_DIR = "data/processed/masks"

TRAIN_IMAGE_DIR = "data/processed/train/images"
TRAIN_MASK_DIR = "data/processed/train/masks"

VAL_IMAGE_DIR = "data/processed/val/images"
VAL_MASK_DIR = "data/processed/val/masks"

VAL_RATIO = 0.2
SEED = 42


def main():

    # Create output directories
    for directory in [
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        VAL_IMAGE_DIR,
        VAL_MASK_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    # Get images
    image_files = sorted(
        f for f in os.listdir(IMAGE_DIR)
        if f.endswith(".tif")
    )

    # Keep only images that have masks
    valid_images = []

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        mask_file = base_name + ".png"

        if os.path.exists(os.path.join(MASK_DIR, mask_file)):
            valid_images.append(image_file)
        else:
            print(f"Missing mask: {image_file}")

    # Reproducible shuffle
    random.seed(SEED)
    random.shuffle(valid_images)

    # Calculate validation size
    val_count = round(len(valid_images) * VAL_RATIO)

    val_files = valid_images[:val_count]
    train_files = valid_images[val_count:]

    print(f"Total images : {len(valid_images)}")
    print(f"Training     : {len(train_files)}")
    print(f"Validation   : {len(val_files)}")

    # Copy files
    for image_file in train_files:

        base_name = os.path.splitext(image_file)[0]
        mask_file = base_name + ".png"

        shutil.copy2(
            os.path.join(IMAGE_DIR, image_file),
            os.path.join(TRAIN_IMAGE_DIR, image_file)
        )

        shutil.copy2(
            os.path.join(MASK_DIR, mask_file),
            os.path.join(TRAIN_MASK_DIR, mask_file)
        )

    for image_file in val_files:

        base_name = os.path.splitext(image_file)[0]
        mask_file = base_name + ".png"

        shutil.copy2(
            os.path.join(IMAGE_DIR, image_file),
            os.path.join(VAL_IMAGE_DIR, image_file)
        )

        shutil.copy2(
            os.path.join(MASK_DIR, mask_file),
            os.path.join(VAL_MASK_DIR, mask_file)
        )

    print("\nDataset split complete.")


if __name__ == "__main__":
    main()
