import os
import numpy as np
from PIL import Image


PATCH_SIZE = 256
STRIDE = 128

SPLITS = ["train", "val"]

BASE_DIR = "data/processed"


def is_useful_patch(image, mask):
    """
    Decide whether a patch contains enough tissue
    to be useful for training.

    Returns True if:
      - tissue occupies at least 50% of the patch
      OR
      - the patch contains a reasonable amount of nuclei.
    """

    # Convert RGB image to grayscale
    gray = np.mean(image, axis=2)

    # White/background pixels have high intensity.
    tissue_pixels = gray < 230

    tissue_ratio = tissue_pixels.mean()

    # Nuclei pixels are non-zero in the binary mask.
    nucleus_ratio = (mask > 0).mean()

    return tissue_ratio >= 0.50 or nucleus_ratio >= 0.005


def extract_split(split):

    image_dir = os.path.join(
        BASE_DIR, split, "images"
    )

    mask_dir = os.path.join(
        BASE_DIR, split, "masks"
    )

    output_image_dir = os.path.join(
        BASE_DIR, split, "patches", "images"
    )

    output_mask_dir = os.path.join(
        BASE_DIR, split, "patches", "masks"
    )

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith(".tif")
    )

    total_patches = 0
    saved_patches = 0

    for image_file in image_files:

        base_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(
            image_dir,
            image_file
        )

        mask_path = os.path.join(
            mask_dir,
            base_name + ".png"
        )

        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        mask = np.array(
            Image.open(mask_path)
        )

        height, width = mask.shape

        image_patch_count = 0

        for y in range(0, height - PATCH_SIZE + 1, STRIDE):

            for x in range(0, width - PATCH_SIZE + 1, STRIDE):

                total_patches += 1

                image_patch = image[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                mask_patch = mask[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                if not is_useful_patch(
                    image_patch,
                    mask_patch
                ):
                    continue

                patch_name = (
                    f"{base_name}_"
                    f"x{x}_y{y}.png"
                )

                Image.fromarray(
                    image_patch
                ).save(
                    os.path.join(
                        output_image_dir,
                        patch_name
                    )
                )

                Image.fromarray(
                    mask_patch
                ).save(
                    os.path.join(
                        output_mask_dir,
                        patch_name
                    )
                )

                saved_patches += 1
                image_patch_count += 1

        print(
            f"{split}: {base_name} → "
            f"{image_patch_count} patches"
        )

    print(
        f"\n{split.upper()} SUMMARY"
    )
    print(
        f"Possible patches : {total_patches}"
    )
    print(
        f"Saved patches    : {saved_patches}"
    )


def main():

    for split in SPLITS:
        extract_split(split)


if __name__ == "__main__":
    main()
