import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from skimage.measure import regionprops
from tqdm import tqdm

from src.unet import UNet


# -----------------------------
# Configuration
# -----------------------------
IMAGE_DIR = "data/processed/val/normalized/images"

MODEL_PATH = "outputs/models/best_unet.pth"

OUTPUT_DIR = "outputs/quantification"

THRESHOLD = 0.5
MIN_AREA = 20


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Load model
# -----------------------------
model = UNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded best model from epoch:", checkpoint["epoch"])


# -----------------------------
# Create output directory
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Find validation images
# -----------------------------
image_paths = sorted(
    [
        os.path.join(IMAGE_DIR, filename)
        for filename in os.listdir(IMAGE_DIR)
        if filename.endswith(".png")
    ]
)


# -----------------------------
# Quantification
# -----------------------------
all_nuclei = []
image_summary = []


for image_path in tqdm(
    image_paths,
    desc="Quantifying nuclei"
):

    filename = os.path.basename(image_path)

    # Load image
    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    # Convert to tensor
    image_tensor = (
        torch.tensor(
            image,
            dtype=torch.float32
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    image_tensor = image_tensor.to(device)

    # -----------------------------
    # U-Net prediction
    # -----------------------------
    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities >= THRESHOLD
        ).float()

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # -----------------------------
    # Connected components
    # -----------------------------
    labeled_mask, num_objects = ndimage.label(
        prediction
    )

    regions = regionprops(labeled_mask)


    nuclei_count = 0
    total_area = 0


    for region in regions:

        area = region.area

        # Remove very small noisy regions
        if area < MIN_AREA:
            continue

        nuclei_count += 1
        total_area += area

        all_nuclei.append(
            {
                "image": filename,
                "nucleus_id": nuclei_count,
                "area_pixels": area,
                "centroid_y": region.centroid[0],
                "centroid_x": region.centroid[1],
                "bbox_min_y": region.bbox[0],
                "bbox_min_x": region.bbox[1],
                "bbox_max_y": region.bbox[2],
                "bbox_max_x": region.bbox[3],
                "major_axis_length": region.major_axis_length,
                "minor_axis_length": region.minor_axis_length,
                "eccentricity": region.eccentricity,
                "solidity": region.solidity,
            }
        )


    # -----------------------------
    # Image-level summary
    # -----------------------------
    image_area = prediction.shape[0] * prediction.shape[1]

    nuclear_density = nuclei_count / image_area

    nuclear_area_fraction = total_area / image_area

    mean_area = (
        total_area / nuclei_count
        if nuclei_count > 0
        else 0
    )

    image_summary.append(
        {
            "image": filename,
            "nuclei_count": nuclei_count,
            "total_nuclear_area": total_area,
            "mean_nuclear_area": mean_area,
            "nuclear_density": nuclear_density,
            "nuclear_area_fraction": nuclear_area_fraction,
        }
    )


# -----------------------------
# Save results
# -----------------------------
nuclei_df = pd.DataFrame(all_nuclei)

summary_df = pd.DataFrame(image_summary)


nuclei_path = os.path.join(
    OUTPUT_DIR,
    "nuclei_measurements.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "image_summary.csv"
)


nuclei_df.to_csv(
    nuclei_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# -----------------------------
# Print summary
# -----------------------------
print("\n" + "=" * 50)
print("NUCLEI QUANTIFICATION")
print("=" * 50)

print("Images analyzed:", len(summary_df))

print(
    "Total detected nuclei:",
    len(nuclei_df)
)

print(
    "Mean nuclei per image:",
    summary_df["nuclei_count"].mean()
)

print(
    "Mean nuclear area:",
    nuclei_df["area_pixels"].mean()
)

print(
    "Median nuclear area:",
    nuclei_df["area_pixels"].median()
)

print(
    "Mean nuclear area fraction:",
    summary_df["nuclear_area_fraction"].mean()
)

print("\nSaved:")
print(nuclei_path)
print(summary_path)

print("=" * 50)
