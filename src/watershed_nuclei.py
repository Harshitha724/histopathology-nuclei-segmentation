import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# =========================
# Paths
# =========================
MASK_DIR = "outputs/predictions"
OUTPUT_DIR = "outputs/quantification/watershed_predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Parameters
# =========================
MIN_AREA = 20
MIN_DISTANCE = 8


# =========================
# Process masks
# =========================
all_measurements = []
image_summary = []


mask_files = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
])


for filename in mask_files:

    mask_path = os.path.join(MASK_DIR, filename)

    mask = np.array(Image.open(mask_path).convert("L"))

    # Binary mask
    binary = mask > 127

    # Remove tiny noise
    binary = ndimage.binary_opening(binary)

    # Distance transform
    distance = ndimage.distance_transform_edt(binary)

    # Find local maxima inside nuclei
    coordinates = peak_local_max(
        distance,
        min_distance=MIN_DISTANCE,
        labels=binary
    )

    # Create marker image
    markers = np.zeros_like(binary, dtype=np.int32)

    for i, (y, x) in enumerate(coordinates, start=1):
        markers[y, x] = i

    # Watershed
    labels = watershed(
        -distance,
        markers,
        mask=binary
    )

    # Measure individual nuclei
    props = regionprops(labels)

    nuclei_count = 0

    for prop in props:

        area = prop.area

        if area < MIN_AREA:
            continue

        nuclei_count += 1

        min_row, min_col, max_row, max_col = prop.bbox

        all_measurements.append({
            "image": filename,
            "nucleus_id": prop.label,
            "area": area,
            "centroid_y": prop.centroid[0],
            "centroid_x": prop.centroid[1],
            "major_axis_length": prop.major_axis_length,
            "minor_axis_length": prop.minor_axis_length,
            "eccentricity": prop.eccentricity,
            "solidity": prop.solidity,
            "bbox_height": max_row - min_row,
            "bbox_width": max_col - min_col
        })

    image_summary.append({
        "image": filename,
        "nuclei_count": nuclei_count,
        "nuclear_area": sum(
            m["area"]
            for m in all_measurements
            if m["image"] == filename
        ),
        "nuclear_area_fraction": (
            sum(
                m["area"]
                for m in all_measurements
                if m["image"] == filename
            ) / binary.size
        )
    })


# =========================
# Save results
# =========================

measurements_df = pd.DataFrame(all_measurements)
summary_df = pd.DataFrame(image_summary)

measurements_path = os.path.join(
    OUTPUT_DIR,
    "watershed_nuclei_measurements.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "watershed_image_summary.csv"
)

measurements_df.to_csv(measurements_path, index=False)
summary_df.to_csv(summary_path, index=False)


# =========================
# Print summary
# =========================

print("=" * 50)
print("WATERSHED NUCLEI QUANTIFICATION")
print("=" * 50)

print(f"Images analyzed: {len(summary_df)}")
print(f"Total detected nuclei: {len(measurements_df)}")

if len(summary_df) > 0:

    print(
        f"Mean nuclei per image: "
        f"{summary_df['nuclei_count'].mean():.2f}"
    )

if len(measurements_df) > 0:

    print(
        f"Mean nuclear area: "
        f"{measurements_df['area'].mean():.2f}"
    )

    print(
        f"Median nuclear area: "
        f"{measurements_df['area'].median():.2f}"
    )

    print(
        f"Mean nuclear area fraction: "
        f"{summary_df['nuclear_area_fraction'].mean():.4f}"
    )

print()
print("Saved:")
print(measurements_path)
print(summary_path)
print("=" * 50)
