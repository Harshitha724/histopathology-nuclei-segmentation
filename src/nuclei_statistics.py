import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
SUMMARY_PATH = (
    "outputs/quantification/"
    "watershed_predictions/"
    "watershed_image_summary.csv"
)

MEASUREMENTS_PATH = (
    "outputs/quantification/"
    "watershed_predictions/"
    "watershed_nuclei_measurements.csv"
)

OUTPUT_DIR = "outputs/figures/statistics"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Load data
# =========================
summary = pd.read_csv(SUMMARY_PATH)
measurements = pd.read_csv(MEASUREMENTS_PATH)


# =========================
# Print basic statistics
# =========================
print("=" * 50)
print("NUCLEI STATISTICS")
print("=" * 50)

print(f"Validation patches: {len(summary)}")
print(f"Total nuclei: {len(measurements)}")

print(
    f"Mean nuclei per patch: "
    f"{summary['nuclei_count'].mean():.2f}"
)

print(
    f"Median nuclei per patch: "
    f"{summary['nuclei_count'].median():.2f}"
)

print(
    f"Mean nuclear area: "
    f"{measurements['area'].mean():.2f} px²"
)

print(
    f"Median nuclear area: "
    f"{measurements['area'].median():.2f} px²"
)

print(
    f"Mean eccentricity: "
    f"{measurements['eccentricity'].mean():.3f}"
)

print(
    f"Mean solidity: "
    f"{measurements['solidity'].mean():.3f}"
)


# =========================
# 1. Nuclei count distribution
# =========================
plt.figure(figsize=(8, 5))

plt.hist(
    summary["nuclei_count"],
    bins=20
)

plt.xlabel("Nuclei count per patch")
plt.ylabel("Number of patches")
plt.title("Distribution of Nuclei Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "nuclei_count_distribution.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================
# 2. Nuclear area distribution
# =========================
plt.figure(figsize=(8, 5))

plt.hist(
    measurements["area"],
    bins=40
)

plt.xlabel("Nuclear area (pixels²)")
plt.ylabel("Number of nuclei")
plt.title("Distribution of Nuclear Area")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "nuclear_area_distribution.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================
# 3. Eccentricity distribution
# =========================
plt.figure(figsize=(8, 5))

plt.hist(
    measurements["eccentricity"],
    bins=30
)

plt.xlabel("Eccentricity")
plt.ylabel("Number of nuclei")
plt.title("Distribution of Nuclear Eccentricity")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "eccentricity_distribution.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================
# 4. Nuclear area fraction
# =========================
plt.figure(figsize=(8, 5))

plt.hist(
    summary["nuclear_area_fraction"],
    bins=20
)

plt.xlabel("Nuclear area fraction")
plt.ylabel("Number of patches")
plt.title("Distribution of Nuclear Area Fraction")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "nuclear_area_fraction_distribution.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================
# Done
# =========================
print()
print("Saved plots to:")
print(OUTPUT_DIR)

print("=" * 50)
