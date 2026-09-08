import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.unet import UNet


# -----------------------------
# Configuration
# -----------------------------
IMAGE_DIR = "data/processed/val/normalized/images"
MASK_DIR = "data/processed/val/normalized/masks"

MODEL_PATH = "outputs/models/best_unet.pth"
OUTPUT_DIR = "outputs/figures/predictions"

THRESHOLD = 0.5
NUM_IMAGES = 252


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


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
# Select validation images
# -----------------------------
image_paths = sorted(
    [
        p for p in os.listdir(IMAGE_DIR)
        if p.endswith(".png")
    ]
)

# Take evenly spaced examples
indices = np.linspace(
    0,
    len(image_paths) - 1,
    NUM_IMAGES,
    dtype=int
)


# -----------------------------
# Generate predictions
# -----------------------------
for plot_number, index in enumerate(indices):

    filename = image_paths[index]

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    mask_path = os.path.join(
        MASK_DIR,
        filename
    )

    # Load image
    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    # Load ground truth
    ground_truth = np.array(
        Image.open(mask_path).convert("L")
    )

    # Prepare image
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

    # Prediction
    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities >= THRESHOLD
        ).float()

    prediction = (
        prediction.squeeze()
        .cpu()
        .numpy()
    )

    # Save raw binary prediction mask
    RAW_OUTPUT_DIR = "outputs/predictions"
    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)

    raw_prediction = (
        prediction * 255
    ).astype(np.uint8)

    raw_prediction_path = os.path.join(
        RAW_OUTPUT_DIR,
        filename
    )

    Image.fromarray(raw_prediction).save(
        raw_prediction_path
    )

    print("Saved raw prediction:", raw_prediction_path)