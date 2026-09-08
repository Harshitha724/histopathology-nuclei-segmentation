import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import NucleiDataset
from src.unet import UNet


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 8
THRESHOLD = 0.5

VAL_IMAGE_DIR = "data/processed/val/normalized/images"
VAL_MASK_DIR = "data/processed/val/normalized/masks"

MODEL_PATH = "outputs/models/best_unet.pth"


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
# Validation dataset
# -----------------------------
val_dataset = NucleiDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("Validation samples:", len(val_dataset))


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

print("Loaded model from:", MODEL_PATH)
print("Best validation loss:", checkpoint["val_loss"])
print("Best epoch:", checkpoint["epoch"])


# -----------------------------
# Metric accumulators
# -----------------------------
total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0


# -----------------------------
# Evaluation
# -----------------------------
with torch.no_grad():

    for images, masks in tqdm(
        val_loader,
        desc="Evaluating"
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # Flatten
        predictions = predictions.view(-1)
        masks = masks.view(-1)

        # Confusion matrix components
        tp = (
            (predictions == 1) &
            (masks == 1)
        ).sum().item()

        fp = (
            (predictions == 1) &
            (masks == 0)
        ).sum().item()

        fn = (
            (predictions == 0) &
            (masks == 1)
        ).sum().item()

        tn = (
            (predictions == 0) &
            (masks == 0)
        ).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn


# -----------------------------
# Calculate metrics
# -----------------------------
epsilon = 1e-7

dice = (
    2 * total_tp
    / (
        2 * total_tp
        + total_fp
        + total_fn
        + epsilon
    )
)

iou = (
    total_tp
    / (
        total_tp
        + total_fp
        + total_fn
        + epsilon
    )
)

precision = (
    total_tp
    / (
        total_tp
        + total_fp
        + epsilon
    )
)

recall = (
    total_tp
    / (
        total_tp
        + total_fn
        + epsilon
    )
)

f1 = (
    2 * precision * recall
    / (
        precision
        + recall
        + epsilon
    )
)

accuracy = (
    (total_tp + total_tn)
    / (
        total_tp
        + total_tn
        + total_fp
        + total_fn
        + epsilon
    )
)


# -----------------------------
# Print results
# -----------------------------
print("\n" + "=" * 45)
print("SEGMENTATION RESULTS")
print("=" * 45)

print(f"Dice       : {dice:.4f}")
print(f"IoU        : {iou:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"Accuracy   : {accuracy:.4f}")

print("\nConfusion Matrix:")
print(f"TP: {total_tp}")
print(f"FP: {total_fp}")
print(f"FN: {total_fn}")
print(f"TN: {total_tn}")

print("=" * 45)
