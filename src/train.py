import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import NucleiDataset
from src.unet import UNet
from src.loss import BCEDiceLoss


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4

TRAIN_IMAGE_DIR = "data/processed/train/normalized/images"
TRAIN_MASK_DIR = "data/processed/train/normalized/masks"

VAL_IMAGE_DIR = "data/processed/val/normalized/images"
VAL_MASK_DIR = "data/processed/val/normalized/masks"

MODEL_DIR = "outputs/models"


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Dataset
# -----------------------------
train_dataset = NucleiDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    train=True
)

val_dataset = NucleiDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    train=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# -----------------------------
# Model
# -----------------------------
model = UNet().to(device)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)


# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Create model directory
# -----------------------------
import os

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Training
# -----------------------------
best_val_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    # ===== Training =====
    model.train()

    train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"
    )

    for images, masks in train_progress:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= len(train_loader)


    # ===== Validation =====
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )


    # ===== Save best model =====
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model_path = os.path.join(
            MODEL_DIR,
            "best_unet.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            model_path
        )

        print(
            f"Saved best model → {model_path}"
        )


print("\nTraining complete!")
print("Best validation loss:", best_val_loss)
