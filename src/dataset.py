from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.augmentation import get_train_transform, get_val_transform


class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.train = train

        self.images = sorted(self.image_dir.glob("*.png"))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        if self.train:
            self.transform = get_train_transform()
        else:
            self.transform = get_val_transform()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name

        # Load image
        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        # Load mask
        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        # Apply the same augmentation to image and mask
        transformed = self.transform(
            image=image,
            mask=mask
        )

        image = transformed["image"]
        mask = transformed["mask"]

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert mask to binary [0, 1]
        mask = (mask > 0).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # Add channel dimension to mask
        mask = np.expand_dims(mask, axis=0)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask