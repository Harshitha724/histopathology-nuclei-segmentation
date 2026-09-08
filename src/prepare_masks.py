import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


# Paths
IMAGE_DIR = "data/raw/MoNuSeg 2018 Training Data/Tissue Images"
ANNOTATION_DIR = "data/raw/MoNuSeg 2018 Training Data/Annotations"
MASK_DIR = "data/processed/masks"

os.makedirs(MASK_DIR, exist_ok=True)


def xml_to_mask(xml_path, image_size):
    """
    Convert MoNuSeg XML nucleus boundaries into a binary mask.

    Background = 0
    Nucleus    = 1
    """

    width, height = image_size

    # Create empty mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Read XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Each Region corresponds to one nucleus
    for region in root.iter("Region"):

        vertices = region.find("Vertices")

        if vertices is None:
            continue

        polygon = []

        for vertex in vertices.findall("Vertex"):
            x = float(vertex.get("X"))
            y = float(vertex.get("Y"))

            polygon.append((x, y))

        # Need at least 3 points to form a polygon
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=1)

    return np.array(mask, dtype=np.uint8)


def main():

    xml_files = [
        f for f in os.listdir(ANNOTATION_DIR)
        if f.endswith(".xml")
    ]

    print(f"Found {len(xml_files)} annotation files.")

    for i, xml_file in enumerate(xml_files):

        base_name = os.path.splitext(xml_file)[0]

        image_path = os.path.join(
            IMAGE_DIR,
            base_name + ".tif"
        )

        xml_path = os.path.join(
            ANNOTATION_DIR,
            xml_file
        )

        # Make sure corresponding image exists
        if not os.path.exists(image_path):
            print(f"Skipping {xml_file}: image not found")
            continue

        # Read image dimensions
        image = Image.open(image_path)

        # Generate mask
        mask = xml_to_mask(
            xml_path,
            image.size
        )

        # Save mask
        mask_path = os.path.join(
            MASK_DIR,
            base_name + ".png"
        )

        Image.fromarray(mask * 255).save(mask_path)

        print(
            f"[{i + 1}/{len(xml_files)}] "
            f"{base_name} → mask generated"
        )

    print("\nMask generation complete.")


if __name__ == "__main__":
    main()
