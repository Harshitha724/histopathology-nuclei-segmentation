# Histopathology Nuclei Segmentation

An end-to-end deep learning pipeline for nuclei segmentation and quantification in H&E-stained histopathology images using U-Net.

## Overview

Nuclei segmentation is an important step in computational pathology because it enables automated analysis of nuclear morphology and distribution.

This project implements a patch-based nuclei segmentation pipeline using the MoNuSeg dataset. The workflow includes XML annotation conversion, dataset splitting, patch extraction, H&E stain normalization, data augmentation, U-Net segmentation, model evaluation, watershed-based nuclei separation, and nuclei quantification.

## Pipeline

```text
MoNuSeg H&E Images
        ↓
XML Annotations → Binary Masks
        ↓
Image-level Train / Validation Split
        ↓
256 × 256 Patch Extraction
        ↓
Macenko Stain Normalization
        ↓
Data Augmentation
        ↓
U-Net Training
        ↓
Segmentation Evaluation
        ↓
U-Net Predictions
        ↓
Watershed-based Nuclei Separation
        ↓
Nuclei Quantification & Statistics

'''

## Dataset

This project uses the **MoNuSeg 2018 Training Dataset**, a dataset developed for nuclei segmentation in multi-organ histopathology images.

The dataset contains:

- H&E-stained histopathology tissue images
- Manually annotated nuclear boundaries
- Images acquired at approximately **40× magnification**
- XML annotation files containing polygon coordinates for individual nuclei
- Multiple tissue types and organs

The XML annotations were converted into binary segmentation masks before model training.

### Dataset Source

Official MoNuSeg dataset page:

https://monuseg.grand-challenge.org/Data/

The dataset is **not included in this repository**.

### Dataset Preparation

The downloaded training data contained:

- **37 annotated tissue images**
- Corresponding XML annotation files
- Corresponding H&E tissue images

The dataset was split at the **original image level** into training and validation sets before patch extraction to avoid having patches from the same original image appear in both sets.

The resulting split was:

- **30 training images**
- **7 validation images**

### Patch Extraction

The images were divided into overlapping patches using:

- Patch size: **256 × 256 pixels**
- Stride: **128 pixels**

Patches were filtered to retain regions containing sufficient tissue or nuclear content.

The resulting dataset contained:

- **1,080 training patches**
- **252 validation patches**
- **1,332 total patches**

## Preprocessing

### XML Annotation to Binary Mask

Each nucleus annotation is represented as a polygon in the XML files.

The polygon coordinates were converted into binary masks:

```text
Background → 0
Nucleus    → 255
