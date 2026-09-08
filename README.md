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
