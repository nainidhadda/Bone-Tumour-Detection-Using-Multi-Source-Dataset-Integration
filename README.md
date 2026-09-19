# Bone Cancer Detection from X-ray Images Using Deep Learning

## Project Objective

This academic machine learning project investigates deep learning methods for identifying cancer-associated patterns in bone X-ray images.

The project is intended for coursework and research experimentation. The system is being developed as an AI-assisted screening and decision-support prototype rather than a clinical diagnostic system.

## Current ML Task

The initial machine learning task is binary image classification:

- Cancer/Tumor
- Normal

Two image datasets are being used:

- BTXRD
- Dataset 2

The datasets are processed and standardized before being combined for model development.

## Dataset Preparation

The current integrated dataset contains:

| Split | Images |
|---|---:|
| Train | 10,052 |
| Validation | 1,257 |
| Test | 1,247 |
| **Total** | **12,556** |

All integrated images are standardized to:

- Image size: `224 × 224`
- Image format: PNG

Metadata is maintained for the integrated dataset, including:

- filename
- cancer label
- dataset source
- split

### Dataset Integrity Checks

The preprocessing stage includes checks for:

- Image dimensions and file formats
- Missing images
- Class distributions
- Duplicate images between datasets
- Duplicate images across dataset splits
- Consistency between image files and metadata

An important dataset limitation was identified during inspection: Dataset 2 contains duplicate images across its predefined train, validation, and test splits. This will be considered when designing the final evaluation methodology and reporting results.

## Data Availability

Raw and processed image datasets are intentionally excluded from Git version control.

The datasets must be obtained separately and placed in the appropriate local directories.

The preprocessing and integration code is maintained in the repository so that team members can reproduce the processed dataset locally.

Expected final local structure:

```text
data/
├── raw/
└── processed/
    └── combined/
        ├── train/
        │   ├── images/
        │   └── metadata.csv
        ├── valid/
        │   ├── images/
        │   └── metadata.csv
        └── test/
            ├── images/
            └── metadata.csv
