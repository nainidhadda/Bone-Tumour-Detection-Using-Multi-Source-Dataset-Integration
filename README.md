# Bone Cancer Detection from X-ray Images

## Project Objective

This academic and research-only project investigates binary bone X-ray classification:

- `cancer = 1`: cancer-associated finding
- `cancer = 0`: normal/no-cancer class

The project is an AI-assisted screening and decision-support prototype, not a clinical diagnostic system. It must not be used for patient diagnosis or treatment decisions.

## Datasets

The project combines two local source datasets:

- BTXRD
- Dataset2

The source integration is preserved at `data/processed/combined/`. The frozen, leakage-free dataset for all ML training and evaluation is `data/processed/final/`:

```text
data/processed/final/
├── train/
│   ├── images/
│   └── metadata.csv
├── valid/
│   ├── images/
│   └── metadata.csv
└── test/
    ├── images/
    └── metadata.csv
```

All final images are PNG files at `224x224`. Metadata contains `filename`, `cancer`, `dataset`, and `split`.

## Final Dataset Status

| Split | Images |
|---|---:|
| Train | 10,052 |
| Validation | 1,084 |
| Test | 1,067 |
| **Total** | **12,203** |

The final dataset is complete and self-contained. Every image is PNG at `224x224`, and every metadata row maps to an image with a binary cancer label. SHA-256 cleanup removed 353 lower-priority cross-split copies: 346 Dataset2 groups and 7 BTXRD groups. Same-split exact duplicates remain unchanged because they do not create cross-split leakage.

Final cross-split duplicate counts are zero for Train/Valid, Train/Test, and Valid/Test. Labels and image contents were not changed.

Dataset preparation and integration are complete. ML training is the next project stage; no model results are claimed yet.

## Notebook Workflow

1. `01_raw_dataset_inspection.ipynb`
2. `02_btxrd_preprocessing.ipynb`
3. `03_dataset2_preprocessing.ipynb`
4. `04_dataset_integration.ipynb`
5. `05_duplicate_analysis.ipynb`
6. `06_baseline_cnn.ipynb`
7. `07_transfer_learning.ipynb`
8. `08_evaluation.ipynb`
9. `09_gradcam.ipynb`

The next ML notebook, `06_baseline_cnn.ipynb`, must read only from `data/processed/final/`. It must not regenerate data, remove duplicates, or alter labels.

## Git and Data Policy

Raw datasets, processed images, metadata generated from local data, trained models, generated results, temporary dataset folders, and ZIP archives are excluded from Git. Source code, notebooks, configuration, and documentation remain eligible for tracking.

The datasets must be obtained separately and placed in the expected local directories. The preprocessing and integration notebooks can recreate the combined dataset from raw data when required.
