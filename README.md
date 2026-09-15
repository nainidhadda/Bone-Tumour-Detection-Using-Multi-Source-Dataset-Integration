# Bone Cancer Detection from X-ray Images Using Deep Learning

## Project Objective

This academic machine learning project investigates deep learning methods for detecting bone cancer from X-ray images. The project is intended for coursework and research experimentation.

## Current Scope

The initial task is binary image classification with two classes:

- Cancer/Tumor
- Normal

The dataset contains approximately 8,811 bone X-ray images and is organized into train, validation, and test folders. Dataset files are kept outside version control and must not be committed to Git.

## Planned ML Workflow

The project will be developed incrementally over a 10-day college project timeline:

1. Data analysis and dataset inspection
2. Image preprocessing
3. Stratified sampling
4. Class imbalance analysis and handling
5. Image augmentation
6. CNN baseline
7. Transfer learning with ResNet50
8. Transfer learning with EfficientNetB0
9. Model comparison
10. Grad-CAM explainability
11. Streamlit deployment

SMOTE will only be considered later where technically appropriate for the data representation. No model training or experiment results are included in this initial setup.

## Project Structure

- `data/`: local raw and processed datasets
- `notebooks/`: ordered analysis and experiment notebooks
- `src/`: planned reusable Python modules
- `models/`: locally saved trained models
- `results/`: locally generated figures, metrics, and confusion matrices
- `app/`: planned Streamlit application

## Academic and Medical Disclaimer

This project is for academic and research purposes only. It is not a medical diagnostic tool and must not be used to diagnose, treat, or make clinical decisions about any person.
