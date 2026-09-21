# Bone Cancer Detection Model Training Report

## 1. Objective

This project aims to build a binary classification model for bone cancer detection from X-ray images. The final model is based on a ResNet-50 architecture with ImageNet-pretrained weights, adapted for cancer-vs-normal classification.

The workflow followed in this project includes dataset preparation, augmentation, model training, validation analysis, checkpoint comparison, and external dataset testing.

---

## 2. Project Files

The following files were used in the training workflow:

- [train_resnet50.py](train_resnet50.py) — main ResNet-50 training script
- [augment_dataset.py](augment_dataset.py) — dataset augmentation script
- [predict_external.py](predict_external.py) — external dataset inference script
- [bone_cancer_pipeline.py](bone_cancer_pipeline.py) — consolidated workflow runner
- [MODEL_TRAINING_PROGRESS.md](MODEL_TRAINING_PROGRESS.md) — training summary report

---

## 3. Dataset Preparation

The original bone dataset was first inspected and loaded into the project. The augmentation step created a second version of the dataset with additional transformed training images to improve model generalization.

### Datasets used

- `bone_dataset` — original dataset
- `bone_dataset_augmented` — augmented version used for training comparison

### Augmentation command

```powershell
cd "D:\BONE PROJECT BY AP\bone-cancer-detection-repo\bone-cancer-detection"

.\.venv\Scripts\python.exe augment_dataset.py \
  --input-dir bone_dataset \
  --output-dir bone_dataset_augmented \
  --copies 2 \
  --seed 42
```

This script copies each image and applies light image transformations such as horizontal flipping, rotation, brightness/contrast adjustments, and minor shift augmentation while preserving the label structure.

---

## 4. Model Training Setup

The core model used in this project is ResNet-50 initialized with ImageNet-pretrained weights.

### Model architecture used

```python
from torchvision import models
from torch import nn

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
```

### Training loop summary

The training script loads image-label pairs from the CSV label files and performs binary classification using BCEWithLogitsLoss.

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_loss += loss.item() * images.size(0)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total
```

---

## 5. Training on the Original Dataset

The original model was trained on the base dataset using a standard image size and validation-based checkpoint saving strategy.

### Command used

```powershell
cd "D:\BONE PROJECT BY AP\bone-cancer-detection-repo\bone-cancer-detection"

.\.venv\Scripts\python.exe train_resnet50.py \
  --data-dir bone_dataset \
  --epochs 10 \
  --batch-size 16 \
  --output resnet50_bone_cancer.pt
```

### Result

- Best validation accuracy: 0.9819
- Checkpoint saved: `resnet50_bone_cancer.pt`

---

## 6. Training on the Augmented Dataset

The main comparison experiment was performed using the augmented dataset to improve learning diversity and generalization.

### Command used for the 20-epoch run

```powershell
cd "D:\BONE PROJECT BY AP\bone-cancer-detection-repo\bone-cancer-detection"

.\.venv\Scripts\python.exe train_resnet50.py \
  --data-dir bone_dataset_augmented \
  --epochs 20 \
  --batch-size 20 \
  --output resnet50_bone_cancer_augmented_20ep_20bs.pt
```

### Training comparison summary

| Model | Dataset | Epochs | Batch Size | Best Validation Accuracy |
|---|---|---:|---:|---:|
| Original pretrained model | bone_dataset | 10 | 16 | 0.9819 |
| Augmented pretrained model | bone_dataset_augmented | 10 | 16 | 0.9762 |
| 20-epoch augmented model | bone_dataset_augmented | 20 | 20 | 0.9887 |
| 20-epoch baseline model | bone_dataset_augmented | 20 | 20 | 0.9637 |

---

## 7. Final Winner

The strongest result in this study was achieved by the augmented dataset experiment with 20 epochs and a batch size of 20.

### Selected checkpoint

- File: `resnet50_bone_cancer_augmented_20ep_20bs.pt`
- Best validation accuracy: `0.9887`

This checkpoint represents the best-performing model in the verified training history and should be considered the final candidate for downstream deployment and external testing.

---

## 8. External BTXRD Testing

After training, the selected model was used to generate predictions on an external dataset located at the placeholder path below.

### External dataset path

```text
D:\BTXRD_PROCESSED\BTXRD_processed\images
```

### Prediction command

```powershell
cd "D:\BONE PROJECT BY AP\bone-cancer-detection-repo\bone-cancer-detection"

.\.venv\Scripts\python.exe predict_external.py \
  --data-dir "D:\BTXRD_PROCESSED\BTXRD_processed\images" \
  --checkpoint resnet50_bone_cancer_augmented_20ep_20bs.pt \
  --output btxrd_predictions_best.csv
```

This script loads the saved checkpoint, runs inference across all valid image files, and writes the prediction output to a CSV file with the file path, class label, and cancer probability.

---

## 9. Operational Notes for Team Members

To reproduce the workflow on another machine:

1. Clone or download the project folder.
2. Open the folder in a Python environment.
3. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

4. Install the required dependencies:

```powershell
pip install torch torchvision pillow
```

5. Run the augmentation step, then the training step, and finally the external prediction step.

---

## 10. Conclusion

The training results indicate that dataset augmentation combined with a longer training schedule improved performance. The 20-epoch augmented run produced the best validation accuracy and is therefore selected as the final model for this project stage.

This model is the recommended checkpoint for further evaluation on external datasets and for any team-level deployment comparison.
