# Cassava Leaf Disease Classification

An image-classification pipeline for the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) Kaggle competition. Given a photo of a cassava leaf, the model predicts one of **5 classes** (4 diseases + healthy). This matters because cassava is a key food-security crop in Africa, and most farmers diagnose disease by eye, this is a fast, phone-based classifier can help catch infections early.

## Classes

| id | class |
|----|-------|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

The training set is **imbalanced** — class 3 (CMD) dominates — which is worth keeping in mind when reading the accuracy numbers.

## Approach

The full pipeline lives in [`cassava_leaf_disease_classification.ipynb`](cassava_leaf_disease_classification.ipynb) and is built on [**tez**](https://github.com/abhishekkrthakur/tez), a lightweight PyTorch training wrapper.

**Pipeline**

1. **Split** — a 90/10 holdout split of `train.csv` (`train_test_split`, `random_state=42`) for validation.
2. **Datasets** — `tez.datasets.ImageDataset` loads images from disk and applies [albumentations](https://albumentations.ai/) transforms.
3. **Augmentation** (train): `RandomResizedCrop(256, 256)`, `Transpose`, `HorizontalFlip`, `VerticalFlip`, `ShiftScaleRotate`, then ImageNet `Normalize`. Validation/test use `CenterCrop` + `Resize(256)` + `Normalize`.
4. **Model** — `torchvision` **ResNet18** (ImageNet-pretrained) with the final FC layer replaced by `Linear(512, 5)` for transfer learning.
5. **Train** — `CrossEntropyLoss`, Adam (`lr=1e-3`), `StepLR` scheduler, mixed-precision (`fp16=True`), `EarlyStopping` on `valid_accuracy` (patience 2). Best weights saved to `model.bin`.
6. **Predict** — run the trained model over `test_images/`, take `argmax`, and write `submission.csv` in the Kaggle format (`image_id,label`).

**Training configuration**

| setting | value |
|---|---|
| backbone | ResNet18 (pretrained) |
| image size | 256 × 256 |
| optimizer | Adam, lr = 1e-3 |
| scheduler | StepLR |
| loss | CrossEntropyLoss |
| batch size | 32 (train) / 64 (valid) |
| epochs | 5 (early stopping, patience 2) |
| precision | fp16 |
| metric | accuracy |

## Repository layout

```
cassava_leaf_disease_classification.ipynb   # end-to-end training + inference notebook
train.csv                                    # image_id -> label (21,397 rows)
label_num_to_disease_map.json                # id -> human-readable class name
sample_submission.csv                        # submission format
train_images/ test_images/ *_tfrecords/      # raw data — NOT tracked (see Data)
```

## Data

The raw image data (`train_images/`, `test_images/`, `*_tfrecords/`, ~2.4 GB) is **not tracked in git** — it's the public competition dataset and is re-downloadable. Only the small metadata files are versioned. Download the data from Kaggle:

```bash
kaggle competitions download -c cassava-leaf-disease-classification
unzip cassava-leaf-disease-classification.zip
```

## Running it

The notebook was developed in **Google Colab** with a GPU runtime (`device="cuda"`). To run it:

1. Open the notebook in Colab (or locally with Jupyter + a CUDA GPU).
2. Install dependencies:
   ```bash
   pip install tez albumentations torch torchvision pandas scikit-learn matplotlib
   ```
3. Make the data available at the paths used in the notebook (`train_images/`, `test_images/`, `train.csv`, `sample_submission.csv`).
4. Run all cells — training writes `model.bin`, inference writes `submission.csv`.

## Possible improvements

- Stronger backbone via `timm` (EfficientNet / ResNeXt) instead of ResNet18.
- K-fold cross-validation and ensembling rather than a single holdout split.
- Class-imbalance handling (weighted loss or sampling) given the CMD skew.
- Test-time augmentation (TTA) and a learning-rate schedule tuned for the backbone.
