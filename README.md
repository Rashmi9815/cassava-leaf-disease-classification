# Cassava Leaf Disease Classification

Work for the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) Kaggle competition — classifying cassava leaf images into 4 disease categories or healthy.

## Labels

| id | class |
|----|-------|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

## Data

The raw image data (`train_images/`, `test_images/`, `*_tfrecords/`, ~2.4 GB) is **not tracked in git** — it's the public competition dataset. Download it from Kaggle:

```bash
kaggle competitions download -c cassava-leaf-disease-classification
```

Tracked in the repo: `train.csv` (labels), `label_num_to_disease_map.json`, `sample_submission.csv`.
