# Satellite Image Classification with a CNN

A deep learning project that trains a convolutional neural network (CNN) in TensorFlow/Keras to classify satellite images into two classes (binary classification).

## Project Structure

```
ts_deeplearning/
├── init.ipynb          # Main notebook: setup, training, evaluation
├── manual.ipynb        # Alternate notebook
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

The following are created when you run the notebook and are **not** stored in this repo (see `.gitignore`):

- `images_dataSAT/` – the image dataset
- `*.tar` – the downloaded dataset archive
- `*.keras` – saved model files (too large for GitHub)
- `training_log.csv` – training history

## Setup

```bash
git clone https://github.com/aspark003/ts_deeplearning.git
cd ts_deeplearning
python -m venv ts_deeplearning_venv
source ts_deeplearning_venv/bin/activate
pip install -r requirements.txt
```

Then open `init.ipynb` in Jupyter or VS Code and run the cells in order.

## Dataset

The notebook downloads `images-dataSAT.tar` automatically and extracts it into `images_dataSAT/`, with one subfolder per class.

- It first tries `skillsnetwork.prepare()`.
- If that fails (for example, when symlinks aren't supported), it falls back to a manual download with `httpx` and extraction with `tarfile`.

## Data Pipeline

- Images resized to **64 × 64**, RGB
- Pixel values rescaled to **0–1**
- **80 / 20** train/validation split
- Training augmentation: rotation, width/height shift, shear, zoom, and horizontal flip
- Validation data is not augmented or shuffled

## Model

A `Sequential` CNN:

| Part | Details |
|------|---------|
| Conv blocks | 6 blocks of `Conv2D (5×5)` → `MaxPooling2D` → `BatchNormalization`, filters 32 → 1024 |
| Pooling | `GlobalAveragePooling2D` |
| Dense blocks | 5 blocks of `Dense` → `BatchNormalization` → `Dropout(0.4)`, units 64 → 1024 |
| Output | `Dense(1, sigmoid)` |

- **Initializer:** He Uniform
- **Optimizer:** Adam, learning rate 0.001
- **Loss:** binary cross-entropy
- **Batch size:** 32
- **Epochs:** up to 10
- **Random seed:** 12345

## Training Callbacks

- **ModelCheckpoint** – saves the best model by validation accuracy to `best_model.keras`
- **EarlyStopping** – stops after 5 epochs without improvement and restores the best weights
- **ReduceLROnPlateau** – halves the learning rate when validation loss stalls (minimum 1e-6)
- **CSVLogger** – writes per-epoch metrics to `training_log.csv`

## Evaluation

After training, the best saved model is reloaded and evaluated on the validation set:

- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1)
- Plot of training vs. validation accuracy

## Requirements

Main libraries: `tensorflow`, `keras`, `numpy`, `scikit-learn`, `matplotlib`, `httpx`, `skillsnetwork`. See `requirements.txt` for the full list.
