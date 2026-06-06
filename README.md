# Bangla Handwritten Word Recognition System

A deep learning system that recognises handwritten Bangla characters and words. It uses a **custom 4-block CNN** trained from scratch on **grayscale** images from the **BanglaLekha-Isolated** dataset, a custom connected-component word segmentation pipeline, experiment tracking via **MLflow**, and an interactive **Streamlit** front-end containerised with **Docker**.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset Preparation](#dataset-preparation)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Experiment Results](#experiment-results)
7. [MLflow Tracking](#mlflow-tracking)
8. [Word Segmentation Strategy](#word-segmentation-strategy)
9. [Streamlit UI](#streamlit-ui)
10. [Configuration (.env)](#configuration-env)
11. [Training - Local vs Modal Cloud](#training--local-vs-modal-cloud)
12. [Docker - Build & Run](#docker--build--run)
13. [Limitations](#limitations)
14. [Possible Improvements](#possible-improvements)

---

## Project Structure

```
Bangla-Handwritten-Word-Recognition-System/
├── BanglaLekha-Isolated/          # Raw dataset (not committed to git)
│   └── Images/
│       ├── 1/                     # Folder "1" → class index 0 (অ)
│       ├── 2/                     # Folder "2" → class index 1 (আ)
│       └── ...                    # 50 class folders in total
├── models/                        # Saved model artifacts (created after training)
│   ├── model.keras
│   ├── labels.json
│   └── train_history.csv          # Per-epoch train/val metrics (created after training)
├── app.py                         # Streamlit front-end
├── config.py                      # Pydantic settings (reads from .env)
├── initial_setup.py               # Modal: upload dataset to cloud volume
├── prediction.py                  # Inference helper
├── requirements.txt               # Python dependencies
├── run.py                         # Modal: run training on cloud GPU
├── segmentation.py                # Word → characters segmentation pipeline
├── train.py                       # Model training script
├── labels.json                    # Root-level class label mapping (0-indexed)
├── .env                           # Local secrets / overrides (never committed)
├── .dockerignore
├── Dockerfile
└── README.md
```

---

## Dataset Preparation

The project uses the **BanglaLekha-Isolated** dataset, a publicly available collection of isolated Bangla handwritten characters.

### Class mapping

| Root `labels.json` key | Character | Dataset folder |
|:---:|:---:|:---:|
| 0 | অ | `1/` |
| 1 | আ | `2/` |
| … | … | … |
| 49 | ঁ | `50/` |

The dataset contains **50 classes** covering all Bangla vowels and consonants.

> **Note:** The folder names inside `BanglaLekha-Isolated/Images/` are **1-indexed** (`1`, `2`, …, `50`), while `labels.json` is **0-indexed** (`0`, `1`, …, `49`). The training code handles this offset automatically via `class_dir_to_index`.

### Acquiring the dataset

1. Download the **BanglaLekha-Isolated** dataset and place it so the structure matches:
   ```
   BanglaLekha-Isolated/
   └── Images/
       ├── 1/
       ├── 2/
       └── ...
   ```
2. The root-level `labels.json` is already included in this repo.

---

## Preprocessing

All preprocessing is done inside `train.py` using the **`tf.data`** pipeline.

| Step | Detail |
|------|--------|
| **Image decode** | `tf.image.decode_image` with `channels=1` (grayscale — matches the CNN's single-channel input) |
| **Resize** | Bilinear resize to `64 × 64` pixels (configurable via `TRAIN__IMAGE_SIZE`) |
| **Normalise** | Pixel values scaled to `[0, 1]` by dividing by `255.0` |
| **Train-time augmentation** | Random rotation `±5°`, random zoom `±10%`, random translation `±5%`, random contrast `±10%` |
| **Validation** | No augmentation — raw normalised images only |

### Class sampling

Set `TRAIN__SAMPLE_PER_CLASS`  to `None` (or `null` in `.env`) to use the full dataset or set 100, 500 etc for faster training.

### Train / Validation split

An 80 / 20 stratified split is applied with `sklearn.model_selection.train_test_split` using the global `SEED` for reproducibility.

---

## Model Architecture

The classifier is a **custom CNN trained from scratch** on grayscale character images.

```
Input (64 × 64 × 1)   ← grayscale
    │
    ├── Block 1
    │   ├── Conv2D(32, 3×3, ReLU, padding=same)
    │   ├── BatchNormalization
    │   ├── Conv2D(32, 3×3, ReLU, padding=same)
    │   ├── BatchNormalization
    │   ├── MaxPooling2D(2×2)          → 32 × 32
    │   └── Dropout(0.20)
    │
    ├── Block 2
    │   ├── Conv2D(64, 3×3, ReLU, padding=same)
    │   ├── BatchNormalization
    │   ├── Conv2D(64, 3×3, ReLU, padding=same)
    │   ├── BatchNormalization
    │   ├── MaxPooling2D(2×2)          → 16 × 16
    │   └── Dropout(0.25)
    │
    ├── Block 3
    │   ├── Conv2D(128, 3×3, ReLU, padding=same)
    │   ├── BatchNormalization
    │   ├── Conv2D(128, 3×3, ReLU, padding=same)
    │   ├── BatchNormalization
    │   ├── MaxPooling2D(2×2)          → 8 × 8
    │   └── Dropout(0.30)
    │
    ├── Block 4
    │   ├── Conv2D(256, 3×3, ReLU, padding=same)
    │   ├── BatchNormalization
    │   ├── MaxPooling2D(2×2)          → 4 × 4
    │   └── Dropout(0.35)
    │
    ├── GlobalAveragePooling2D
    │
    ├── Dense(256, activation="relu")
    │   ├── BatchNormalization
    │   └── Dropout(0.50)
    │
    └── Dense(50, activation="softmax")   ← 50 Bangla character classes
```

---

## Training Process

Training is orchestrated by the `BengaliWordTrainer` class in `train.py`.

### Callbacks

| Callback | Configuration |
|----------|--------------|
| `EarlyStopping` | Monitors `val_loss`, patience = 5 epochs, restores best weights |
| `ReduceLROnPlateau` | Monitors `val_loss`, reduces LR by ×0.5 after 2 stagnant epochs |
| `ModelCheckpoint` | Saves the best model to `models/model.keras` (only when `val_loss` improves) |

### Execution order

```
1. load_label_mapping()     → reads root labels.json, builds class_dir → index map
2. prepare_datasets()       → scans image folders, splits into train/val tf.data pipelines
3. build_model()            → constructs the custom 4-block grayscale CNN
4. train()                  → fits the model with callbacks
5. save_model()             → writes models/model.keras and models/labels.json
6. save_history_csv()       → writes per-epoch metrics to models/train_history.csv
7. log to MLflow            → params, best-epoch metrics, model artifact, labels.json,
                              and train_history.csv (under artifacts/metrics/)
```

---

## Experiment Results

The table below summarises all training runs conducted so far, sorted by validation accuracy (highest to lowest).  

All runs use:
- `batch_size = 32`
- `seed = 42`
- Full custom 4-block CNN architecture
- All samples from the first 50 classes of the BanglaLekha-Isolated dataset

| Image Size | Epochs | Learning Rate | Best Epoch | **Best Val Accuracy** | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|
| **64** | **20** | **0.001** | **19** | **🏆 95.53%** | **Best overall — compact input, well-tuned LR** |
| 32 | 25 | 0.001 | 22 | 95.14% | Near-best; more epochs needed to close the gap |
| 32 | 20 | 0.001 | 19 | 95.10% | On par with 25-epoch run; 32 px converges quickly |
| 128 | 20 | 0.001 | 16 | 94.41% | Higher resolution; diminishing returns vs. 64 px |
| 128 | 10 | 0.001 | 9 | 93.47% | Too few epochs for 128 px to reach full potential |
| 128 | 20 | 0.0001 | 19 | 92.94% | LR 10× lower; underfits — converges too slowly |

### Key Observations

- 🏆 **`image_size=64, epochs=20, lr=0.001`** achieves the highest validation accuracy of **95.53%** and is the recommended default configuration.
- **32 × 32 images** perform surprisingly well (95.1%) and train faster, making them a good choice when compute time is a constraint.
- **128 × 128 images** do *not* outperform smaller sizes at equivalent epoch counts — the extra spatial detail adds noise for isolated character classification without a commensurate accuracy gain.
- **Learning rate matters more than image resolution:** dropping `lr` from `0.001` → `0.0001` on 128 px cuts accuracy by ~1.5 pp, confirming that a higher learning rate is important for fast convergence with this architecture.
- The **best epoch is consistently near the final epoch** across runs (epochs 16–22 out of 20–25), suggesting that EarlyStopping is *not* triggering early and training could benefit from a few additional epochs.

---

## MLflow Tracking

This project supports two MLflow tracking approaches depending on your workflow — a **remote DagsHub server** for persistent, shareable experiment history, or a **local MLflow server** for quick, offline experimentation.

### What gets logged (both options)

| Item | Type | MLflow path |
|------|------|-------------|
| `image_size`, `batch_size`, `epochs`, `learning_rate`, `seed`, `sample_per_class` | Parameters | — |
| `best_epoch`, `best_val_accuracy`, `best_val_loss`, `best_train_accuracy`, `best_train_loss` | Metrics | — |
| `model.keras` | Model artifact | `artifacts/model/` |
| `labels.json` | Artifact | `artifacts/model/` |
| `train_history.csv` | Per-epoch CSV (epoch, train_loss, train_accuracy, val_loss, val_accuracy) | `artifacts/metrics/` |

---

### Option A — DagsHub (Remote Tracking)

DagsHub provides a free, hosted MLflow tracking URI tied to your repository. Experiments, runs, parameters, metrics, and model artifacts are all stored remotely and accessible via the DagsHub web UI.

#### Required `.env` variables

```dotenv
# DagsHub / MLflow
MLFLOW__TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
MLFLOW__REGISTRY_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
MLFLOW__EXPERIMENT_NAME=Bengali_Word_Recognition

MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

> **Tip:** Your DagsHub token can be generated from **User Settings → Tokens** on DagsHub.

No extra server setup is needed — DagsHub handles everything. This option is recommended when training on Modal Cloud or when you want a persistent, shareable experiment history.

---

### Option B — Local MLflow Server

If you prefer to track experiments entirely on your own machine (no internet, no account required), you can run MLflow locally.

#### Required `.env` variables

```dotenv
# Local MLflow
MLFLOW__TRACKING_URI=http://localhost:5000
MLFLOW__REGISTRY_URI=http://localhost:5000
MLFLOW__EXPERIMENT_NAME=Bengali_Word_Recognition
```

> **Note:** `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are **not needed** when running locally — omit them from your `.env` file entirely.

#### Start the local MLflow UI

```bash
# Install MLflow if not already installed
pip install mlflow

# Launch the tracking server (run from the project root)
mlflow ui --host 0.0.0.0 --port 5000
```

The MLflow UI will be available at **http://localhost:5000** or **http://127.0.0.1:5000**. Keep this terminal running while you train. After training completes, all experiments and artifacts will appear in the browser automatically.

---

## Word Segmentation Strategy

The segmentation pipeline in `segmentation.py` splits a drawn Bangla word into individual character images using **connected-component analysis**. It is implemented as the `BanglaWordSegmenter` class; all parameters are set at construction time and the cached instance is reused across predictions.

### Algorithm (step by step)

```
1. Binary mask
   Convert the canvas image to grayscale, then threshold at pixel value 30.
   Result: a 2-D boolean array where True = ink, False = background.

2. Connected-component labelling
   Use scipy.ndimage.label() to find all distinct blobs of ink pixels.
   Small blobs (< 0.1% of total pixels) are discarded as noise.

3. Horizontal merging
   Bounding boxes whose horizontal gap is ≤ merge_gap pixels are iteratively
   merged. merge_gap = merge_threshold_ratio × canvas_width.
   This handles Bangla characters whose strokes nearly touch or share a
   common matra (top horizontal line).
   The merge loop repeats until no further merges occur (handles chains of
   3+ boxes that chain-merge).

4. Left-to-right sort
   Merged bounding boxes are sorted by their left edge (x_min).
   Bangla is written left-to-right.

5. Crop + square padding
   Each bounding box is expanded by 15% padding on each side (clamped to
   image boundaries), then zero-padded to a square aspect ratio using black
   pixels. The result is a PIL Image ready for the predictor.
```

---

## Streamlit UI

Launch the app locally:

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Features

| Feature | Description |
|---------|-------------|
| **Single Character mode** | Draw one Bangla character on a `300 × 300` canvas; shows predicted character + confidence score |
| **Word mode** | Draw a multi-character Bangla word on a `600 × 220` canvas; the pipeline segments it, classifies each character, and assembles the predicted word |
| **Character breakdown grid** | Each detected character crop is shown with its predicted label and confidence (up to 5 columns per row) |

> The predictor model and segmenter are both loaded once via `@st.cache_resource` and reused across all reruns, keeping inference fast.

---

## Configuration (.env)

All settings are managed through Pydantic-Settings in `config.py` and loaded from a `.env` file. Here are the available variables:

```dotenv
# ── Randomness ──────────────────────────────────────────────────────────
SEED=42

# ── Dataset ─────────────────────────────────────────────────────────────
DATA__DATASET_PATH=BanglaLekha-Isolated/Images
DATA__LABELS_JSON_PATH=labels.json

# ── Training ────────────────────────────────────────────────────────────
TRAIN__IMAGE_SIZE=64               # 64 recommended for the custom CNN (or 128 for more detail)
TRAIN__BATCH_SIZE=32             
TRAIN__EPOCHS=20
TRAIN__LEARNING_RATE=0.001
TRAIN__SAMPLE_PER_CLASS=1000        # Set to None for full dataset
TRAIN__MODEL_SAVE_PATH=models/model.keras
TRAIN__SAVE_JSON_PATH=models/labels.json
TRAIN__SAVE_SUMMARY_PATH=models/train_history.csv

# ── MLflow / DagsHub ────────────────────────────────────────────────────
MLFLOW__TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW__REGISTRY_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW__EXPERIMENT_NAME=Bengali_Word_Recognition
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>

# ── Inference ───────────────────────────────────────────────────────────
PREDICT__MODEL_PATH=models/model.keras
PREDICT__LABELS_JSON_PATH=models/labels.json
PREDICT__IMAGE_SIZE=64             # Must match TRAIN__IMAGE_SIZE
```

> **Note:** The double underscore `__` is used as the nested delimiter for Pydantic-Settings (e.g. `TRAIN__BATCH_SIZE` maps to `settings.train.batch_size`). `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are single-level because they are read directly by the MLflow client from the environment.

---

## Training — Local vs Modal Cloud

### Option A — Local Training

**Prerequisites:**
- Python 3.11
- The `BanglaLekha-Isolated/Images/` folder present locally
- A `.env` file configured (see [Configuration](#configuration-env))

**Steps:**

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Uncomment tensorflow in requirements.txt, then install
#    Edit requirements.txt → uncomment:
#    tensorflow==2.16.1
pip install -r requirements.txt

# 3. Set up your .env file

# 4. Run training
python train.py
```

After training, the best model is saved to `models/model.keras` and `models/labels.json`.

> **Note:** `requirements.txt` has `tensorflow==2.16.1` commented out by default because Modal's base image already includes a GPU-optimised TensorFlow build. When training locally, uncomment that line before installing.

---

### Option B — Modal Cloud Training (GPU)

[Modal](https://modal.com) is a serverless GPU cloud platform. The project uses it to train on an **NVIDIA L40S GPU** without any local GPU requirement. No server setup is needed — Modal handles container orchestration automatically.

#### How it works

- **`initial_setup.py`** — uploads the dataset to a persistent Modal Volume (`datasets-volume`) so the training container can access it.
- **`run.py`** — defines the Modal app that launches training inside a GPU container (`tensorflow/tensorflow:2.16.1-gpu`), mounting the dataset volume and injecting secrets.

#### Prerequisites

1. **Install Modal** locally (only needed on your machine for orchestration):
   ```bash
   pip install modal
   modal setup   # authenticate with your Modal account
   ```

2. **Create a Modal Secret** named `project-config` containing all your `.env` variables (DagsHub credentials, dataset paths, training config, etc.):
   - Go to [modal.com](https://modal.com) → **Secrets** → **New Secret**
   - Name it `project-config`
   - Add all the key-value pairs from your `.env` file

#### Step 1 — Upload dataset to Modal Volume (one-time)

Make sure the zip file `BanglaLekha-Isolated.zip` is present in the project root, then run:

```bash
modal run initial_setup.py
```

This extracts the dataset into the persistent `datasets-volume` volume and also copies `labels.json` into the volume. You only need to do this **once**.

#### Step 2 — Run training on cloud GPU

```bash
modal run run.py
```

This will:
1. Build a Docker image from `tensorflow/tensorflow:2.16.1-gpu`
2. Install all dependencies from `requirements.txt`
3. Mount `datasets-volume` at `/root/datasets`
4. Inject your `project-config` secret as environment variables
5. Execute `python /root/train.py` on an L40S GPU
6. Log all results to DagsHub MLflow automatically

#### `run.py` at a glance

```python
image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.16.1-gpu")
    .run_commands(["python -m pip install --ignore-installed blinker==1.9.0"])
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("train.py", remote_path="/root/train.py")
    .add_local_file("config.py", remote_path="/root/config.py")
)

@app.function(
    image=image,
    volumes={"/root/datasets": volume},   # persistent dataset volume
    gpu="L40S",
    secrets=[modal.Secret.from_name("project-config")],
    timeout=86400
)
def run_training():
    subprocess.run(["python", "/root/train.py"], check=True)
```

> **After training:** Download `models/model.keras` and `models/labels.json` from your DagsHub MLflow run artifacts and place them in the local `models/` folder to run the Streamlit app.

---

## Docker — Build & Run

Docker is used to containerise the **Streamlit inference application** (not training). The image is based on `python:3.11-slim`.

### Build the image

```bash
docker build -t bangla-recognition .
```

> **What happens inside the build:**
> - `libgomp1` and `libglib2.0-0` are installed (required by TensorFlow and Pillow on slim images)
> - `requirements.txt` is installed (layer-cached separately for fast rebuilds)
> - All application code is copied in via `COPY . .`
> - The `BanglaLekha-Isolated/` dataset folder and `.env` are excluded by `.dockerignore`
> - Port `8501` is exposed for Streamlit

### Run the container

The `.env` file is **not** baked into the image for security reasons. Pass it at runtime:

```bash
docker run --rm -p 8501:8501 --env-file .env bangla-recognition
```

Then open **http://localhost:8501** in your browser.

### Full reference

| Command | Description |
|---------|-------------|
| `docker build -t bangla-recognition .` | Build the image |
| `docker run --rm -p 8501:8501 --env-file .env bangla-recognition` | Run with env file |
| `docker ps` | List running containers |
| `docker stop <container-id>` | Stop a running container |
| `docker rmi bangla-recognition` | Remove the image |

> **Important:** The `models/` folder (containing `model.keras` and `labels.json`) must be present **before building** the Docker image, as it is copied into the container via `COPY . .`. If you trained on Modal, download the artifacts from your MLflow run and place them in `models/` first.

---

## Limitations

| Limitation | Detail |
|------------|--------|
| **Isolated characters only** | The model is trained on the BanglaLekha-Isolated dataset, which contains individually written characters — not naturally connected cursive handwriting. Performance on real cursive words may be lower. |
| **No conjunct character support** | Bangla has hundreds of conjunct consonants (যুক্তাক্ষর / juktakkhar). This system does not recognise them; each segment is classified as a single base character. |
| **Segmentation brittleness** | The connected-component segmenter works well for clearly separated characters but can fail when strokes overlap significantly, especially with the মাত্রা (top horizontal line) shared across characters. |
| **Fixed class set** | Only the 50 character classes from BanglaLekha-Isolated are supported. Numerals (০–৯) and common matras (া, ি, ী, ু, ূ, etc.) are not included. |
| **Trained from scratch** | The custom CNN has no pre-trained weights. It requires sufficient data and epochs to converge.
---

## Possible Improvements

| Area | Improvement |
|------|-------------|
| **Dataset** | Include Bangla numerals (০–৯), conjunct consonants (যুক্তাক্ষর / juktakkhar) and common diacritics (vowel signs / matras) |
| **Model** | Add label smoothing and mixup augmentation to reduce over-confidence |
| **Segmentation** | Replace the connected-component heuristic with a learned line-segmentation model (e.g., a lightweight CNN trained to detect character boundaries) |
| **Conjuncts** | Build a separate conjunct-character classifier or a sequence model (CTC / attention) to handle full words end-to-end |
| **Analysis** | Use `train_history.csv` from MLflow artifacts to plot learning curves and detect overfitting/underfitting early |
