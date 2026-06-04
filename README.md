# Bangla Handwritten Word Recognition System

A deep learning system that recognises handwritten Bangla characters and words. It uses a **DenseNet121** backbone transfer learning on the **BanglaLekha-Isolated** dataset, a custom connected-component word segmentation pipeline, experiment tracking via **MLflow**, and an interactive **Streamlit** front-end containerised with **Docker**.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset Preparation](#dataset-preparation)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [MLflow Tracking](#mlflow-tracking)
7. [Word Segmentation Strategy](#word-segmentation-strategy)
8. [Streamlit UI](#streamlit-ui)
9. [Configuration (.env)](#configuration-env)
10. [Training - Local vs Modal Cloud](#training--local-vs-modal-cloud)
11. [Docker - Build & Run](#docker--build--run)
12. [Limitations](#limitations)
13. [Possible Improvements](#possible-improvements)

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
│   └── labels.json
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
| **Image decode** | `tf.image.decode_image` with `channels=3` (forces RGB) |
| **Resize** | Bilinear resize to `224 × 224` pixels (configurable via `TRAIN__IMAGE_SIZE`) |
| **Normalise** | Pixel values scaled to `[0, 1]` by dividing by `255.0` |
| **Train-time augmentation** | Random rotation `±5°`, random zoom `±10%`, random translation `±5%`, random contrast `±10%` |
| **Validation** | No augmentation — raw normalised images only |

### Class sampling

Set `TRAIN__SAMPLE_PER_CLASS`  to `None` (or `null` in `.env`) to use the full dataset or set 100, 500 etc for faster training.

### Train / Validation split

An 80 / 20 stratified split is applied with `sklearn.model_selection.train_test_split` using the global `SEED` for reproducibility.

---

## Model Architecture

The classifier is a **transfer-learning** model built on top of **DenseNet121** pre-trained on ImageNet.

```
Input (224 × 224 × 3)
    │
    ▼
DenseNet121 (frozen, weights="imagenet")
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dropout (0.3)
    │
    ▼
Dense (256, activation="relu")
    │
    ▼
Dropout (0.3)
    │
    ▼
Dense (50, activation="softmax")   ← 50 Bangla character classes
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
1. load_label_mapping()    → reads root labels.json, builds class_dir → index map
2. prepare_datasets()      → scans image folders, splits into train/val tf.data pipelines
3. build_model()           → constructs the DenseNet121-based classifier
4. train()                 → fits the model with callbacks
5. save_model()            → writes models/model.keras and models/labels.json
6. log to MLflow           → params, best-epoch metrics, model artifact, labels.json
```

---

## MLflow Tracking

This project supports two MLflow tracking approaches depending on your workflow — a **remote DagsHub server** for persistent, shareable experiment history, or a **local MLflow server** for quick, offline experimentation.

### What gets logged (both options)

| Item | Type |
|------|------|
| `image_size`, `batch_size`, `epochs`, `learning_rate`, `seed`, `sample_per_class` | Parameters |
| `best_epoch`, `best_val_accuracy`, `best_val_loss`, `best_train_accuracy`, `best_train_loss` | Metrics |
| `model.keras` | Model artifact |
| `labels.json` | Artifact (alongside model) |

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

The segmentation pipeline in `segmentation.py` splits a drawn Bangla word into individual character images using **connected-component analysis**.

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

### Tuning from the UI

The Streamlit app exposes an **"Advanced Segmentation Settings"** panel in word mode:

| Control | Effect |
|---------|--------|
| **Merge gap threshold** slider (1%–20%) | Increase if one character is split into multiple segments; decrease if two characters are merged together |
| **Show segmentation bounding boxes** checkbox | Overlays coloured rectangles on the canvas to visualise what was detected |

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
| **Debug overlay** | Coloured bounding boxes drawn over detected character segments |
| **Character breakdown grid** | Each detected character crop is shown with its predicted label and confidence (up to 5 columns per row) |
| **Clear button** | Resets the canvas and results without a full page reload |

> The predictor model is loaded once via `@st.cache_resource` and reused across all reruns, keeping inference fast.

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
TRAIN__IMAGE_SIZE=224
TRAIN__BATCH_SIZE=32
TRAIN__EPOCHS=30
TRAIN__LEARNING_RATE=0.0001
TRAIN__SAMPLE_PER_CLASS=100        # Set to None for full dataset
TRAIN__MODEL_SAVE_PATH=models/model.keras
TRAIN__SAVE_JSON_PATH=models/labels.json

# ── MLflow / DagsHub ────────────────────────────────────────────────────
MLFLOW__TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW__REGISTRY_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW__EXPERIMENT_NAME=Bengali_Word_Recognition
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>

# ── Inference ───────────────────────────────────────────────────────────
PREDICT__MODEL_PATH=models/model.keras
PREDICT__LABELS_JSON_PATH=models/labels.json
PREDICT__IMAGE_SIZE=224
```

> **Note:** The double underscore `__` is used as the nested delimiter for Pydantic-Settings (e.g. `TRAIN__BATCH_SIZE` maps to `settings.train.batch_size`). `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are single-level because they are read directly by the MLflow client from the environment.

---

## Training — Local vs Modal Cloud

### Option A — Local Training

**Prerequisites:**
- Python 3.10+
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
| **Frozen backbone** | The DenseNet121 backbone is never fine-tuned (only the top layers are trained). This limits the model's ability to adapt to handwriting-specific low-level features. |

---

## Possible Improvements

| Area | Improvement |
|------|-------------|
| **Model** | Unfreeze the top layers of DenseNet121 after initial training (two-phase fine-tuning) to further improve accuracy |
| **Model** | Experiment with lighter backbones (MobileNetV3, EfficientNetV2-S) for faster inference |
| **Model** | Add label smoothing and mixup augmentation to reduce over-confidence |
| **Dataset** | Include Bangla numerals (০–৯) and common diacritics (vowel signs / matras) |
| **Segmentation** | Replace the connected-component heuristic with a learned line-segmentation model (e.g., a lightweight CNN trained to detect character boundaries) |
| **Conjuncts** | Build a separate conjunct-character classifier or a sequence model (CTC / attention) to handle full words end-to-end |
| **MLflow** | Log per-epoch metrics (not just best-epoch) for richer experiment analysis on DagsHub |
| **Reproducibility** | Pin the `BanglaLekha-Isolated` dataset version and add a `DVC` config to track data alongside model artifacts |