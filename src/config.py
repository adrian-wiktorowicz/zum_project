# config stuff

import os
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUDIO_DIR = DATA_DIR / "audio"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# jamendo api
JAMENDO_CLIENT_ID = os.environ.get("JAMENDO_CLIENT_ID", "184df574")
JAMENDO_API_BASE = "https://api.jamendo.com/v3.0"
JAMENDO_RATE_LIMIT = 5
JAMENDO_MAX_RETRIES = 3
JAMENDO_RETRY_BACKOFF = 2.0

# mtg dataset urls
MTG_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data"
MTG_GENRE_TSV_URL = f"{MTG_GITHUB_RAW_BASE}/autotagging_genre.tsv"
MTG_META_TSV_URL = f"{MTG_GITHUB_RAW_BASE}/raw.meta.tsv"

# classes
TARGET_CLASSES = ["house", "techno", "trance", "drum_and_bass"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

# tag synonyms for matching
TAG_SYNONYMS = {
    "house": ["house"],
    "techno": ["techno"],
    "trance": ["trance"],
    "drum_and_bass": [
        "drum and bass",
        "drum & bass", 
        "drum_and_bass",
        "dnb",
        "drum'n'bass",
        "drum n bass",
        "drumandbass"
    ]
}

# dataset limits
MAX_PER_CLASS = 1000
MIN_PER_CLASS = 300
RANDOM_SEED = 42

# audio params
SAMPLE_RATE = 22050
CLIP_DURATION_SEC = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION_SEC

# mel spectrogram
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

# data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# CNN training
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 30
CNN_LEARNING_RATE = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_EARLY_STOPPING_PATIENCE = 7

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5

# AST (transformers)
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_BATCH_SIZE = 16
AST_EPOCHS = 10
AST_LEARNING_RATE = 1e-5
AST_WARMUP_STEPS = 100

# augmentation
USE_SPECAUGMENT = True
FREQ_MASK_PARAM = 10
TIME_MASK_PARAM = 20
