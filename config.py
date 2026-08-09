import os
import torch


# ==========================================================
# PROJECT ROOT
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================================
# DATASET SOURCE
# ==========================================================

DATASET_SOURCE = os.path.join(
    BASE_DIR,
    "FaceForensics++_C23"
)

REAL_VIDEO_DIR = os.path.join(
    DATASET_SOURCE,
    "original"
)

FAKE_VIDEO_DIR = os.path.join(
    DATASET_SOURCE,
    "Deepfakes"
)


# ==========================================================
# EXTRACTED DATASET
# ==========================================================

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_DIR,
    "val"
)

TRAIN_REAL_DIR = os.path.join(
    TRAIN_DIR,
    "real"
)

TRAIN_FAKE_DIR = os.path.join(
    TRAIN_DIR,
    "fake"
)

VAL_REAL_DIR = os.path.join(
    VAL_DIR,
    "real"
)

VAL_FAKE_DIR = os.path.join(
    VAL_DIR,
    "fake"
)


# ==========================================================
# OPENCV DNN FACE DETECTOR
# ==========================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PROTOTXT = os.path.join(
    MODEL_DIR,
    "deploy.prototxt"
)

CAFFE_MODEL = os.path.join(
    MODEL_DIR,
    "res10_300x300_ssd_iter_140000.caffemodel"
)

CONFIDENCE_THRESHOLD = 0.60

MIN_FACE_SIZE = 40


# ==========================================================
# CHECKPOINTS
# ==========================================================

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

LAST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "last_model.pth"
)


# ==========================================================
# STATIC
# ==========================================================

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

UPLOAD_FOLDER = os.path.join(
    STATIC_DIR,
    "uploads"
)


# ==========================================================
# IMAGE SETTINGS
# ==========================================================

IMAGE_SIZE = 224

CHANNELS = 3

FRAME_SKIP = 15

IMAGE_FORMAT = ".jpg"

BLUR_THRESHOLD = 100


# ==========================================================
# DATA SPLIT
# ==========================================================

TRAIN_SPLIT = 0.80

VALIDATION_SPLIT = 0.20


# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 32

NUM_WORKERS = 0

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING = 5

RANDOM_SEED = 42


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# CLASS LABELS
# ==========================================================

CLASS_NAMES = [
    "Real",
    "Fake"
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Deepfake Image Detection"

MODEL_NAME = "EfficientNet-B0"

VERSION = "2.0"

AUTHOR = "Final Year Project"


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

DIRECTORIES = [

    DATASET_DIR,

    TRAIN_DIR,
    VAL_DIR,

    TRAIN_REAL_DIR,
    TRAIN_FAKE_DIR,

    VAL_REAL_DIR,
    VAL_FAKE_DIR,

    MODEL_DIR,

    CHECKPOINT_DIR,

    STATIC_DIR,

    UPLOAD_FOLDER
]

for directory in DIRECTORIES:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ==========================================================
# VERIFY REQUIRED FILES
# ==========================================================

if not os.path.exists(REAL_VIDEO_DIR):

    print(
        f"[WARNING] Missing real videos:\n"
        f"{REAL_VIDEO_DIR}"
    )

if not os.path.exists(FAKE_VIDEO_DIR):

    print(
        f"[WARNING] Missing fake videos:\n"
        f"{FAKE_VIDEO_DIR}"
    )

if not os.path.exists(PROTOTXT):

    print(
        f"[WARNING] Missing deploy.prototxt:\n"
        f"{PROTOTXT}"
    )

if not os.path.exists(CAFFE_MODEL):

    print(
        f"[WARNING] Missing Caffe model:\n"
        f"{CAFFE_MODEL}"
    )