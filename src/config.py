from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"
FITZ_CSV = RAW / "fitzpatrick17k" / "fitzpatrick17ku.csv"
FITZ_IMG_DIR = RAW / "fitzpatrick17k"
DDI_CSV = RAW / "ddi" / "ddi_metadata.csv"
DDI_IMG_DIR = RAW / RAW / "ddi"

# Reproducibility
RANDOM_SEED = 42

# Data splits
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Problem definition
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_NAME = "efficientnet_b0"

BATCH_SIZE     = 32
LR             = 1e-4
WEIGHT_DECAY   = 1e-2
MAX_EPOCHS     = 30
PATIENCE       = 4
NUM_WORKERS    = 4
RANDOM_SEED    = 42



