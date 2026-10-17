from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "splits"


# Reproducibility
RANDOM_SEED = 42

# Data splits
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Problem definition
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



