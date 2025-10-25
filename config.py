# Configuration file for Bean Lesion Classification

# Hyperparameters
LR = 1e-3
EPOCHS = 15
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)

# Data paths
TRAIN_CSV = 'archive/train.csv'
VAL_CSV = 'archive/val.csv'
DATA_DIR = 'archive/'

# Model parameters
NUM_CLASSES = 3  # angular_leaf_spot, bean_rust, healthy
MODEL_NAME = 'googlenet'

# Device configuration
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Normalization parameters (ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Class names
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']
