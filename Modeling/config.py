import os

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Data directory
DATA_DIRECTION = r"D:\Ai courses\ML mostafa saad\projects\Credit Card Fraud Detection"
TRAIN_PATH = os.path.join(DATA_DIRECTION, "train.csv")
VAL_PATH = os.path.join(DATA_DIRECTION, "val.csv")
TEST_PATH = os.path.join(DATA_DIRECTION, "test.csv")

# Model path (optional fallback)
MODEL_PATH = os.path.join(os.getcwd(), "Saved_Models", "model.pkl")

# Preprocessing settings
SCALE = True
REMOVE_LOW_VARIANCE = True
LOW_VARIANCE_THRESHOLD = 0.01

# Resampling
RESAMPLING_TECHNIQUE = "smotetomek"  # Options: undersample, smote, smotetomek
