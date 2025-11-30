import os

# Define paths for MLflow artifacts
REFINER_MODEL_PATH = "embedding_refiner_checkpoint.pth"
LGBM_MODEL_PATH = "trained_lgbm_model.txt"


def test_refiner_model_exists():
    """
    Verify that the Refiner model checkpoint downloaded from MLflow exists.
    """
    assert os.path.exists(REFINER_MODEL_PATH), \
        f"Refiner model not found at {REFINER_MODEL_PATH}. Run download_models.py first."


def test_lgbm_model_exists():
    """
    Verify that the LightGBM model file downloaded from MLflow exists.
    """
    assert os.path.exists(LGBM_MODEL_PATH), \
        f"LGBM model not found at {LGBM_MODEL_PATH}. Run download_models.py first."
