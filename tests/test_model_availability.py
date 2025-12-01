import os

# ------------------- Paths for local model files -------------------
REFINER_MODEL_PATH = "embedding_refiner_checkpoint.pth"
LGBM_MODEL_PATH = "trained_lgbm_model.lgb"  # <-- updated to correct .lgb file

def test_refiner_model_exists():
    """
    Verify that the Refiner model checkpoint downloaded from MLflow exists.
    """
    assert os.path.exists(REFINER_MODEL_PATH), \
        f"Refiner model not found at {REFINER_MODEL_PATH}. Run download_models.py first."
    print(f"✅ Found Refiner model at {REFINER_MODEL_PATH}")

def test_lgbm_model_exists():
    """
    Verify that the LightGBM model file downloaded from MLflow exists.
    """
    assert os.path.exists(LGBM_MODEL_PATH), \
        f"LGBM model not found at {LGBM_MODEL_PATH}. Run download_models.py first."
    print(f"✅ Found LGBM model at {LGBM_MODEL_PATH}")

if __name__ == "__main__":
    test_refiner_model_exists()
    test_lgbm_model_exists()
