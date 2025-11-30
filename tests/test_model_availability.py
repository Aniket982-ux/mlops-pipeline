import os
import pytest
from transformers import AutoTokenizer, AutoModel, ViTFeatureExtractor, ViTModel

# Define paths for MLflow artifacts
REFINER_MODEL_PATH = "embedding_refiner_checkpoint.pth"
LGBM_MODEL_PATH = "trained_lgbm_model.txt"

# Define model names for embeddings
TEXT_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
IMAGE_MODEL_NAME = 'google/vit-base-patch16-224-in21k'

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

def test_text_embedding_model_available():
    """
    Verify that the text embedding model is available locally.
    """
    try:
        AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, local_files_only=True)
        AutoModel.from_pretrained(TEXT_MODEL_NAME, local_files_only=True)
    except OSError:
        pytest.fail(f"Text model {TEXT_MODEL_NAME} not found locally. Ensure it is cached.")

def test_image_embedding_model_available():
    """
    Verify that the image embedding model is available locally.
    """
    try:
        ViTFeatureExtractor.from_pretrained(IMAGE_MODEL_NAME, local_files_only=True)
        ViTModel.from_pretrained(IMAGE_MODEL_NAME, local_files_only=True)
    except OSError:
        pytest.fail(f"Image model {IMAGE_MODEL_NAME} not found locally. Ensure it is cached.")
