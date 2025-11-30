# image_embed.py

import logging
import os
from PIL import Image, UnidentifiedImageError
import torch
from transformers import ViTModel, ViTFeatureExtractor

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'google/vit-base-patch16-224-in21k'

logging.info(f"Loading feature extractor for model: {model_name}")
feature_extractor = ViTFeatureExtractor.from_pretrained(
    model_name,
    local_files_only=True   # 🔥 prevents Cloud Run downloading
)

logging.info(f"Loading pretrained ViT model: {model_name}")
model = ViTModel.from_pretrained(
    model_name,
    local_files_only=True   # 🔥 ensures weight files must exist in Docker image
)

logging.info("ViT model loaded successfully, moving to device...")
model = model.to(device)
logging.info(f"Model moved to device: {device}")
model.eval()

def preprocess_image(image_path):
    """
    Opens and preprocesses image for embedding extraction.
    Raises ValueError on unreadable or corrupted image.
    """
    try:
        logging.info(f"Opening image: {image_path}")
        image = Image.open(image_path).convert('RGB')
    except (IOError, UnidentifiedImageError) as e:
        logging.error(f"Error opening image: {str(e)}")
        raise ValueError(f"Error opening image: {str(e)}")

    inputs = feature_extractor(images=image, return_tensors="pt")
    logging.info("Image preprocessed for model input")
    return inputs

@torch.no_grad()
def get_embedding(image_path):
    """
    Extracts CLS token embedding from image.
    """
    inputs = preprocess_image(image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logging.info(f"Running inference on image: {image_path}")
    outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    logging.info("Image embedding extracted")

    return cls_embedding.squeeze().cpu().numpy()
