import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import numpy as np
import logging
import json
import sys


def preprocess_image(feature_extractor, image_path: Path):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors='pt')
    return inputs


@torch.no_grad()
def get_embedding(model, feature_extractor, image_path: Path, device):
    inputs = preprocess_image(feature_extractor, image_path)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().cpu().numpy()


def run(images_dir: Path, out_path: Path, model_name: str):
    logger.info("Starting image embedding: images=%s out=%s model=%s", images_dir, out_path, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device)
    model.eval()

    sample_ids = []
    embeddings = []
    for p in tqdm(sorted(images_dir.iterdir()), desc='Image embedding'):
        if not p.is_file():
            continue
        try:
            emb = get_embedding(model, feature_extractor, p, device)
            sample_id = p.stem
            sample_ids.append(str(sample_id))
            embeddings.append(emb)
        except UnidentifiedImageError:
            logger.warning("Skipping corrupted image %s", p)
        except Exception as e:
            logger.warning("Skipping %s due to %s", p, e)

    out_df = pd.DataFrame({'sample_id': sample_ids, 'embedding': embeddings})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(out_path)
    logger.info("Wrote image embeddings to %s (n=%d)", out_path, len(out_df))

    metrics = {'n_samples': int(len(out_df))}
    with open(metrics_dir / 'image_embeddings.json', 'w') as f:
        json.dump(metrics, f)
    logger.info("Wrote metrics to %s", (metrics_dir / 'image_embeddings.json'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images', '-i', type=Path, required=True)
    p.add_argument('--out', '-o', type=Path, required=True)
    p.add_argument('--model', default='google/vit-base-patch16-224-in21k')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # ensure logs/metrics dirs and logger
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_dir / 'image_embed.log')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    try:
        run(args.images, args.out, args.model)
    except Exception:
        logger.exception("Image embedding failed")
        sys.exit(1)
