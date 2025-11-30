import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import sys

# logging/metrics
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
    fh = logging.FileHandler(log_dir / 'merge_embed.log')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
logger.setLevel(logging.INFO)


def run(text_path: Path, image_path: Path, csv_path: Path, out_path: Path):
    logger.info("Merging embeddings: text=%s image=%s csv=%s out=%s", text_path, image_path, csv_path, out_path)
    text_df = pd.read_pickle(text_path)
    image_df = pd.read_pickle(image_path)

    text_df['sample_id'] = text_df['sample_id'].astype(str)
    image_df['sample_id'] = image_df['sample_id'].astype(str)

    merged_df = pd.merge(text_df, image_df, on='sample_id', how='inner', suffixes=('_text', '_image'))

    def combine_embeddings(row):
        emb_text = np.array(row['embedding_text'])
        emb_image = np.array(row['embedding_image'])
        return np.concatenate((emb_text, emb_image))

    merged_df['combined_embedding'] = merged_df.apply(combine_embeddings, axis=1)

    original_df = pd.read_csv(csv_path)
    original_df['sample_id'] = original_df['sample_id'].astype(str)
    final_df = merged_df[['sample_id', 'combined_embedding']]
    final_df_with_price = pd.merge(final_df, original_df[['sample_id', 'price']], on='sample_id', how='inner')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df_with_price.to_pickle(out_path)
    logger.info("Wrote combined embeddings with price to %s (n=%d)", out_path, len(final_df_with_price))

    metrics = {'n_samples': int(len(final_df_with_price))}
    with open(metrics_dir / 'merge.json', 'w') as f:
        json.dump(metrics, f)
    logger.info("Wrote metrics to %s", (metrics_dir / 'merge.json'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--text', required=True, type=Path)
    p.add_argument('--image', required=True, type=Path)
    p.add_argument('--csv', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # ensure logs/metrics dirs and logger are set
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
        fh = logging.FileHandler(log_dir / 'merge_embed.log')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    try:
        run(args.text, args.image, args.csv, args.out)
    except Exception:
        logger.exception("Merge stage failed")
        sys.exit(1)
