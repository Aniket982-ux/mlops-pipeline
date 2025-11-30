import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import logging
import json
import sys


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_long_text(tokenizer, model, text, device, max_length=512, stride=50):
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    chunk_embeddings = []
    for i in range(len(encoded['input_ids'])):
        inputs = {k: v[i].unsqueeze(0).to(device) for k, v in encoded.items() if k != 'overflow_to_sample_mapping'}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = mean_pooling(outputs, inputs['attention_mask'])
        chunk_embeddings.append(emb.cpu())
    item_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    return item_embedding.squeeze().numpy()


def run(input_csv: Path, out_path: Path, model_name: str):
    logger.info("Starting text embedding: input=%s out=%s model=%s", input_csv, out_path, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(input_csv)
    # Expect columns: sample_id, catalog_content
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    sample_ids = []
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Text embedding"):
        sample_id = str(row['sample_id'])
        text = str(row.get('catalog_content', ''))
        emb = embed_long_text(tokenizer, model, text, device)
        sample_ids.append(sample_id)
        embeddings.append(emb)

    out_df = pd.DataFrame({'sample_id': sample_ids, 'embedding': embeddings})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(out_path)
    logger.info("Wrote text embeddings to %s (n=%d)", out_path, len(out_df))

    # write a small metrics file for dvc
    metrics = {'n_samples': int(len(out_df))}
    with open(metrics_dir / 'text_embeddings.json', 'w') as f:
        json.dump(metrics, f)
    logger.info("Wrote metrics to %s", (metrics_dir / 'text_embeddings.json'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', type=Path, required=True)
    p.add_argument('--out', '-o', type=Path, required=True)
    p.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # logging + metrics dir ensured here
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    # configure logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_dir / 'text_embed.log')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    try:
        run(args.input, args.out, args.model)
    except Exception:
        logger.exception("Text embedding failed")
        sys.exit(1)
