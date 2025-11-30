import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import json
import sys
import mlflow
import mlflow.pytorch

# logging + metrics setup
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
    fh = logging.FileHandler(log_dir / 'refiner.log')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
logger.setLevel(logging.INFO)


class EmbeddingPriceDataset(Dataset):
    def __init__(self, df, embedding_col='combined_embedding', price_col='price'):
        self.sample_ids = df['sample_id'].values
        self.embeddings = df[embedding_col].values
        self.prices = df[price_col].values

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx):
        emb = torch.tensor(np.array(self.embeddings[idx]), dtype=torch.float32)
        price = torch.tensor(self.prices[idx], dtype=torch.float32)
        sample_id = self.sample_ids[idx]
        return emb, price, sample_id


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, hidden_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        query = self.norm1(query + attn_output)
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        return output


class EmbeddingRefinerWithRegressor(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.ffn = FFN(input_dim, hidden_dim, dropout_rate)
        self.decoder = Decoder(input_dim, num_heads, hidden_dim, dropout_rate)
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.ffn(x)
        x_seq = x.unsqueeze(1)
        x_dec = self.decoder(x_seq, x_seq)
        x_out = x_dec.squeeze(1)
        price_pred = self.regressor(x_out)
        return price_pred.squeeze(1), x_out


def run(input_path: Path, out_model: Path, out_emb: Path, epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):
    logger.info("Starting refiner training: in=%s out_model=%s out_emb=%s epochs=%d batch_size=%d lr=%g", input_path, out_model, out_emb, epochs, batch_size, lr)
    
    mlflow.set_tracking_uri("http://136.111.62.53:5000")
    mlflow.set_experiment("RefinerTraining")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "input_path": str(input_path)
        })

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df = pd.read_pickle(input_path)
        dataset = EmbeddingPriceDataset(df)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim = len(np.array(df['combined_embedding'].iloc[0]))
        model = EmbeddingRefinerWithRegressor(input_dim=input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss_history = []
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for emb_batch, price_batch, _ in loader:
                emb_batch = emb_batch.to(device)
                price_batch = price_batch.to(device)
                optimizer.zero_grad()
                outputs, _ = model(emb_batch)
                loss = criterion(outputs, price_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * emb_batch.size(0)
            total_loss /= len(loader.dataset)
            loss_history.append(float(total_loss))
            logger.info("Epoch %d/%d, Loss: %.4f", epoch+1, epochs, total_loss)
            mlflow.log_metric("loss", total_loss, step=epoch)

        out_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, out_model)
        logger.info("Saved refiner checkpoint to %s", out_model)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model", registered_model_name="RefinerModel")

        # write metrics
        metrics = {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}
        with open(metrics_dir / 'refiner_train.json', 'w') as f:
            json.dump(metrics, f)
        logger.info("Wrote metrics to %s", (metrics_dir / 'refiner_train.json'))

        # Generate refined embeddings
        model.eval()
        all_refined = []
        sample_ids = []
        prices = []
        with torch.no_grad():
            for emb_batch, price_batch, sample_batch in DataLoader(dataset, batch_size=256, shuffle=False):
                emb_batch = emb_batch.to(device)
                _, refined = model(emb_batch)
                all_refined.append(refined.cpu().numpy())
                prices.extend(price_batch.numpy())
                sample_ids.extend(sample_batch)

        refined_np = np.vstack(all_refined) if all_refined else np.zeros((0, input_dim))
        out_df = pd.DataFrame({'sample_id': sample_ids, 'refined_embedding': list(refined_np), 'price': prices})
        out_emb.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_pickle(out_emb)
        logger.info("Saved refined embeddings to %s (n=%d)", out_emb, len(out_df))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='input', required=True, type=Path)
    p.add_argument('--out-model', required=True, type=Path)
    p.add_argument('--out-emb', required=True, type=Path)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
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
        fh = logging.FileHandler(log_dir / 'refiner.log')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    try:
        run(args.input, args.out_model, args.out_emb, args.epochs, args.batch_size, args.lr)
    except Exception:
        logger.exception("Refiner training failed")
        sys.exit(1)
