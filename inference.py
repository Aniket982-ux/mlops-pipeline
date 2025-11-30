import torch
import lightgbm as lgb
import numpy as np
import os

# Define ANN + decoder classes as before (FFN, Decoder, EmbeddingRefinerWithRegressor) or import

class FFN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.dense1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.dense2 = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, hidden_dim, dropout_rate)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        x = self.norm1(query + attn_output)
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)
        return output

class EmbeddingRefinerWithRegressor(torch.nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.ffn = FFN(input_dim, hidden_dim, dropout_rate)
        self.decoder = Decoder(input_dim, num_heads, hidden_dim, dropout_rate)
        self.regressor = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.ffn(x)
        x = x.unsqueeze(1)  # batch_size x seq_len=1 x embed_dim
        x = self.decoder(x, x)
        x = x.squeeze(1)
        price_pred = self.regressor(x)
        return price_pred.squeeze(1), x

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the refined embedding model
def load_embedding_model(path=None):
    if path is None:
        path = os.getenv("EMBEDDING_CHECKPOINT_PATH", "embedding_refiner_checkpoint.pth")
    model = EmbeddingRefinerWithRegressor()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Load LightGBM model
def load_lgbm_model(path=None):
    if path is None:
        path = os.getenv("LGBM_MODEL_PATH", "trained_lgbm_model.txt")
    return lgb.Booster(model_file=path)

# Prediction function: Input combined embedding -> refined embedding -> price prediction
def predict_price_from_embedding(combined_embedding, embedding_model, lgbm_model):
    # Convert to tensor
    emb_tensor = torch.tensor(combined_embedding, dtype=torch.float32).to(device)
    emb_tensor = emb_tensor.unsqueeze(0)  # batch size 1

    with torch.no_grad():
        _, refined_emb = embedding_model(emb_tensor)

    refined_emb_np = refined_emb.cpu().numpy()
    price_pred = lgbm_model.predict(refined_emb_np)[0]
    return price_pred


