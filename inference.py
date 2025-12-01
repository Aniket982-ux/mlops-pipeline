import torch
import lightgbm as lgb
import mlflow.pytorch
import mlflow.lightgbm
import numpy as np

# Define ANN + decoder classes
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

# ------------------- Device setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- MLflow model URIs -------------------
MLFLOW_TRACKING_URI = "http://136.111.62.53:5000"
EMBEDDING_MODEL_NAME = "RefinerModel"
LGBM_MODEL_NAME = "LGBMModel"
EMBEDDING_MODEL_URI = f"{MLFLOW_TRACKING_URI}/models/{EMBEDDING_MODEL_NAME}/Production"
LGBM_MODEL_URI = f"{MLFLOW_TRACKING_URI}/models/{LGBM_MODEL_NAME}/Production"

# ------------------- Global model cache -------------------
_embedding_model = None
_lgbm_model = None

def load_embedding_model():
    """
    Load the PyTorch embedding model from MLflow Production stage.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = mlflow.pytorch.load_model(EMBEDDING_MODEL_URI, map_location=device)
        _embedding_model.to(device)
        _embedding_model.eval()
    return _embedding_model

def load_lgbm_model():
    """
    Load the LightGBM model from MLflow Production stage.
    """
    global _lgbm_model
    if _lgbm_model is None:
        _lgbm_model = mlflow.lightgbm.load_model(LGBM_MODEL_URI)
    return _lgbm_model

# ------------------- Prediction function -------------------
def predict_price_from_embedding(combined_embedding, embedding_model=None, lgbm_model=None):
    """
    Predict price from embedding.
    
    Args:
        combined_embedding: np.array or torch.Tensor of shape [embedding_dim]
        embedding_model: optional, PyTorch EmbeddingRefinerWithRegressor
        lgbm_model: optional, LightGBM Booster
    
    Returns:
        price_pred: float
    """
    if embedding_model is None:
        embedding_model = load_embedding_model()
    if lgbm_model is None:
        lgbm_model = load_lgbm_model()

    # Convert to torch tensor if needed
    if not isinstance(combined_embedding, torch.Tensor):
        emb_tensor = torch.tensor(combined_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        emb_tensor = combined_embedding.float().unsqueeze(0).to(device)

    # Get refined embedding
    with torch.no_grad():
        _, refined_emb = embedding_model(emb_tensor)

    refined_emb_np = refined_emb.cpu().numpy()
    price_pred = lgbm_model.predict(refined_emb_np)[0]  # single example
    return price_pred
