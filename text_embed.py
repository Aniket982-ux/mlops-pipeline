# text_embed.py

import logging
from transformers import AutoTokenizer, AutoModel
import torch

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'sentence-transformers/all-mpnet-base-v2'

logging.info(f"Loading tokenizer for model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True   # 🔥 prevents downloading in Cloud Run
)

logging.info(f"Loading pretrained model: {model_name}")
model = AutoModel.from_pretrained(
    model_name,
    local_files_only=True   # 🔥 forces model to load from Docker layer only
)
logging.info("Model loaded successfully, moving to device...")

model = model.to(device)
logging.info(f"Model moved to device: {device}")

model.eval()  # Required for inference

max_length = 512
stride = 128

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_long_text(text: str) -> torch.Tensor:
    logging.info("Encoding input text for embedding...")
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )

    logging.info(f"Number of chunks generated: {len(encoded['input_ids'])}")

    chunk_embeddings = []
    for i in range(len(encoded['input_ids'])):
        inputs = {
            k: v[i].unsqueeze(0).to(device) 
            for k, v in encoded.items() 
            if k != 'overflow_to_sample_mapping'
        }
        with torch.no_grad():
            outputs = model(**inputs)
            emb = mean_pooling(outputs, inputs['attention_mask'])
        chunk_embeddings.append(emb.cpu())

    logging.info("Chunk embeddings computed, combining...")

    item_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    logging.info("Combined embedding ready")

    return item_embedding
