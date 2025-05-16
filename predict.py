import torch
from utils import token2idx, idx2label, load_model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None  # Lazy-loaded model

def get_model():
    global _model
    if _model is None:
        _model = load_model("model/model_2.2.2_bigru.pth", "embeddings/bio_embedding_matrix.npy")
        _model.to(device)
        _model.eval()
    return _model

def predict_tags(text):
    model = get_model()
    tokens = text.strip().split()
    token_ids = [token2idx.get(tok.lower(), 0) for tok in tokens]
    x = torch.tensor(token_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=-1)
        tags = [idx2label[i.item()] for i in preds[0]]
    return list(zip(tokens, tags))