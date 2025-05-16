import numpy as np
import torch
from model_arch import BiGRUclass  # use the same architecture you used in training

# Load label mappings
token2idx = np.load("utils/token2idx.npy", allow_pickle=True).item()
idx2label = np.load("utils/idx2label.npy", allow_pickle=True).item()

def load_model(model_path, embedding_path):
    emb_matrix = np.load(embedding_path)
    model = BiGRUclass(embedding_matrix=emb_matrix, hidden_dim=64, output_dim=len(idx2label))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model

def get_label_color(label):
    return {
        "B-AC": "#d1e7dd",
        "I-AC": "#badbcc",
        "B-LF": "#f8d7da",
        "I-LF": "#f5c2c7",
        "O": "#dee2e6"
    }.get(label, "#ffffff")