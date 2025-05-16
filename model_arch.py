import torch
import torch.nn as nn

#definiingg BiGRU model
class BiGRUclass(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(BiGRUclass, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix).float(),freeze=False,padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.bigru(x)
        x = self.dropout(x)
        return self.out(x)