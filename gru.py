import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=256, hidden_dim=128, num_layers=1):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        gru_out, _ = self.gru(embeds)
        logits = self.fc(gru_out)
        return logits