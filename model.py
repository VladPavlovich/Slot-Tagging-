import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=256, hidden_dim=256, num_layers=2):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
    
        )
        #for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        #non bidirectional
        #self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return logits
