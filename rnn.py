import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, num_layers=1):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False

        )
        #self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    
    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, _ = self.rnn(embeds)
        logits = self.fc(rnn_out)
        return logits
        