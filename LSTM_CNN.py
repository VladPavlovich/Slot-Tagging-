import torch
import torch.nn as nn


class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, num_layers=1):
        super(LSTM_CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5

        )
        self.conv1 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.permute(0, 2, 1)  # Change shape to (batch_size, hidden_dim*2, seq_len) for Conv1d
        conv_out = self.conv1(lstm_out)
        conv_out = conv_out.permute(0, 2, 1)  # Change back to (batch_size, seq_len, hidden_dim)
        logits = self.fc(conv_out)
        return logits

