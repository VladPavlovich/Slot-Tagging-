import torch
from torch import nn


def train_model(model, X, Y, lengths, epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset_size = X.size(0)
    num_batches = (dataset_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, dataset_size)

            batch_X = X[start_idx:end_idx].to(device)
            batch_Y = Y[start_idx:end_idx].to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
            outputs = outputs.view(-1, outputs.shape[-1])
            batch_Y = batch_Y.view(-1)

            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")