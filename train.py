import torch
from torch import nn

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ignore padding index (0) in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_size = X_train.size(0)
    val_size = X_val.size(0)
    num_train_batches = (train_size + batch_size - 1) // batch_size
    num_val_batches = (val_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for i in range(num_train_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_size)

            batch_X = X_train[start_idx:end_idx].to(device)
            batch_Y = Y_train[start_idx:end_idx].to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
            
            outputs = outputs.view(-1, outputs.shape[-1])
            # batch_Y: (batch_size * seq_len)
            batch_Y = batch_Y.view(-1)

            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_train_batches

        # val phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for i in range(num_val_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, val_size)

                batch_X = X_val[start_idx:end_idx].to(device)
                batch_Y = Y_val[start_idx:end_idx].to(device)

                outputs = model(batch_X)
                
                outputs = outputs.view(-1, outputs.shape[-1])
                batch_Y = batch_Y.view(-1)

                loss = criterion(outputs, batch_Y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / num_val_batches

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")