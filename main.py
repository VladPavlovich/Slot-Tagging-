from preprocess import prepare_dataset
from model import LSTM
from train import train_model
from eval import evaluate
import torch


# preprocess the training data
X_pad, Y_pad, lengths, word2id, tag2id = prepare_dataset("train.csv")

print("Padded Input Sequences:\n", X_pad)
print("Padded Target Sequences:\n", Y_pad)
print("Sequence Lengths:\n", lengths)


model = LSTM(vocab_size=len(word2id), tagset_size=len(tag2id))

#train the model
train_model(model, X_pad, Y_pad, lengths, epochs=10, batch_size=32, learning_rate=0.001)

# Save the trained model
torch.save(model.state_dict(), "lstm_model.pth")


#evaluate the model
f1 = evaluate(model, X_pad, Y_pad, tag2id)
print(f"F1 Score on training data: {f1:.4f}")








