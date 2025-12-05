from preprocess import prepare_dataset
#for testing with existing vocabularies
from preprocess import prepare_test_dataset
from generate_test_csv import generate_submission
from model import LSTM
from train import train_model
from gru import GRU

# preprocess the training data
X_pad, Y_pad, lengths, word2id, tag2id = prepare_dataset("train.csv")


print("Padded Input Sequences:\n", X_pad)
print("Padded Target Sequences:\n", Y_pad)
print("Sequence Lengths:\n", lengths)


#model = LSTM(vocab_size=len(word2id), tagset_size=len(tag2id))
model = GRU(vocab_size=len(word2id), tagset_size=len(tag2id))
print(model)

#train the model
train_model(model, X_pad, Y_pad, lengths, epochs=15, batch_size=32, learning_rate=0.001)

# saved model state but already kept in memory for generating predictions
#torch.save(model.state_dict(), "lstm_model.pt")

#generate predictions and save to CSV
generate_submission(model, word2id, tag2id, input_path="test.csv", output_path="test_pred.csv")
print("Predictions saved to test_pred.csv")














