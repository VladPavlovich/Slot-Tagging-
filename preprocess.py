import csv
from collections import Counter
import torch


def load_data(path):
    sentences = []
    labels = []

    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:

            # skip header
            if row[0] == "ID":
                continue

            idx = row[0]
            words = row[1].split()
            tags = row[2].split()

            if len(words) != len(tags):
                print(" BAD ROW:", row)
                print("words:", words)
                print("tags:", tags)
                raise ValueError(f"Length mismatch at id={idx}")

            sentences.append(words)
            labels.append(tags)

    return sentences, labels





def build_vocab(sentences, min_freq=1):
    word_counter = Counter()

    for sent in sentences:
        for w in sent:
            word_counter[w] += 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in word_counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)

    return vocab


def build_tag_vocab(labels):
    tag2id = {"<PAD>": 0}
    for seq in labels:
        for tag in seq:
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)
    return tag2id

def encode(sentences, labels, word2id, tag2id):
    enc_words = []
    enc_tags = []

    for sent, lab in zip(sentences, labels):
        w_ids = [word2id.get(w, word2id["<UNK>"]) for w in sent]
        t_ids = [tag2id[t] for t in lab]
        enc_words.append(w_ids)
        enc_tags.append(t_ids)

    return enc_words, enc_tags



def pad_sequences(seqs, pad_value):
    max_len = max(len(s) for s in seqs)
    padded = []
    lengths = []

    for s in seqs:
        lengths.append(len(s))
        padded_seq = s + [pad_value] * (max_len - len(s))
        padded.append(padded_seq)

    return torch.tensor(padded), torch.tensor(lengths)



def prepare_dataset(path):
    sentences, labels = load_data(path)

    word2id = build_vocab(sentences)
    tag2id = build_tag_vocab(labels)

    X, Y = encode(sentences, labels, word2id, tag2id)

    X_pad, lengths = pad_sequences(X, pad_value=word2id["<PAD>"])
    Y_pad, _ = pad_sequences(Y, pad_value=tag2id["<PAD>"])

    return X_pad, Y_pad, lengths, word2id, tag2id

#for testing purposes so that we can use existing vocabularies
def prepare_dataset_with_vocab(path, word2id, tag2id):
    sentences, labels = load_data(path)
    X, Y = encode(sentences, labels, word2id, tag2id)
    X_pad, lengths = pad_sequences(X, pad_value=word2id["<PAD>"])
    Y_pad, _ = pad_sequences(Y, pad_value=tag2id["<PAD>"])
    return X_pad, Y_pad, lengths



if __name__ == "__main__":
    data_path = "train.csv"
    X_pad, Y_pad, lengths, word2id, tag2id = prepare_dataset(data_path)

    print("Padded Input shape:", X_pad.shape)
    print("Padded Labels shape:", Y_pad.shape)
    print("Lengths shape:", lengths.shape)



