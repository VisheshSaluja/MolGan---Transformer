import pandas as pd
import numpy as np
import pickle
from tokenizer import tokenize_smiles, build_vocab

def encode_smiles(smiles_list, vocab):
    """
    Encode tokenized SMILES into integer sequences using the vocabulary.
    Adds <bos> and <eos> tokens.
    """
    stoi = {s: i for i, s in enumerate(vocab)}
    encoded = []
    for smi in smiles_list:
        tokens = ['<bos>'] + tokenize_smiles(smi) + ['<eos>']
        token_ids = [stoi.get(tok, stoi['<unk>']) for tok in tokens]
        encoded.append(token_ids)
    return encoded

def pad_sequences(sequences, pad_token='<pad>', vocab=None):
    """
    Pad sequences to the maximum length using the pad token ID.
    """
    pad_id = vocab.index(pad_token)
    max_len = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded

if __name__ == "__main__":
    df = pd.read_csv("data/smiles.csv")
    smiles_list = df["smiles"].tolist()

    # Build vocab and encode
    vocab = build_vocab(smiles_list)
    encoded_sequences = encode_smiles(smiles_list, vocab)
    padded_sequences = pad_sequences(encoded_sequences, pad_token='<pad>', vocab=vocab)

    # Save tokenized dataset
    with open("data/tokenized.pkl", "wb") as f:
        pickle.dump(padded_sequences, f)

    # Save vocab
    with open("data/vocab.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")

    print(f"[INFO] Saved tokenized dataset of shape {padded_sequences.shape} and vocab size {len(vocab)}.")
