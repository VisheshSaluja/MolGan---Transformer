import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tokenizer import tokenize_smiles, build_vocab
from rdkit import Chem

def encode_smiles(smiles_list, vocab):
    """
    Tokenize and encode SMILES strings into sequences of integer token IDs.
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
    Pad sequences to the maximum length in the batch using pad_id.
    """
    pad_id = vocab.index(pad_token)
    max_len = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded

if __name__ == "__main__":
    # Load SMILES from CSV
    df = pd.read_csv("data/smiles.csv")
    smiles_list = df["smiles"].tolist()

    # 🔍 Filter: keep only valid SMILES
    smiles_list = [smi for smi in smiles_list if Chem.MolFromSmiles(smi)]

    print(f"[INFO] Total valid SMILES: {len(smiles_list)}")
    # Build vocabulary
    vocab = build_vocab(smiles_list)
    print(f"[INFO] Vocab size: {len(vocab)}")

    # Encode SMILES to token IDs
    encoded_sequences = encode_smiles(smiles_list, vocab)

    # Split into train/val sets (90/10 split)
    train_seq, val_seq = train_test_split(encoded_sequences, test_size=0.1, random_state=42)
    print(f"[INFO] Train: {len(train_seq)} | Val: {len(val_seq)}")

    # Pad both sets
    train_padded = pad_sequences(train_seq, pad_token='<pad>', vocab=vocab)
    val_padded = pad_sequences(val_seq, pad_token='<pad>', vocab=vocab)

    # Save datasets
    with open("data/train.pkl", "wb") as f:
        pickle.dump((train_padded, train_padded), f)  # (src, tgt)
    with open("data/val.pkl", "wb") as f:
        pickle.dump((val_padded, val_padded), f)

    # Save vocab
    with open("data/vocab.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")

    print(f"[INFO] Saved train: {train_padded.shape}, val: {val_padded.shape}")