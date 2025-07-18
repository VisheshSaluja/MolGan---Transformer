import re

def tokenize_smiles(smiles):
    """
    Tokenize SMILES using a chemically-aware regex pattern.
    Splits multi-character atoms (like Cl, Br, [nH]) correctly.
    """
    pattern = "(\[[^\[\]]{1,6}\])"  # captures [CH3], [nH], etc.
    regex = re.compile(pattern)
    tokens = []
    for token in regex.split(smiles):
        if token.startswith('['):  # chemical token
            tokens.append(token)
        else:  # split further into characters
            tokens.extend(list(token))
    return tokens

def build_vocab(smiles_list):
    """
    Build a unique sorted vocabulary from list of tokenized SMILES.
    Includes special tokens: <pad>, <bos>, <eos>, <unk>
    """
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenize_smiles(smi))
    vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + sorted(tokens)
    return vocab
