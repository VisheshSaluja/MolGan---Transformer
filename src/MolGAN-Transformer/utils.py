from rdkit import Chem
from rdkit.Chem import rdmolfiles

def load_smiles_from_sdf(path):
    suppl = rdmolfiles.SDMolSupplier(path)
    smiles = [Chem.MolToSmiles(m) for m in suppl if m is not None]
    return smiles

def build_tokenizer(smiles_list):
    charset = set(''.join(smiles_list))
    char_to_idx = {ch: idx + 1 for idx, ch in enumerate(sorted(charset))}
    char_to_idx['<PAD>'] = 0
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def smiles_to_tensor(smiles, char_to_idx, max_len):
    tensor = [char_to_idx.get(ch, 0) for ch in smiles]
    tensor = tensor[:max_len] + [0] * (max_len - len(tensor))
    return tensor

def tensor_to_smiles(tensor, idx_to_char):
    chars = [idx_to_char[idx] for idx in tensor if idx != 0]
    return ''.join(chars)
