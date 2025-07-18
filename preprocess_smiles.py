from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import os

def get_augmented_smiles(mol, n=3):
    """
    Generate n randomized (non-canonical) SMILES strings for a molecule.
    """
    smiles = set()
    for _ in range(n):
        try:
            smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if smi:
                smiles.add(smi)
        except:
            continue
    return list(smiles)

def extract_smiles_from_sdf(sdf_path, csv_path=None, output_path="smiles.csv", augment_n=3):
    suppl = Chem.SDMolSupplier(sdf_path)
    all_smiles = []
    all_props = []

    properties_df = pd.read_csv(csv_path) if csv_path else None

    for i, mol in enumerate(suppl):
        if mol is not None:
            aug_smiles = get_augmented_smiles(mol, n=augment_n)
            for smi in aug_smiles:
                all_smiles.append(smi)
                if properties_df is not None:
                    all_props.append(properties_df.iloc[i].values)

    df = pd.DataFrame({'smiles': all_smiles})

    if csv_path:
        props_df = pd.DataFrame(all_props, columns=properties_df.columns)
        df = pd.concat([df, props_df.reset_index(drop=True)], axis=1)

    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved {len(df)} augmented molecules to {output_path}")

if __name__ == "__main__":
    sdf_file = "data/gdb9.sdf"
    csv_file = "data/gdb9.sdf.csv"
    output_file = "data/smiles.csv"
    extract_smiles_from_sdf(sdf_file, csv_file, output_file, augment_n=3)
