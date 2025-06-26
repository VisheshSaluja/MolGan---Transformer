import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
import os
from utils import load_smiles_from_sdf, build_tokenizer, smiles_to_tensor, tensor_to_smiles
from model.transformer import MoleculeTransformer
from rdkit import Chem

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SMILES
smiles_list = load_smiles_from_sdf("data/gdb9.sdf")
char_to_idx, idx_to_char = build_tokenizer(smiles_list)
vocab_size = len(char_to_idx)

# Dataset
class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.data = smiles_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        tensor = smiles_to_tensor(smiles, char_to_idx, MAX_LEN)
        return torch.tensor(tensor, dtype=torch.long)

dataset = SmilesDataset(smiles_list)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = MoleculeTransformer(vocab_size=vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# Training
print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(dataloader):
        batch = batch.to(DEVICE)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs.transpose(0,1))
        logits = logits.transpose(0,1).reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), "outputs/transformer_qm9.pth")



## Evaluation and Generation





# Generation
def generate(model, start_token='<', max_len=MAX_LEN):
    model.eval()
    idx = char_to_idx.get(start_token, random.choice(list(char_to_idx.values())))
    generated = [idx]
    for _ in range(max_len):
        inp = torch.tensor(generated, dtype=torch.long).unsqueeze(1).to(DEVICE)
        logits = model(inp)
        next_idx = torch.argmax(logits[-1, 0]).item()
        if next_idx == 0:
            break
        generated.append(next_idx)
    return tensor_to_smiles(generated, idx_to_char)

print("\nGenerating molecules...")
generated_smiles = [generate(model) for _ in range(1000)]

# Evaluation
valid_smiles = []
unique_smiles = set()

for s in generated_smiles:
    mol = Chem.MolFromSmiles(s)
    if mol:
        valid_smiles.append(s)
        unique_smiles.add(s)

validity = len(valid_smiles) / len(generated_smiles) * 100
uniqueness = len(unique_smiles) / len(valid_smiles) * 100
novelty = len(set(unique_smiles) - set(smiles_list)) / len(unique_smiles) * 100

print(f"Validity: {validity:.2f}%")
print(f"Uniqueness: {uniqueness:.2f}%")
print(f"Novelty: {novelty:.2f}%")

with open("outputs/generated_smiles.txt", "w") as f:
    for s in valid_smiles:
        f.write(s + "\n")
