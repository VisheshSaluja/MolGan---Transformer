import torch
import torch.nn as nn

class MoleculeTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim, vocab_size)

    def forward(self, tgt, memory=None):
        emb = self.embed(tgt)
        output = self.transformer(emb, memory if memory is not None else emb)
        return self.fc_out(output)
