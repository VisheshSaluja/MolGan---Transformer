# import torch
# import torch.nn as nn

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)].to(x.device)
#         return x


# class TransformerDecoder(nn.Module):
#     def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=128):
#         super().__init__()
#         self.token_embed = nn.Embedding(vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, max_len)
#         decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
#         self.output_layer = nn.Linear(d_model, vocab_size)

#     def forward(self, tgt):
#         tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
#         embedded = self.token_embed(tgt)
#         embedded = self.pos_encoder(embedded)
#         memory = torch.zeros_like(embedded)  # dummy memory for decoder-only model
#         output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask)
#         return self.output_layer(output)


import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final output projection
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None):
        # Embed and encode the source (conditioning input)
        src_emb = self.token_embed(src)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # Embed and decode the target (SMILES)
        tgt_emb = self.token_embed(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Default autoregressive mask if not provided
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_layer(output)
