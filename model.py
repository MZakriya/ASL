
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: [max_len, d_model]
        # Reshape for batch_first=True: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [batch, seq_len, d_model]
        # Add PE up to seq_len
        # self.pe is [1, max_len, d_model]
        # Slice columns (dim 1)
        return x + self.pe[:, :x.size(1), :].detach()

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim=1536, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, vocab_size=10160):
        super(SignLanguageTransformer, self).__init__()

        self.input_dim = input_dim  # Keep as instance variable for consistency
        self.d_model = d_model

        # Encoder projection (mapping input to d_model)
        self.src_proj = nn.Linear(input_dim, d_model)

        # Initialize positional encoding manually
        self.pos_encoder = PositionalEncoding(d_model, max_len=300)

        # Decoder embedding
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=300)

        # Transformer - Using more standard parameters that might match the original Kaggle model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Generator
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def encode(self, search_source):
        # Input: [batch, seq_len, features]
        # print(f"DEBUG: encode input shape: {search_source.shape}")

        # Project features (Using src_proj)
        src = self.src_proj(search_source) * math.sqrt(self.d_model)
        src = self.pos_encoder(src) # Apply PE (Restores Order)

        # No permute needed because batch_first=True
        # src is [batch, seq_len, d_model]

        # Encoder
        memory = self.transformer_encoder(src)

        # Validate memory
        # memory is [batch, seq_len, d_model]
        variance = torch.var(memory).item()
        # print(f"DEBUG: Encoder Variance: {variance:.4f}")

        if variance > 0.5:
            # print("[STATUS: ALIVE] Encoder Variance is healthy.")
            pass
        else:
             # print(f"WARNING: Low Encoder Variance! ({variance:.4f})")
             pass

        return memory

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def decode(self, tgt, memory, tgt_mask=None):
        # tgt: [batch, seq_len]
        # memory: [batch, src_len, d_model]

        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt) # Apply PE (Restores Order)
        # No permute: [batch, seq_len, d_model]

        # Transformer decoder takes tgt and memory
        # Ensure tgt_mask is properly shaped if provided
        if tgt_mask is not None and tgt_mask.size(0) != tgt.size(1):
            # Regenerate mask if it doesn't match the target sequence length
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

        return output # [batch, seq_len, d_model]

    def forward(self, src, tgt):
        # Simplified forward
        src = self.src_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)

        # Encoder
        memory = self.transformer_encoder(src)

        # Decoder
        output = self.transformer_decoder(tgt, memory)

        return self.fc_out(output)
