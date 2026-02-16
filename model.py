
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
        return x + self.pe[:, :x.size(1), :]

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim=2653, d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1, vocab_size=10160):
        super(SignLanguageTransformer, self).__init__()
        
        # Enforce strict architecture (Lean v18)
        input_dim = 2653
        
        self.d_model = d_model
        
        # Encoder projection (Renamed from input_proj to src_proj per v18 checkpoint)
        self.src_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=300) # Increased max_len to be safe
        
        # Decoder embedding
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=300)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # User Request: Fix UserWarning
        )
        
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
        memory = self.transformer.encoder(src)
        
        # Validate memory
        # memory is [batch, seq_len, d_model]
        variance = torch.var(memory).item()
        print(f"DEBUG: Encoder Variance: {variance:.4f}")
        
        if variance > 0.5:
            print("[STATUS: ALIVE] Encoder Variance is healthy.")
        else:
             print(f"WARNING: Low Encoder Variance! ({variance:.4f})")
        
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
        
        # nn.Transformer decoder takes tgt and memory
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        return output # [batch, seq_len, d_model]

    def forward(self, src, tgt):
        # Simplified forward
        
        # Project src
        src = self.src_proj(src) * math.sqrt(self.d_model)
        
        # Embed tgt
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        
        output = self.transformer(src, tgt)
        return self.fc_out(output)
