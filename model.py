
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200): # Hardcoded to 200 per user request
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Clip input if longer than max_len
        if x.size(0) > self.pe.size(0):
            x = x[:self.pe.size(0), :]
        return x + self.pe[:x.size(0), :]

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_dim=2653, d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1, vocab_size=10160):
        super(SignLanguageTransformer, self).__init__()
        
        # Enforce strict architecture
        input_dim = 2653
        vocab_size = 10160
        num_encoder_layers = 4
        num_decoder_layers = 4
        
        self.d_model = d_model
        
        # Encoder projection with LayerNorm for stability
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)
        
        # Decoder embedding
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=200)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Generator
        self.fc_out = nn.Linear(d_model, vocab_size)
        # Aliasing for backward compatibility with predictors
        self.generator = self.fc_out
        
    def encode(self, search_source):
        # Input: [batch, seq_len, features]
        print(f"DEBUG: encode input shape: {search_source.shape}")
        
        # Apply LayerNorm first for stability
        search_source = self.input_norm(search_source)
        print(f"DEBUG: After LayerNorm - Mean: {search_source.mean().item():.6f}, Std: {search_source.std().item():.6f}")
        
        # Project features
        src = self.input_proj(search_source) * math.sqrt(self.d_model)
        print(f"DEBUG: encode projected src shape: {src.shape}")
        
        # Permute for transformer: [seq_len, batch, features]
        # Check dim before permute
        if src.dim() != 3:
             print(f"CRITICAL ERROR: src dim is {src.dim()}, shape {src.shape}. Permute(1,0,2) will fail.")
        src = src.permute(1, 0, 2)
        
        src = self.pos_encoder(src)
        
        # nn.Transformer doesn't expose encoder directly easily unless we use its submodules
        memory = self.transformer.encoder(src)
        
        # Validate memory before returning
        memory_batch_first = memory.permute(1, 0, 2)
        print(f"DEBUG: Memory stats - Mean: {memory_batch_first.mean().item():.6f}, Std: {memory_batch_first.std().item():.6f}, Min: {memory_batch_first.min().item():.6f}, Max: {memory_batch_first.max().item():.6f}")
        
        if memory_batch_first.std().item() < 0.01:
            print("WARNING: Memory has very low variance! Cross-attention may fail.")
        
        return memory_batch_first # Return batch-first

    def decode(self, tgt, memory, tgt_mask=None):
        # tgt: [batch, seq_len]
        # memory: [batch, src_len, d_model]
        print(f"DEBUG: decode tgt shape: {tgt.shape}, memory shape: {memory.shape}")
        
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = tgt.permute(1, 0, 2) # [seq_len, batch, d_model]
        tgt = self.pos_decoder(tgt) # Use pos_decoder for decoder positional encoding
        
        memory = memory.permute(1, 0, 2) # [src_len, batch, d_model]
        
        # nn.Transformer decoder takes tgt and memory
        output = self.transformer.decoder(tgt, memory)
        
        return output.permute(1, 0, 2) # [batch, seq_len, d_model]

    def forward(self, src, tgt):
        # Simplified forward
        print(f"DEBUG: forward src {src.shape}, tgt {tgt.shape}")
        
        # Apply LayerNorm
        src = self.input_norm(src)
        
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = tgt.permute(1, 0, 2)
        tgt = self.pos_decoder(tgt)  # Use pos_decoder for target
        
        output = self.transformer(src, tgt)
        return self.fc_out(output)

