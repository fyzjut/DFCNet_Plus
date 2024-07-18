import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=2, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.pos_encoder = PositionalEncoding(input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, src_feats, src_lens):
        src_feats = self.pos_encoder(src_feats)
        src_feats = src_feats.permute(1, 0, 2)  # (max_src_len, batch_size, input_size)
        outputs = self.transformer_encoder(src_feats)
        outputs = outputs.permute(1, 0, 2)  # (batch_size, max_src_len, input_size)
        return {
            "predictions": outputs,
            "hidden": None
        }