import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for ROPE."
        self.d_model = d_model
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
        sinusoid_inp = positions * self.inv_freq
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)

        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)

        x_reshaped = x.view(x.size(0), x.size(1), -1, 2)
        real, imag = x_reshaped.unbind(-1)

        rotated_real = real * cos - imag * sin
        rotated_imag = real * sin + imag * cos

        x_rotated = torch.stack((rotated_real, rotated_imag), dim=-1).view_as(x)
        return x_rotated

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEncoding(d_model)

    def split_heads(self, x):
        """
        This helper function splits the input tensor into multiple attention heads for parallel processing.
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        q = self.rope(q)
        k = self.rope(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(x.size(0), x.size(1), self.d_model)

        return self.fc_out(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.encoder_blocks:
            x = block(x)
        return self.norm(x)

# Example Usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    input_dim = 50
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 4

    x = torch.rand(batch_size, seq_len, input_dim)

    encoder = TransformerEncoder(input_dim, d_model, num_heads, d_ff, num_layers)
    output = encoder(x)

    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)
