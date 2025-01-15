import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
       
        super(PositionalEncoding, self).__init__()
        
        # Precompute positional encodings
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', self.pe)  # Ensure it's not updated during training

    def forward(self, x, relative_week_indices=None):
        """
        Add positional encoding to input embeddings.
        If `relative_week_indices` is provided, it uses these as positions.
        """
        if relative_week_indices is not None:
            relative_pe = self.pe[:, relative_week_indices]
            return x + relative_pe
        return x + self.pe[:, :x.size(1)]

class InputEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        """
        Linear projection for transforming inputs to match transformer dimensions.
        input_dim: Number of input features per timestep.
        d_model: Embedding dimension for transformer.
        """
        super(InputEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        """
        Project input features into d_model-dimensional space.
        """
        return self.linear(x)

class TransformerInputPreparation(nn.Module):
    def __init__(self, input_dim, d_model, max_len=5000):
        """
        Combines input embedding and positional encoding for transformer preparation.
        input_dim: Number of input features.
        d_model: Embedding dimension for transformer.
        max_len: Maximum length of positional encodings.
        """
        super(TransformerInputPreparation, self).__init__()
        self.input_embedding = InputEmbedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x, relative_week_indices=None):
        """
        Prepare inputs for the transformer.
        x: Input tensor of shape (batch_size, seq_len, input_dim).
        relative_week_indices: Relative week indices for custom positional encodings.
        """
        x_emb = self.input_embedding(x)
        x_with_pe = self.positional_encoding(x_emb, relative_week_indices)
        return x_with_pe


# Example Usage
if __name__ == "__main__":
    # Define parameters
    batch_size = 2
    seq_len = 4
    input_dim = 50  # Number of features
    d_model = 64  # Transformer embedding size

    # Example input: batch of (sequence length x input features)
    x = torch.rand(batch_size, seq_len, input_dim)

    # Example relative week indices
    relative_week_indices = torch.tensor([0, 1, 2, 0])

    # Initialize model
    transformer_input = TransformerInputPreparation(input_dim, d_model)

    # Forward pass
    output = transformer_input(x, relative_week_indices)
    print("Output Shape:", output.shape)  # Should be (batch_size, seq_len, d_model)
