model = TransformerEncoder(
        input_dim=len(input_cols),
        d_model=256,
        num_heads=4,
        d_ff=256,
        num_layers=8
    ).to(device)