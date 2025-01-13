import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_layers, output_size, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(5000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, output_size)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        x = self.input_projection(x)

        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0).to(x.device)

        x = self.transformer_encoder(x)

        x = x[:, -1, :]  # Shape: [batch_size, d_model]

        output = self.output_layer(x)
        return output


    def _generate_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
