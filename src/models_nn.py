import torch
import torch.nn as nn


class TabularNN(nn.Module):
    """
    2-layer MLP for tabular features.

    This matches the model used in 03_model_nn.ipynb:
    - Input: standardized tabular features (one row per admission)
    - Hidden layers: 128 -> 64 with ReLU and dropout
    - Output: single logit (use with BCEWithLogitsLoss)
    """

    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim)
        # returns logits: (batch_size,)
        return self.net(x).squeeze(1)