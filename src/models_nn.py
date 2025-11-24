import torch
import torch.nn as nn

class MedicationEmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 10, hidden_dim), # 10 = placeholder for demographics
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, med_ids, demo_features):
        med_vec = self.embed(med_ids).mean(dim=1)
        x = torch.cat([med_vec, demo_features], dim=1)
        return self.fc(x)
