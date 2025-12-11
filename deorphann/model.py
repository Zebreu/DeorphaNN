import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, Linear

class DeorphaNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels=128,
                 gatheads=10, gatdropout=0.5, finaldropout=0.5):
        super().__init__()
        self.finaldropout = finaldropout
        self.norm = BatchNorm(input_channels)
        self.conv1 = GATv2Conv(
            input_channels, hidden_channels,
            dropout=gatdropout, heads=gatheads,
            concat=False, edge_dim=128
        )
        self.pooling = global_mean_pool
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, edge_attr, batch, hidden=False):
        x = self.norm(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()

        if hidden:
            return x

        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.finaldropout, training=self.training)
        return self.lin(x)
