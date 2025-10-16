import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ST_GAT(torch.nn.Module):
    """Spatio-Temporal Graph Attention Network."""

    def __init__(self, in_channels, out_channels, n_nodes, heads=8, dropout=0.2):
        super().__init__()
        self.n_pred = out_channels
        self.n_nodes = n_nodes
        self.dropout = dropout

        # Graph Attention
        self.gat = GATConv(in_channels, in_channels, heads=heads, concat=False)

        # LSTMs
        self.lstm1 = torch.nn.LSTM(input_size=n_nodes, hidden_size=32, num_layers=1)
        self.lstm2 = torch.nn.LSTM(input_size=32, hidden_size=128, num_layers=1)

        # Final Linear layer
        self.linear = torch.nn.Linear(128, n_nodes * out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph Attention
        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        # Reshape for LSTMs
        batch_size = data.num_graphs
        n_node = int(data.num_nodes / batch_size)
        x = torch.reshape(x, (batch_size, n_node, data.num_features))  # [B, N, F]
        x = torch.movedim(x, 2, 0)  # [F, B, N]

        # Two LSTMs
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Take last timestep
        x = x[-1, :, :]  # [B, 128]

        # FC Layer
        x = self.linear(x)  # [B, N*P]
        x = torch.reshape(x, (x.shape[0], self.n_nodes, self.n_pred))
        x = torch.reshape(x, (x.shape[0] * self.n_nodes, self.n_pred))

        return x
