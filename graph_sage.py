import pandas as pd
import numpy as np


# Sample data in pandas DataFrame format
data = {
    'player_id': [1, 2, 3, 4],
    'team_id': [10, 10, 20, 20],
    'points_avg': [22.5, 15.3, 18.7, 11.2],
    'assists_avg': [6.1, 2.3, 5.4, 1.8],
    'rebounds_avg': [7.8, 4.1, 6.2, 3.3],
    'next_game_points': [25, 14, 21, 12],  # this is the target
}
df = pd.DataFrame(data)

import torch
from torch_geometric.data import Data
import networkx as nx


features = torch.tensor(df[['points_avg', 'assists_avg', 'rebounds_avg']].values, dtype=torch.float)


edges = []
for i in range(len(df)):
    for j in range(len(df)):
        if i != j and df.loc[i, 'team_id'] == df.loc[j, 'team_id']:
            edges.append([i, j])


edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


y = torch.tensor(df['next_game_points'].values, dtype=torch.float)


data = Data(x=features, edge_index=edge_index, y=y)

import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()  

model = GraphSAGE(in_channels=3, hidden_channels=16, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch} - Loss: {loss.item():.4f}')

model.eval()
predictions = model(data).detach().numpy()
print("Predicted next game points:", predictions)

import matplotlib.pyplot as plt
import numpy as np

# Actual values (from your dataset)
actual_points = np.array([25, 14, 21, 12])  # Adjust if your actual values are different

# Predicted values (from your model)
predicted_points = np.array([25.236351, 14.669236, 20.680695, 11.183954])

# Plot
plt.figure(figsize=(8, 5))
x = np.arange(len(actual_points))
bar_width = 0.35

plt.bar(x - bar_width/2, actual_points, width=bar_width, label='Actual', color='skyblue')
plt.bar(x + bar_width/2, predicted_points, width=bar_width, label='Predicted', color='salmon')

# Labels & formatting
plt.xlabel('Player Index')
plt.ylabel('Next Game Points')
plt.title('Actual vs Predicted Player Points')
plt.xticks(x, [f'Player {i+1}' for i in x])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
