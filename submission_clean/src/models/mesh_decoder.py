import torch
import torch.nn as nn

class MeshDecoder(nn.Module):
    def __init__(self, in_dim, num_vertices=642):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 1024), nn.ReLU(), nn.Linear(1024, num_vertices*3))
        self.num_vertices = num_vertices

    def forward(self, x):
        y = self.fc(x)
        y = y.view(x.size(0), self.num_vertices, 3)
        return y