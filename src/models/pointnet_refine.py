import torch
import torch.nn as nn

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, k*k)
        )
    def forward(self, x):
        b = x.size(0)
        y = self.net(x)
        y = torch.max(y, 2)[0]
        y = self.fc(y)
        iden = torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(b,1,1)
        y = y.view(b, x.size(1), x.size(1)) + iden
        return torch.bmm(y, x)

class PointNetRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_t = TNet(3)
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.out = nn.Sequential(
            nn.Linear(256, 3)
        )
    def forward(self, pts):
        x = pts.transpose(1, 2)
        x = self.input_t(x)
        f = self.feat(x)
        f = torch.max(f, 2)[0]
        g = self.fc(f)
        g = g.unsqueeze(2).repeat(1, 256, pts.size(1))
        g = g.transpose(1, 2)
        o = self.out(g)
        return pts + o