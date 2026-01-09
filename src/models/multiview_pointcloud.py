import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import LayerNorm

class ResNetEncoder(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        if name == "resnet50":
            m = models.resnet50(pretrained=pretrained)
            self.dims = [256, 512, 1024, 2048]
        elif name == "resnet18":
            m = models.resnet18(pretrained=pretrained)
            self.dims = [64, 128, 256, 512]
        else:
            m = models.resnet50(pretrained=pretrained)
            self.dims = [256, 512, 1024, 2048]
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1
        self.l2 = m.layer2
        self.l3 = m.layer3
        self.l4 = m.layer4
        self.out_dim = sum(self.dims)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        g1 = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f1, (1,1)), 1)
        g2 = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f2, (1,1)), 1)
        g3 = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f3, (1,1)), 1)
        g4 = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f4, (1,1)), 1)
        return torch.cat([g1, g2, g3, g4], dim=1)

class FusionModule(nn.Module):
    def __init__(self, d1, d2, hidden=1024, heads=4):
        super().__init__()
        self.proj1 = nn.Linear(d1, hidden)
        self.proj2 = nn.Linear(d2, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, batch_first=True)
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), LayerNorm(hidden))

    def forward(self, f1, f2):
        a = self.proj1(f1).unsqueeze(1)
        b = self.proj2(f2).unsqueeze(1)
        t = torch.cat([a, b], dim=1)
        h = self.attn(t)
        z = h.mean(dim=1)
        z = self.out(z)
        return z

class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = LayerNorm(d)
        self.fc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
    def forward(self, x):
        return x + self.fc(self.ln(x))

class PointCloudDecoder(nn.Module):
    def __init__(self, in_dim, num_points=2048, hidden=1024):
        super().__init__()
        self.num_points = num_points
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), LayerNorm(hidden))
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.out = nn.Linear(hidden, num_points*3)

    def forward(self, x):
        h = self.pre(x)
        h = self.res1(h)
        h = self.res2(h)
        y = self.out(h)
        y = y.view(x.size(0), self.num_points, 3)
        return y

class MultiViewPointCloudModel(nn.Module):
    def __init__(self, num_points=2048, encoder_name="resnet50"):
        super().__init__()
        self.enc_front = ResNetEncoder(encoder_name, pretrained=True)
        self.enc_side = ResNetEncoder(encoder_name, pretrained=True)
        d = self.enc_front.out_dim
        self.fusion = FusionModule(d, d, hidden=1024, heads=4)
        self.decoder = PointCloudDecoder(1024, num_points=num_points, hidden=1024)

    def forward(self, front, side):
        f = self.enc_front(front)
        s = self.enc_side(side)
        z = self.fusion(f, s)
        pts = self.decoder(z)
        return pts