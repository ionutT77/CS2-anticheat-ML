"""Numerical encounter encoders and player-match aggregation for CS2CD."""
import math

import torch
from torch import nn


def positional(length, width, device):
    p = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000.) / width))
    out = torch.zeros(length, width, device=device)
    out[:, 0::2] = torch.sin(p * f)
    out[:, 1::2] = torch.cos(p * f)
    return out


class ResidualConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation),
                                 nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(.15),
                                 nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                                 nn.GroupNorm(8, width))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class EncounterEncoder(nn.Module):
    def __init__(self, channels, architecture):
        super().__init__()
        self.architecture = architecture
        if architecture == 'lstm':
            self.first = nn.LSTM(channels, 128, batch_first=True)
            self.drop = nn.Dropout(.4)
            self.second = nn.LSTM(128, 64, batch_first=True)
        elif architecture == 'tcn':
            self.net = nn.Sequential(nn.Conv1d(channels, 64, 7, padding=3), nn.GroupNorm(8,64),
                                     nn.GELU(), ResidualConv(64, 1), ResidualConv(64, 2),
                                     ResidualConv(64, 4), nn.AvgPool1d(2), ResidualConv(64, 4))
            self.out = nn.Sequential(nn.Linear(128,64), nn.GELU(), nn.Dropout(.2))
        elif architecture == 'transformer':
            self.input = nn.Linear(channels * 8, 96)
            layer = nn.TransformerEncoderLayer(96, 4, 192, dropout=.2, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.net = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(192), nn.Linear(192,64), nn.GELU(), nn.Dropout(.2))
        else:
            raise ValueError(architecture)

    def forward(self, x):
        if self.architecture == 'lstm':
            x, _ = self.first(x)
            x, _ = self.second(self.drop(x))
            return x[:, -1]
        if self.architecture == 'tcn':
            x = self.net(x.transpose(1,2))
            return self.out(torch.cat([x.mean(-1), x.amax(-1)], dim=1))
        b, t, c = x.shape
        x = self.input(x.reshape(b, t//8, c*8))
        x = self.net(x + .1 * positional(x.shape[1], x.shape[2], x.device))
        return self.out(torch.cat([x.mean(1), x.amax(1)], dim=1))


class MatchDetector(nn.Module):
    def __init__(self, channels=39, architecture='tcn', hierarchical=False):
        super().__init__()
        self.channels = channels
        self.architecture = architecture
        self.hierarchical = hierarchical
        self.encoder = EncounterEncoder(channels, architecture)
        self.event_head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(.4), nn.Linear(32,1))
        if hierarchical:
            self.project = nn.Linear(64,96)
            layer = nn.TransformerEncoderLayer(96,4,192,dropout=.2,activation='gelu',
                                               batch_first=True,norm_first=True)
            self.sequence = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.match_head = nn.Sequential(nn.LayerNorm(192),nn.Linear(192,32),nn.GELU(),nn.Dropout(.3),nn.Linear(32,1))

    def aggregate(self, z, mask):
        event_logits = self.event_head(z).squeeze(-1)
        if not self.hierarchical:
            return (event_logits * mask).sum(1) / mask.sum(1), event_logits
        x = self.project(z)
        x = x + .1 * positional(x.shape[1], x.shape[2], x.device)
        x = self.sequence(x, src_key_padding_mask=~mask)
        avg = (x * mask[...,None]).sum(1) / mask.sum(1,keepdim=True)
        maximum = x.masked_fill(~mask[...,None], -1e4).amax(1)
        return self.match_head(torch.cat([avg,maximum],dim=1)).squeeze(1), event_logits

    def forward(self, x, mask):
        b, n, t, c = x.shape
        z = self.encoder(x.reshape(b*n,t,c)).reshape(b,n,64)
        return self.aggregate(z,mask)
