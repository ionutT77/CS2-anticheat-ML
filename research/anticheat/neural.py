"""Hierarchical sequence models: encode engagements, aggregate player evidence."""
import torch
from torch import nn
import torch.nn.functional as F


def sequence_channels(raw, enriched=True):
    """[B,E,T,5] -> [B,E,C,T]; fixed physics transforms, no test statistics."""
    v=raw[...,:2]
    e=raw[...,2:4]
    parts=[torch.asinh(v),torch.asinh(e/5),raw[...,4:5]]
    if enriched:
        acc=torch.diff(v,dim=2,prepend=v[:,:,:1])
        de=torch.diff(e,dim=2,prepend=e[:,:,:1])
        de=torch.stack(((de[...,0]+180)%360-180,de[...,1]),dim=-1)
        parts += [torch.asinh(acc),torch.asinh(de/5),torch.asinh(v.norm(dim=-1,keepdim=True)),
                  torch.asinh(e.norm(dim=-1,keepdim=True)/5),
                  torch.linspace(-1,1,raw.shape[2],device=raw.device)[None,None,:,None].expand(*raw.shape[:3],1)]
    return torch.cat(parts,dim=-1).clamp(-12,12).permute(0,1,3,2).contiguous()


class MultiScaleBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        branch=cout//4
        self.bottleneck=nn.Conv1d(cin,branch,1,bias=False)
        self.branches=nn.ModuleList([nn.Conv1d(branch,branch,k,stride=stride,padding=k//2,bias=False) for k in [3,7,15]])
        self.pool=nn.Sequential(nn.MaxPool1d(3,stride=stride,padding=1),nn.Conv1d(cin,branch,1,bias=False))
        self.norm=nn.BatchNorm1d(cout)
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv1d(cin,cout,1,stride=stride,bias=False),nn.BatchNorm1d(cout))
    def forward(self,x):
        b=self.bottleneck(x)
        h=torch.cat([c(b) for c in self.branches]+[self.pool(x)],dim=1)
        return F.gelu(self.norm(h)+self.skip(x))


class TemporalEncoder(nn.Module):
    def __init__(self,channels=12,width=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(channels,width,7,padding=3,bias=False),nn.BatchNorm1d(width),nn.GELU(),
                               MultiScaleBlock(width,width),MultiScaleBlock(width,width,stride=2),
                               MultiScaleBlock(width,width*2,stride=2))
        self.project=nn.Sequential(nn.Linear(width*2*5,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15))
        self.output_dim=128
    def forward(self,x):
        h=self.net(x)
        # Preserve coarse event timing in addition to global pooling.
        pooled=torch.cat([h.mean(-1),h.amax(-1),F.adaptive_avg_pool1d(h,3).flatten(1)],dim=1)
        return self.project(pooled)


class LSTMEncoder(nn.Module):
    def __init__(self,channels=5,hidden=48):
        super().__init__()
        self.rnn=nn.LSTM(channels,hidden,num_layers=2,batch_first=True,dropout=.2)
        self.project=nn.Sequential(nn.Linear(hidden*2,128),nn.LayerNorm(128),nn.GELU())
        self.output_dim=128
    def forward(self,x):
        h,_=self.rnn(x.transpose(1,2))
        return self.project(torch.cat([h.mean(1),h[:,-1]],dim=1))


class PlayerModel(nn.Module):
    def __init__(self,kind="cnn",feature_dim=0,width=64,input_channels=12):
        super().__init__()
        self.kind=kind
        self.encoder=LSTMEncoder() if kind=="lstm" else TemporalEncoder(channels=input_channels,width=width)
        dim=self.encoder.output_dim
        self.attention_v=nn.Linear(dim,64)
        self.attention_u=nn.Linear(dim,64)
        self.attention_w=nn.Linear(64,1)
        self.feature_net=nn.Sequential(nn.Linear(feature_dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.35),nn.Linear(128,64),nn.GELU()) if feature_dim else None
        self.head=nn.Sequential(nn.Linear(dim*4+(64 if feature_dim else 0),128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.4),nn.Linear(128,1))
    def forward(self,x,features=None,return_attention=False):
        b,e,c,t=x.shape
        h=self.encoder(x.reshape(b*e,c,t)).reshape(b,e,-1)
        a=self.attention_w(torch.tanh(self.attention_v(h))*torch.sigmoid(self.attention_u(h))).softmax(1)
        pooled=torch.cat([(a*h).sum(1),h.mean(1),h.amax(1),h.std(1,unbiased=False)],dim=1)
        if self.feature_net is not None:pooled=torch.cat([pooled,self.feature_net(features)],dim=1)
        logits=self.head(pooled).squeeze(-1)
        return (logits,a.squeeze(-1)) if return_attention else logits
