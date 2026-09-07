"""Shared normalization and inference; receives explicit eligible player indices."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def validation_metrics(y, scores, reviewed):
    all_auc = float(roc_auc_score(y,scores))
    reviewed_auc = float(roc_auc_score(y[reviewed],scores[reviewed]))
    return {'all_auc':all_auc,'reviewed_auc':reviewed_auc,'selection_score':(all_auc+reviewed_auc)/2,
            'average_precision':float(average_precision_score(y,scores))}


def normalize(x, mean, std, channels):
    return ((x[...,:channels].float()-mean[:channels])/std[:channels]).clamp(-12,12)


@torch.inference_mode()
def predict_players(model, x, mean, std, bags, player_indices, batch_size=512, horizon=None):
    model.eval()
    selected = [bags[i] if horizon is None else bags[i][:horizon] for i in player_indices]
    lengths = np.array([len(b) for b in selected])
    indices = np.concatenate(selected)
    encodings=[]
    for start in range(0,len(indices),batch_size):
        xx=normalize(x[indices[start:start+batch_size]],mean,std,model.channels)
        encodings.append(model.encoder(xx).float())
    z=torch.cat(encodings)
    starts=np.cumsum(np.r_[0,lengths]); output=[]
    event_logits=model.event_head(z).squeeze(-1).float().cpu().numpy()
    if not model.hierarchical:
        return np.array([event_logits[a:b].mean() for a,b in zip(starts[:-1],starts[1:])]),event_logits,indices
    for start in range(0,len(selected),32):
        lens=lengths[start:start+32]
        zz=torch.zeros((len(lens),lens.max(),64),device=x.device)
        mask=torch.zeros(zz.shape[:2],device=x.device,dtype=torch.bool)
        for j, length in enumerate(lens):
            zz[j,:length]=z[starts[start+j]:starts[start+j+1]]
            mask[j,:length]=True
        scores,_=model.aggregate(zz,mask)
        output.extend(scores.float().cpu().numpy())
    return np.array(output),event_logits,indices
