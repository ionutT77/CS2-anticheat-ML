"""Fixed event summaries and full-match pooling, shared by training/inference."""
import numpy as np


def summarize_events(x):
    outputs = []
    for a in [x, x[:, -64:]]:
        outputs.extend([a.mean(1), a.std(1), np.quantile(a,.05,axis=1),
                        np.quantile(a,.5,axis=1), np.quantile(a,.95,axis=1),
                        np.abs(a).mean(1), np.abs(a).max(1)])
    return np.concatenate(outputs,axis=1).astype(np.float32)


def summarize_player(event_features):
    return np.concatenate([event_features.mean(0), event_features.std(0),
                           np.quantile(event_features,.9,axis=0)]).astype(np.float32)
