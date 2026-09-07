"""Ionuț’s unchanged 120,897-parameter LSTM, with stable training helpers.

Architecture and state-dict names match ionutT77/CS2-anticheat-ML commit
06427ceabd9990a8bc278b21758e1b04aa9d46f7. Training uses logits; forward returns
the same sigmoid output as the original network. No extra channels or pooling
parameters are added to the network.
"""
from contextlib import nullcontext
import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from torch import nn


class ReferenceLSTM(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.lstm1 = nn.LSTM(5, 128, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward_logits(self, x):
        first, _ = self.lstm1(x)
        second, _ = self.lstm2(self.dropout1(first))
        return self.classifier[:-1](second[:, -1]).squeeze(-1)

    def forward(self, x):
        return self.forward_logits(x).sigmoid().unsqueeze(-1)


def transform(raw, kind):
    out = np.array(raw, dtype=np.float32, copy=True)
    if kind == "asinh":
        out[..., :2] = np.arcsinh(out[..., :2])
        out[..., 2:4] = np.arcsinh(out[..., 2:4] / 5)
    elif kind != "zscore":
        raise ValueError(f"Unknown transform: {kind}")
    return out


def pool_scores(probabilities, method):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 30:
        raise ValueError("Player aggregation expects [N,30] engagement scores")
    if method == "mean":
        return p.mean(axis=1)
    if method == "logitmean":
        return expit(logit(np.clip(p, 1e-7, 1 - 1e-7)).mean(axis=1))
    if method == "top10":
        return np.sort(p, axis=1)[:, -10:].mean(axis=1)
    raise ValueError(method)


def fit_calibration(y, p):
    z = logit(np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7))
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
    result = {"slope": float(fit.coef_[0, 0]), "intercept": float(fit.intercept_[0])}
    if result["slope"] <= 0:
        raise ValueError("Calibration reversed ranking; investigate before evaluation")
    return result


def apply_calibration(p, calibration):
    z = logit(np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7))
    return expit(calibration["slope"] * z + calibration["intercept"])


def choose_thresholds(y, probabilities):
    """Exact tied-score sweep in O(N log N); ties prefer the higher threshold."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    yy, pp = y[order], p[order]
    end = np.r_[np.flatnonzero(np.diff(pp) != 0), len(pp) - 1]
    tp = np.r_[0, np.cumsum(yy)[end]]
    fp = np.r_[0, end + 1 - np.cumsum(yy)[end]]
    positives, negatives = y.sum(), (y == 0).sum()
    fn, tn = positives - tp, negatives - fp
    candidates = np.r_[np.nextafter(pp[0], np.inf), pp[end]]
    accuracy = (tp + tn) / len(y)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    result = {
        "accuracy": float(candidates[np.argmax(accuracy)]),
        "f1": float(candidates[np.argmax(f1)]), "default_0_5": 0.5,
    }
    negative_scores = np.sort(p[y == 0])[::-1]
    for name, alpha in [("fpr_1pct", 0.01), ("fpr_0_1pct", 0.001)]:
        permitted = int(np.floor(alpha * len(negative_scores)))
        result[name] = float(np.nextafter(negative_scores[permitted], np.inf))
    return result


def predict_checkpoint(raw, checkpoint, device="cpu", batch_size=512):
    x = np.asarray(raw)
    if x.ndim not in (3, 4) or x.shape[-2:] != (192, 5):
        raise ValueError("Input must have shape [N,192,5] or [N,30,192,5]")
    if x.ndim == 4 and x.shape[1] != 30:
        raise ValueError("Each player record must contain 30 engagements")
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Input must be nonempty and finite")
    if not np.isin(x[..., 4], [0, 1]).all():
        raise ValueError("Firing must be binary")
    model = ReferenceLSTM(checkpoint["config"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    flat = x.reshape(-1, 192, 5)
    output = []
    # cuDNN TF32 can change small-batch LSTM scores relative to training-time
    # validation batches. IEEE inference also agrees closely with CPU outputs.
    precision = torch.backends.cudnn.flags(allow_tf32=False) if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode(), precision:
        for lo in range(0, len(flat), batch_size):
            chunk = transform(flat[lo:lo + batch_size], checkpoint["config"]["normalization"])
            chunk = (chunk - checkpoint["mean"].numpy()) / checkpoint["std"].numpy()
            output.append(model(torch.from_numpy(chunk).to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(output).reshape(x.shape[:-2])
