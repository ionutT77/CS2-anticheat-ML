"""Train Ionuț’s exact architecture on the existing audited benchmark.

This is a supplementary experiment on a previously reported test set. All new
choices use validation/calibration only; original benchmark artifacts are kept.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/csgo-mpl")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datetime
import hashlib
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from anticheat.data import ROOT, load_raw, load_splits
from anticheat.metrics import evaluate, ranking
from anticheat.reference_lstm import (
    ReferenceLSTM, transform, pool_scores, predict_checkpoint,
    fit_calibration, apply_calibration, choose_thresholds,
)

OUT = ROOT / "experiments/lstm_revision"
CONFIGS = [
    {"name": "reference_recipe", "normalization": "zscore", "balanced_sampler": True, "positive_weight": 2., "seed": 42, "selection": "validation_weighted_loss"},
    {"name": "natural_zscore", "normalization": "zscore", "balanced_sampler": False, "positive_weight": 1., "seed": 42, "selection": "validation_engagement_auc"},
    {"name": "natural_asinh", "normalization": "asinh", "balanced_sampler": False, "positive_weight": 1., "seed": 42, "selection": "validation_engagement_auc"},
    {"name": "natural_asinh_seed123", "normalization": "asinh", "balanced_sampler": False, "positive_weight": 1., "seed": 123, "selection": "validation_engagement_auc"},
]
for config in CONFIGS:
    config.update(dropout=0.4, lr=5e-4, weight_decay=1e-4, epochs=50, patience=10, batch_size=512)


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_records(ids, clean, cheat):
    ids = np.asarray(ids)
    output = np.empty((len(ids), 30, 192, 5), dtype=np.float32)
    negative = ids < len(clean)
    output[negative] = clean[ids[negative]]
    output[~negative] = cheat[ids[~negative] - len(clean)]
    return output


def train(config, frame, splits, clean, cheat):
    start = time.monotonic()
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ids = np.r_[splits["train"], splits["validation"]]
    ntrain = len(splits["train"]) * 30
    x = torch.empty((len(ids) * 30, 192, 5), dtype=torch.float32, device="cuda")
    for lo in range(0, len(ids), 128):
        part = transform(get_records(ids[lo:lo + 128], clean, cheat), config["normalization"])
        x[lo * 30:(lo + len(part)) * 30] = torch.from_numpy(part.reshape(-1, 192, 5)).cuda()
    # Statistics use training records only. Binary firing remains untouched.
    mean = x[:ntrain].mean(dim=(0, 1)); std = x[:ntrain].std(dim=(0, 1), correction=0).clamp_min(1e-8)
    mean[4] = 0.; std[4] = 1.
    x.sub_(mean).div_(std)
    labels = np.repeat(frame.loc[ids, "label"].to_numpy(), 30)
    y = torch.tensor(labels, dtype=torch.float32, device="cuda")
    val_y = labels[ntrain:]
    model = ReferenceLSTM(config["dropout"]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    reference = config["selection"] == "validation_weighted_loss"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min" if reference else "max", factor=0.5, patience=5,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["positive_weight"], device="cuda"))
    counts = torch.bincount(y[:ntrain].long()).double()
    sample_weights = 1. / counts[y[:ntrain].long()]
    history, best, best_epoch = [], -float("inf"), 0
    name = config["name"]
    print(json.dumps({"training": name, "parameters": sum(p.numel() for p in model.parameters()),
                      "train_engagements": ntrain, "validation_engagements": len(val_y)}), flush=True)
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.monotonic()
        model.train()
        order = (torch.multinomial(sample_weights, ntrain, replacement=True) if config["balanced_sampler"]
                 else torch.randperm(ntrain, device="cuda"))
        losses = []
        for index in order.split(config["batch_size"]):
            if len(index) < config["batch_size"]:
                continue  # Same drop_last training behavior as the published recipe.
            optimizer.zero_grad(set_to_none=True)
            z = model.forward_logits(x[index])
            loss = criterion(z, y[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.detach())
        model.eval()
        val_logits = []
        with torch.inference_mode():
            for lo in range(ntrain, len(x), config["batch_size"]):
                val_logits.append(model.forward_logits(x[lo:lo + config["batch_size"]]))
            logits = torch.cat(val_logits)
            val_loss = float(criterion(logits, y[ntrain:]))
            probabilities = logits.sigmoid().cpu().numpy()
        metrics = ranking(val_y, probabilities)
        score = -val_loss if reference else metrics["roc_auc"]
        scheduler.step(val_loss if reference else metrics["roc_auc"])
        row = {"epoch": epoch, "train_loss": float(torch.stack(losses).mean()),
               "validation_loss": val_loss, "validation_engagement": metrics,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - epoch_start}
        history.append(row)
        print(json.dumps({"run": name, **row}), flush=True)
        if score > best + 1e-5:
            best, best_epoch = score, epoch
            checkpoint = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config, "mean": mean.detach().cpu(), "std": std.detach().cpu(),
                "best_epoch": epoch, "validation_engagement": metrics,
            }
            torch.save(checkpoint, OUT / f"{name}.pt")
            np.savez(OUT / f"{name}_validation.npz", ids=splits["validation"], p=probabilities.reshape(-1, 30))
        (OUT / f"{name}_history.json").write_text(json.dumps(history, indent=2))
        if epoch - best_epoch >= config["patience"]:
            break
    selected = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=True)
    result = {"config": config, "best_epoch": best_epoch, "epochs_run": epoch,
              "seconds": time.monotonic() - start, "validation_engagement": selected["validation_engagement"]}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    del model, optimizer, x, y, sample_weights, logits, val_logits, selected, checkpoint
    torch.cuda.empty_cache()
    return result


def weighted_auc_precompute(y, p):
    order = np.argsort(p, kind="stable")
    yy, pp = np.asarray(y)[order], np.asarray(p)[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(pp) != 0) + 1]
    def auc(weights):
        w = weights[order]
        pos = np.add.reduceat(w * yy, boundaries)
        neg = np.add.reduceat(w * (1 - yy), boundaries)
        return np.sum(pos * (np.cumsum(neg) - neg / 2)) / (pos.sum() * neg.sum())
    return auc


def cluster_intervals(y, probabilities, groups, iterations=1000):
    unique, inverse = np.unique(groups, return_inverse=True)
    events = len(next(iter(probabilities.values()))) == len(y) * 30
    yy = np.repeat(y, 30) if events else y
    functions = {k: weighted_auc_precompute(yy, p) for k, p in probabilities.items()}
    rng = np.random.default_rng(20260907)
    samples = []
    for _ in range(iterations):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[inverse]
        if events:
            weights = np.repeat(weights, 30)
        aucs = {name: fn(weights) for name, fn in functions.items()}
        samples.append([aucs["reference"], aucs["improved"], aucs["improved"] - aucs["reference"]])
    return {name: np.quantile(np.array(samples)[:, i], [0.025, 0.975]).tolist()
            for i, name in enumerate(["reference_auc", "improved_auc", "improved_minus_reference_auc"])}


def main():
    if (OUT / "test_results.json").exists() or (OUT / "selection.json").exists():
        raise RuntimeError("Supplement already frozen/evaluated; use saved results, do not retune on them")
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU access is required for these training runs")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    clean, cheat = load_raw()
    frame, splits = load_splits()
    frame = frame.set_index("record_id", drop=False)
    assert frame.groupby("group").split.nunique().max() == 1
    original = json.loads((ROOT / "models/selection.json").read_text())
    original_hashes = {**original["sha256"], "results/test_metrics.json": file_hash(ROOT / "results/test_metrics.json"),
                       "results/test_predictions.csv": file_hash(ROOT / "results/test_predictions.csv")}
    for rel, expected in original_hashes.items():
        assert file_hash(ROOT / rel) == expected, rel
    protocol = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_commit": "06427ceabd9990a8bc278b21758e1b04aa9d46f7",
        "architecture": "5 -> LSTM128 -> dropout -> LSTM64 -> last state -> FC32/ReLU/dropout -> FC1/sigmoid",
        "parameters": 120897, "configs": CONFIGS,
        "selection": "One corrected model by validation engagement ROC-AUC. Per-model player pooling chosen among mean, logitmean, top10 by validation player ROC-AUC. No ensembles.",
        "calibration": "Separate existing calibration split, natural prevalence, positive-slope Platt calibration, thresholds chosen here only.",
        "test_status": "Supplementary evaluation on the previously reported benchmark test set, not a new untouched holdout. No supplementary test scores used for new model selection.",
        "reference_status": "Published architecture and latest training recipe reimplemented on our overlap-grouped split; not an exact reproduction of journal metrics.",
        "precision": "FP32 inputs and storage; training-time cuDNN TF32 allowed; restored checkpoint inference uses IEEE cuDNN FP32 for batch-size consistency. No AMP or FP16 cache.",
        "torch_version": str(torch.__version__).split("+")[0],
        "original_artifact_hashes": original_hashes,
        "supplement_source_hashes": {name: file_hash(ROOT / name) for name in ["scripts/improve_reference_lstm.py", "anticheat/reference_lstm.py"]},
    }
    initial_protocol = OUT / "protocol_before_inference_precision_fix.json"
    if initial_protocol.exists():
        protocol["started_utc"] = json.loads(initial_protocol.read_text())["started_utc"]
        protocol["inference_precision_amendment"] = "See preserved initial protocol and inference_precision_check.json; numerical inference fix after training, before calibration/test. All candidate configurations and trained weights retained."
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    runs = []
    for config in CONFIGS:
        # A fully saved training run can be resumed without repeating it; partial
        # training is rerun from its declared seed before selection is frozen.
        path = OUT / f"{config['name']}.json"
        runs.append(json.loads(path.read_text()) if path.exists() else train(config, frame, splits, clean, cheat))
    candidates = [r for r in runs if r["config"]["name"] != "reference_recipe"]
    winner = max(candidates, key=lambda r: r["validation_engagement"]["roc_auc"])["config"]["name"]
    names = {"reference": "reference_recipe", "improved": winner}
    pooling, parity = {}, {}
    vy = frame.loc[splits["validation"], "label"].to_numpy()
    for role, name in names.items():
        p = np.load(OUT / f"{name}_validation.npz")["p"]
        metrics = {method: ranking(vy, pool_scores(p, method)) for method in ["mean", "logitmean", "top10"]}
        pooling[role] = max(metrics, key=lambda method: metrics[method]["roc_auc"])
        checkpoint = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=True)
        raw = get_records(splits["validation"][:4], clean, cheat)
        restored = predict_checkpoint(raw, checkpoint, device="cuda")
        cpu = predict_checkpoint(raw[:1], checkpoint, device="cpu")
        parity[role] = {"saved_validation_gpu_max_abs_error": float(np.abs(restored - p[:4]).max()),
                        "cpu_gpu_max_abs_error": float(np.abs(cpu - restored[:1]).max())}
        assert parity[role]["saved_validation_gpu_max_abs_error"] < 2e-4
        assert parity[role]["cpu_gpu_max_abs_error"] < 2e-4
    selection = {"models": names, "player_pooling": pooling, "validation_runs": runs,
                 "parity": parity, "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "sha256": {f"{name}.pt": file_hash(OUT / f"{name}.pt") for name in names.values()}}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))
    print("Frozen supplementary selection: " + json.dumps(names), flush=True)

    cal_ids, test_ids = splits["calibration"], splits["test"]
    cy, ty = (frame.loc[ix, "label"].to_numpy() for ix in [cal_ids, test_ids])
    calibration, results, raw_test = {}, {}, {}
    # Finish all calibration choices before making any new test predictions.
    for role, name in names.items():
        checkpoint = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=True)
        pc = predict_checkpoint(get_records(cal_ids, clean, cheat), checkpoint, device="cuda")
        calibration[role] = {}
        for unit in ["engagement", "player"]:
            y = np.repeat(cy, 30) if unit == "engagement" else cy
            p = pc.ravel() if unit == "engagement" else pool_scores(pc, pooling[role])
            fit = fit_calibration(y, p)
            calibrated = apply_calibration(p, fit)
            calibration[role][unit] = {**fit, "thresholds": choose_thresholds(y, calibrated)}
        np.savez(OUT / f"{role}_calibration_predictions.npz", ids=cal_ids, p=pc)
    (OUT / "calibration.json").write_text(json.dumps(calibration, indent=2))
    for role, name in names.items():
        checkpoint = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=True)
        pt = predict_checkpoint(get_records(test_ids, clean, cheat), checkpoint, device="cuda")
        raw_test[role] = pt
        results[role] = {"model": name}
        for unit in ["engagement", "player"]:
            y = np.repeat(ty, 30) if unit == "engagement" else ty
            p = pt.ravel() if unit == "engagement" else pool_scores(pt, pooling[role])
            fit = calibration[role][unit]
            calibrated = apply_calibration(p, fit)
            results[role][unit] = {
                "ranking": ranking(y, p), "uncalibrated_0_5": evaluate(y, p, 0.5),
                "operating_points": {key: evaluate(y, calibrated, threshold) for key, threshold in fit["thresholds"].items()},
            }
        np.savez(OUT / f"{role}_test_predictions.npz", ids=test_ids, p=pt)
    groups = frame.loc[test_ids, "group"].to_numpy()
    intervals = {
        "engagement": cluster_intervals(ty, {k: v.ravel() for k, v in raw_test.items()}, groups),
        "player": cluster_intervals(ty, {k: pool_scores(v, pooling[k]) for k, v in raw_test.items()}, groups),
    }
    report = {"test_status": protocol["test_status"], "records": len(test_ids), "engagements": len(test_ids) * 30,
              "cheater_records": int(ty.sum()), "legitimate_records": int((ty == 0).sum()),
              "majority_accuracy": float((ty == 0).mean()), "selection": selection,
              "results": results, "group_bootstrap_95pct": intervals, "bootstrap_iterations": 1000}
    for rel, expected in original_hashes.items():
        assert file_hash(ROOT / rel) == expected, rel
    (OUT / "test_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
