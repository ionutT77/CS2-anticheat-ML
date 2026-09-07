import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from anticheat.reference_lstm import ReferenceLSTM, choose_thresholds, predict_checkpoint, pool_scores

ROOT = Path(__file__).resolve().parents[1]


class ReferenceLSTMTests(unittest.TestCase):
    def test_original_architecture_and_outputs_match(self):
        # This fully inspected source file contains only the original model class.
        path = ROOT.parent / "src/models/lstm_detector.py"
        spec = importlib.util.spec_from_file_location("reviewed_reference_model", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        torch.manual_seed(7)
        theirs = original.LSTMAimbotDetector(dropout=0.4).eval()
        ours = ReferenceLSTM().eval()
        ours.load_state_dict(theirs.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in ours.parameters()), 120897)
        x = torch.randn(3, 192, 5)
        with torch.no_grad():
            torch.testing.assert_close(ours(x), theirs(x), rtol=1e-6, atol=1e-7)

    def test_checkpoint_restores_predictions(self):
        torch.manual_seed(8)
        model = ReferenceLSTM().eval()
        raw = np.random.default_rng(8).normal(size=(4, 192, 5)).astype(np.float32)
        raw[..., 4] = 0
        checkpoint = {"config": {"dropout": 0.4, "normalization": "zscore"},
                      "state_dict": model.state_dict(), "mean": torch.zeros(5), "std": torch.ones(5)}
        with torch.no_grad():
            expected = model(torch.from_numpy(raw)).squeeze(-1).numpy()
        actual = predict_checkpoint(raw, checkpoint, batch_size=2)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)

    def test_threshold_sweep_matches_exhaustive_search_with_ties(self):
        rng = np.random.default_rng(9)
        y = np.r_[np.zeros(300, dtype=int), np.ones(50, dtype=int)]
        p = np.round(rng.random(len(y)), 2)
        got = choose_thresholds(y, p)
        candidates = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
        for name, metric in [("accuracy", accuracy_score), ("f1", f1_score)]:
            scores = np.array([metric(y, p >= t) for t in candidates])
            self.assertEqual(got[name], candidates[np.flatnonzero(scores == scores.max())[-1]])
        self.assertLessEqual(np.sum(p[y == 0] >= got["fpr_1pct"]), 3)
        self.assertEqual(np.sum(p[y == 0] >= got["fpr_0_1pct"]), 0)

    def test_aggregation_is_invariant_to_engagement_order(self):
        p = np.random.default_rng(10).random((5, 30))
        for method in ["mean", "logitmean", "top10"]:
            np.testing.assert_allclose(pool_scores(p, method), pool_scores(p[:, ::-1], method))

    def test_bootstrap_auc_matches_sklearn_with_weighted_ties(self):
        from scripts.improve_reference_lstm import weighted_auc_precompute
        rng = np.random.default_rng(11)
        y = np.r_[np.zeros(40, dtype=int), np.ones(10, dtype=int)]
        p = np.round(rng.random(len(y)), 1)
        auc = weighted_auc_precompute(y, p)
        for _ in range(5):
            weights = rng.integers(0, 4, len(y))
            self.assertAlmostEqual(auc(weights), roc_auc_score(y, p, sample_weight=weights), places=12)


if __name__ == "__main__":
    unittest.main()
