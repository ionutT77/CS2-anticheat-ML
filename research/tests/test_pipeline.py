"""Scientific invariants: no cross-split duplicates, no future data in pre features."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from anticheat.data import ROOT,load_splits
from anticheat.features import player_features
from anticheat.neural import PlayerModel,sequence_channels
from anticheat.inference import predict_component


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(1)
        self.raw=self.rng.normal(size=(2,30,192,5)).astype(np.float32)
        self.raw[...,4]=(self.raw[...,4]>1).astype(np.float32)

    def test_player_summaries_ignore_engagement_order(self):
        a,n=player_features(self.raw)
        b,m=player_features(self.raw[:,::-1].copy())
        self.assertEqual(n,m)
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)

    def test_pre_features_cannot_see_post_engagement(self):
        a,n=player_features(self.raw)
        changed=self.raw.copy();changed[:,:,160:,:4]*=100
        b,_=player_features(changed)
        columns=[i for i,k in enumerate(n) if k.startswith(("pre.","near."))]
        np.testing.assert_array_equal(a[:,columns],b[:,columns])

    def test_neural_aggregation_ignores_engagement_order(self):
        torch.set_num_threads(2);torch.manual_seed(1)
        model=PlayerModel("cnn",width=16).eval()
        x=sequence_channels(torch.from_numpy(self.raw))
        with torch.no_grad():
            a=model(x);b=model(x.flip(1))
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)

    def test_no_duplicate_groups_cross_splits(self):
        frame,splits=load_splits()
        self.assertEqual(frame.groupby("group").split.nunique().max(),1)
        self.assertEqual(frame.groupby("sha256").split.nunique().max(),1)
        self.assertTrue((frame.loc[frame.split=="test","initial_split"]=="test").all())
        self.assertFalse(frame.loc[frame.split=="calibration","initial_split"].isin(["train","validation"]).any())
        for name,a in splits.items():
            for other,b in splits.items():
                if other!=name:self.assertFalse(np.intersect1d(a,b).size)
        edges_path=ROOT/"data/processed/overlap_duplicate_edges.npy"
        if edges_path.exists():
            edges=np.load(edges_path,allow_pickle=False)
            groups=frame.group.to_numpy()
            self.assertTrue(np.all(groups[edges[:,0]]==groups[edges[:,1]]))

    def test_nonfinite_input_rejected(self):
        self.raw[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):player_features(self.raw)

    def test_inference_restores_saved_weights(self):
        model=PlayerModel("cnn",width=16)
        with torch.no_grad():
            model.head[-1].weight.zero_();model.head[-1].bias.fill_(2.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"models").mkdir()
            torch.save({"state_dict":model.state_dict(),"config":{"kind":"cnn","width":16},"feature_dim":0},root/"models/example.pt")
            with patch("anticheat.inference.ROOT",root):
                p=predict_component("example",self.raw,device="cpu")
        np.testing.assert_allclose(p,1/(1+np.exp(-2)),atol=1e-6)


if __name__=="__main__":unittest.main()
