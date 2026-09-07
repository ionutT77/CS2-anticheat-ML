import unittest
import numpy as np
from scripts.evaluate import thresholds,calibrate


class ThresholdTests(unittest.TestCase):
    def test_low_fpr_threshold_handles_ties_and_small_samples(self):
        p=np.r_[np.zeros(920),np.full(20,.3),np.full(5,.5),[.9],np.linspace(.1,.99,181)]
        y=np.r_[np.zeros(946),np.ones(181)]
        t=thresholds(y,p)
        self.assertLessEqual(np.mean(p[y==0]>=t["fpr_1pct"]),.01)
        self.assertEqual(np.sum(p[y==0]>=t["fpr_0_1pct"]),0)
        self.assertGreater(t["fpr_0_1pct"],p[y==0].max())

    def test_calibration_preserves_ranking(self):
        p=np.linspace(.01,.99,100)
        y=np.r_[np.zeros(65),np.ones(35)]
        slope,intercept,calibrated=calibrate(y,p)
        self.assertGreater(slope,0)
        self.assertTrue(np.all(np.diff(calibrated)>=0))


if __name__=="__main__":unittest.main()
