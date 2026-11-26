"""Unit tests for src/ml/metrics.py.  Run:  python -m unittest discover -s tests -v"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ml"))
import metrics as m  # noqa: E402


class Accuracy(unittest.TestCase):
    def test_mae_rmse(self):
        self.assertAlmostEqual(m.mae([1, 2, 3], [1, 4, 0]), (0 + 2 + 3) / 3)
        self.assertAlmostEqual(m.rmse([1, 2, 3], [1, 4, 0]), math.sqrt((0 + 4 + 9) / 3))

    def test_length_and_empty_errors(self):
        with self.assertRaises(ValueError):
            m.mae([1], [1, 2])
        with self.assertRaises(ValueError):
            m.rmse([], [])


class Correlation(unittest.TestCase):
    def test_pearson(self):
        self.assertAlmostEqual(m.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(m.pearson([1, 2, 3], [3, 2, 1]), -1.0)

    def test_constant_prediction_has_no_correlation(self):
        self.assertIsNone(m.pearson([0, 0, 0], [1, 5, 2]))

    def test_ranks_with_ties(self):
        self.assertEqual(m.ranks([10, 20, 20, 5]), [2.0, 3.5, 3.5, 1.0])

    def test_spearman_is_scale_free(self):
        # Monotonic but non-linear relationship: Spearman = 1, Pearson < 1.
        x, y = [1, 2, 3, 4], [1, 4, 9, 100]
        self.assertAlmostEqual(m.spearman(x, y), 1.0)
        self.assertLess(m.pearson(x, y), 1.0)


class Ranking(unittest.TestCase):
    def test_top_k_per_group(self):
        gw = [1, 1, 1, 2, 2, 2]
        pred = [5, 1, 3, 2, 9, 1]
        actual = [10, 0, 2, 4, 1, 7]
        # GW1 top-1 = 10, GW2 top-1 = 1  -> mean 5.5
        self.assertAlmostEqual(m.top_k_actual(gw, pred, actual, 1), 5.5)
        # GW1 top-2 = (10+2)/2 = 6, GW2 top-2 = (1+4)/2 = 2.5 -> 4.25
        self.assertAlmostEqual(m.top_k_actual(gw, pred, actual, 2), 4.25)

    def test_ties_never_use_actual(self):
        # Equal predictions: the first row wins, even though the second scored more.
        self.assertEqual(m.top_k_actual([1, 1], [3, 3], [0, 20], 1), 0)

    def test_k_larger_than_group(self):
        self.assertAlmostEqual(m.top_k_actual([1, 1], [1, 2], [4, 6], 5), 5.0)


class Summary(unittest.TestCase):
    def test_keys(self):
        s = m.summarise([1, 2], [1, 3])
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["mae"], 0.5)
        self.assertAlmostEqual(s["mean_actual"], 2.0)


if __name__ == "__main__":
    unittest.main()
