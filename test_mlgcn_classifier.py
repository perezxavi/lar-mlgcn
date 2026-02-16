import unittest

import numpy as np

from MLGNNClassifiers import LaRMLGCNClassifier


class TestLaRMLGCNClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LaRMLGCNClassifier(
            input_dim=4,
            num_labels=3,
            proj_dim=8,
            mlp_hidden_dims=(16,),
            gcn_hidden_dims=(8,),
            device="cpu",
        )
        self.Y = np.array(
            [
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
            ],
            dtype=np.int64,
        )

    def test_build_label_graph_shape_and_row_sums(self) -> None:
        A_hat = self.model.build_label_graph(self.Y)
        self.assertEqual(tuple(A_hat.shape), (3, 3))
        row_sums = A_hat.sum(dim=1).cpu().numpy()
        np.testing.assert_allclose(row_sums, np.ones(3, dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_build_label_graph_invalid_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model.build_label_graph(np.array([1, 0, 1], dtype=np.int64))

    def test_predict_proba_invalid_feature_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model.predict_proba(np.random.randn(5, 5).astype(np.float32))

    def test_fit_invalid_sample_count_raises(self) -> None:
        X = np.random.randn(6, 4).astype(np.float32)
        Y = np.random.randint(0, 2, size=(5, 3)).astype(np.int64)
        with self.assertRaises(ValueError):
            self.model.fit(X, Y, epochs=1, batch_size=2, verbose=False)

    def test_fit_and_predict_proba_integration(self) -> None:
        X = np.random.randn(24, 4).astype(np.float32)
        Y = np.random.randint(0, 2, size=(24, 3)).astype(np.int64)

        self.model.fit(X, Y, epochs=2, batch_size=8, lr=1e-3, verbose=False, seed=7)
        proba = self.model.predict_proba(X[:10])

        self.assertEqual(proba.shape, (10, 3))
        self.assertTrue(np.all(proba >= 0.0))
        self.assertTrue(np.all(proba <= 1.0))

    def test_predict_binary_output_integration(self) -> None:
        X = np.random.randn(20, 4).astype(np.float32)
        Y = np.random.randint(0, 2, size=(20, 3)).astype(np.int64)

        self.model.fit(X, Y, epochs=1, batch_size=5, lr=1e-3, verbose=False, seed=11)
        pred = self.model.predict(X[:6], threshold=0.5)

        self.assertEqual(pred.shape, (6, 3))
        unique_vals = np.unique(pred)
        self.assertTrue(np.all(np.isin(unique_vals, [0, 1])))


if __name__ == "__main__":
    unittest.main()

