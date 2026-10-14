import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from quanteit_lung.data import DEFAULT_DATA, load_dataset
from quanteit_lung.reconstruction import Config, Reconstructor, analytic_qexpvals


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.dataset = load_dataset()

    def test_circuit_expectations_against_two_qubit_state(self):
        theta = torch.tensor([[0.31, 1.23], [-0.72, 0.94]], dtype=torch.float64)
        expected = []
        for t0, t1 in theta.numpy():
            # RY(t0) ⊗ RY(t1), followed by CNOT(control=0,target=1).
            a, b = np.cos(t0 / 2), np.sin(t0 / 2)
            c, d = np.cos(t1 / 2), np.sin(t1 / 2)
            probabilities = np.array([a * c, a * d, b * d, b * c]) ** 2
            expected.extend([probabilities @ [1, 1, -1, -1], probabilities @ [1, -1, 1, -1]])
        np.testing.assert_allclose(analytic_qexpvals(theta).numpy(), expected, atol=1e-14)

    def test_coordinate_map_matches_original_grid_indices(self):
        with np.load(DEFAULT_DATA / "geometry.npz", allow_pickle=False) as arrays:
            y, x, z = arrays["source_indices_yxz"].T
        vector = np.arange(self.dataset.sensitivity.shape[1], dtype=np.float32)
        volume = self.dataset.to_volume(vector)
        np.testing.assert_array_equal(volume[z, y, x], vector * self.dataset.projection_weights)

    def test_projection_gradient(self):
        r = Reconstructor(self.dataset, Config(epochs=1), "cpu")
        vector = torch.randn(self.dataset.sensitivity.shape[1], requires_grad=True)
        r.to_volume(vector).square().sum().backward()
        expected = 2 * vector.detach().numpy() * self.dataset.projection_weights ** 2
        np.testing.assert_allclose(vector.grad.numpy(), expected, atol=1e-6)

    def test_per_frame_independence_without_reference_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("operator_01.npz", "operator_02.npz", "geometry.npz", "measurements.npz"):
                shutil.copyfile(DEFAULT_DATA / name, Path(temporary) / name)
            data = load_dataset(temporary)  # No reference file exists in this directory.
            r = Reconstructor(data, Config(epochs=2), "cpu")
            first, _ = r.reconstruct(data.voltages[0])
            r.reconstruct(data.voltages[0] * np.float32(0.9))
            again, _ = r.reconstruct(data.voltages[0])
            np.testing.assert_array_equal(first, again)
            self.assertTrue(np.isfinite(first).all())

    def test_data_integrity(self):
        manifest = json.loads((DEFAULT_DATA / "manifest.json").read_text())
        for name, record in manifest["files"].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((DEFAULT_DATA / name).read_bytes()).hexdigest(), record["sha256"])
        self.assertEqual(self.dataset.voltages.shape, (1, 328))
        np.testing.assert_array_equal(self.dataset.frame_ids, [20])

    def test_invalid_measurement_is_rejected(self):
        r = Reconstructor(self.dataset, Config(epochs=1), "cpu")
        with self.assertRaises(ValueError):
            r.reconstruct(np.zeros(327))
        with self.assertRaises(ValueError):
            r.reconstruct(np.full(328, np.nan))


if __name__ == "__main__":
    unittest.main()

