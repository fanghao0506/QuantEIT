"""Portable numerical inputs; reference images are loaded separately."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"


@dataclass(frozen=True)
class LungDataset:
    sensitivity: np.ndarray
    voltages: np.ndarray
    frame_ids: np.ndarray
    volume_indices: np.ndarray
    projection_weights: np.ndarray
    grid_shape: tuple

    def to_volume(self, vectors):
        """Map FEM coefficients to [z, y, x], preserving signed amplitudes."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim < 1 or vectors.shape[-1] != self.sensitivity.shape[1]:
            raise ValueError("The last vector dimension must match the FEM coefficients")
        out = np.zeros(vectors.shape[:-1] + (int(np.prod(self.grid_shape)),), np.float32)
        out[..., self.volume_indices] = vectors * self.projection_weights
        return out.reshape(vectors.shape[:-1] + self.grid_shape)


def load_dataset(data_dir=DEFAULT_DATA):
    """Load only the operator, measurements and coordinate map, never ground truth."""
    directory = Path(data_dir)
    blocks = []
    for name in ("operator_01.npz", "operator_02.npz"):
        with np.load(directory / name, allow_pickle=False) as arrays:
            blocks.append(arrays["sensitivity"].astype(np.float32, copy=True))
    sensitivity = np.concatenate(blocks, axis=0)
    with np.load(directory / "measurements.npz", allow_pickle=False) as arrays:
        voltages = arrays["voltages"].astype(np.float32, copy=True)
        frame_ids = arrays["frame_ids"].astype(np.int32, copy=True)
    with np.load(directory / "geometry.npz", allow_pickle=False) as arrays:
        indices = arrays["volume_indices"].astype(np.int64, copy=True)
        weights = arrays["projection_weights"].astype(np.float32, copy=True)
        shape = tuple(int(i) for i in arrays["grid_shape"])
    n = sensitivity.shape[1] if sensitivity.ndim == 2 else 0
    if sensitivity.ndim != 2 or voltages.shape != (len(frame_ids), sensitivity.shape[0]):
        raise ValueError("Operator and measurement dimensions do not agree")
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected a positive three-dimensional grid shape")
    if indices.shape != (n,) or weights.shape != (n,):
        raise ValueError("Coordinate map dimensions do not agree with the operator")
    if np.unique(indices).size != n or indices.min() < 0 or indices.max() >= np.prod(shape):
        raise ValueError("The coordinate map must be one-to-one and inside the grid")
    if np.unique(frame_ids).size != len(frame_ids):
        raise ValueError("Frame IDs must be unique")
    if not all(np.isfinite(a).all() for a in (sensitivity, voltages, weights)):
        raise ValueError("Nonfinite values in the supplied numerical inputs")
    return LungDataset(sensitivity, voltages, frame_ids, indices, weights, shape)


def load_ground_truth(data_dir, frame_ids):
    """Used only for evaluation/visualization after reconstruction."""
    with np.load(Path(data_dir) / "ground_truth.npz", allow_pickle=False) as arrays:
        ids = arrays["frame_ids"].tolist()
        rows = [ids.index(int(frame)) for frame in frame_ids]
        return arrays["conductivity_changes"][rows].astype(np.float32, copy=True)

