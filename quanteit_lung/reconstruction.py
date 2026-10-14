"""Analytic QuantEIT representation and independent per-frame optimization.

The two circuit expectation values are evaluated in closed form. No quantum
hardware, circuit simulator, training images, or ground truth enter fitting.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    epochs: int = 1000
    learning_rate: float = 0.12
    laplace_weight: float = 0.001
    tv_weight: float = 0.0022500000000000003
    l1_weight: float = 0.0003333333333333333
    n_circuits: int = 2
    seed: int = 0
    output_activation: str = "signed_sigmoid"
    deterministic_convolutions: bool = True

    def __post_init__(self):
        if self.epochs < 1 or self.n_circuits < 1 or self.learning_rate <= 0:
            raise ValueError("Epochs, circuit count and learning rate must be positive")
        if min(self.laplace_weight, self.tv_weight, self.l1_weight) < 0:
            raise ValueError("Regularization weights must be nonnegative")
        if self.output_activation not in ("signed_sigmoid", "sigmoid"):
            raise ValueError("Unsupported output activation")


def analytic_qexpvals(params):
    theta0, theta1 = params[:, 0], params[:, 1]
    return torch.stack((torch.cos(theta0), torch.cos(theta0) * torch.cos(theta1)), -1).reshape(-1)


class QuantumEIT3D(nn.Module):
    def __init__(self, output_dim, n_circuits=2, output_activation="signed_sigmoid"):
        super().__init__()
        self.params = nn.Parameter(torch.randn(n_circuits, 2) * 0.01)
        self.linear = nn.Linear(2 * n_circuits, output_dim)
        self.output_activation = output_activation

    def forward(self):
        raw = F.linear(analytic_qexpvals(self.params), self.linear.weight, None)
        value = torch.sigmoid(raw + self.linear.bias)
        return 2.0 * (value - 0.5) if self.output_activation == "signed_sigmoid" else value


def laplace_kernel(device):
    values = (1.0 / 26.0) * np.array([
        [[2, 3, 2], [3, 6, 3], [2, 3, 2]],
        [[3, 6, 3], [6, -88, 6], [3, 6, 3]],
        [[2, 3, 2], [3, 6, 3], [2, 3, 2]],
    ], dtype=np.float32)
    return torch.from_numpy(values)[None, None].to(device)


def tv_loss_iso(volume, n_fem):
    vertical = F.pad((volume[:, 1:, :] - volume[:, :-1, :]).square(), (0, 0, 0, 1, 0, 0))
    horizontal = F.pad((volume[:, :, 1:] - volume[:, :, :-1]).square(), (0, 1, 0, 0, 0, 0))
    depth = F.pad((volume[1:, :, :] - volume[:-1, :, :]).square(), (0, 0, 0, 0, 0, 1))
    return torch.sqrt(vertical + horizontal + depth + 1e-10).sum() / n_fem


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Reconstructor:
    """Prepared operator; every frame starts from the same fixed random state."""
    def __init__(self, dataset, config=None, device="auto"):
        self.config = config or Config()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch")
        self.device = torch.device(device)
        if self.config.deterministic_convolutions:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.sensitivity = torch.from_numpy(dataset.sensitivity).to(self.device)
        self.indices = torch.from_numpy(dataset.volume_indices).to(self.device)
        self.weights = torch.from_numpy(dataset.projection_weights).to(self.device)
        self.grid_shape = dataset.grid_shape
        z, y, x = np.unravel_index(dataset.volume_indices, self.grid_shape)
        nz, ny, nx = self.grid_shape
        self.storage_shape = (nz, nx, ny)
        storage_rows = z * (nx * ny) + x * ny + y
        matrix = sparse.coo_matrix((dataset.projection_weights,
            (storage_rows, np.arange(self.sensitivity.shape[1]))),
            shape=(int(np.prod(self.grid_shape)), self.sensitivity.shape[1])).tocsr()
        self.projection = torch.sparse_csr_tensor(
            torch.from_numpy(matrix.indptr.astype(np.int64)),
            torch.from_numpy(matrix.indices.astype(np.int64)),
            torch.from_numpy(matrix.data), size=matrix.shape, device=self.device)
        self.kernel = laplace_kernel(self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            model = QuantumEIT3D(self.sensitivity.shape[1], self.config.n_circuits,
                                 self.config.output_activation)
            self.initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def to_volume(self, vector):
        # Preserve the original CSR forward/backward arithmetic during fitting.
        # Changing summation/gradient order can alter a long float32 Adam trajectory.
        out = torch.sparse.mm(self.projection, vector.unsqueeze(1)).squeeze(1)
        return out.reshape(self.storage_shape).transpose(1, 2).contiguous()

    def objective(self, vector, voltage):
        volume = self.to_volume(vector)
        return (
            F.mse_loss(self.sensitivity @ vector, voltage)
            + self.config.l1_weight * vector.abs().mean()
            + self.config.tv_weight * tv_loss_iso(volume, vector.numel())
            + self.config.laplace_weight * (F.conv3d(volume[None, None], self.kernel, padding=1).square().sum() / vector.numel())
        )

    def reconstruct(self, voltage, progress=None):
        """Accept one voltage frame; return signed coefficients and run diagnostics."""
        v = np.asarray(voltage, dtype=np.float32)
        if v.shape != (self.sensitivity.shape[0],) or not np.isfinite(v).all():
            raise ValueError("Expected one finite measurement vector")
        v = torch.from_numpy(v.copy()).to(self.device)
        model = QuantumEIT3D(self.sensitivity.shape[1], self.config.n_circuits,
                             self.config.output_activation).to(self.device)
        model.load_state_dict(self.initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        losses = []
        synchronize(self.device)
        started = perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            loss = self.objective(model(), v)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch == 1 or epoch % 100 == 0 or epoch == self.config.epochs:
                value = float(loss.detach().cpu())
                if not np.isfinite(value):
                    raise RuntimeError(f"Optimization became nonfinite at epoch {epoch}")
                losses.append((epoch, value))
                if progress is not None:
                    progress(epoch, value)
        synchronize(self.device)
        elapsed = perf_counter() - started
        with torch.no_grad():
            vector = model().cpu().numpy().copy()
        if not np.isfinite(vector).all():
            raise RuntimeError("The reconstructed field is nonfinite")
        return vector, {"optimization_seconds": elapsed, "loss_history": losses}

