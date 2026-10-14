"""Descriptive volume metrics; positive peak normalization is explicit."""
import numpy as np


def normalize_positive(array):
    array = np.maximum(np.asarray(array, dtype=np.float32), 0)
    maximum = array.max(axis=(-3, -2, -1), keepdims=True)
    return np.divide(array, maximum, out=np.zeros_like(array), where=maximum > 1e-12)


def evaluate_volume(reference, prediction):
    reference = normalize_positive(reference).astype(np.float64)
    prediction = normalize_positive(prediction).astype(np.float64)
    ref = reference.ravel()
    pred = prediction.ravel()
    ref_c = ref - ref.mean()
    pred_c = pred - pred.mean()
    vr = np.mean(ref_c ** 2)
    vp = np.mean(pred_c ** 2)
    cov = np.mean(ref_c * pred_c)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * ref.mean() * pred.mean() + c1) * (2 * cov + c2)
            / ((ref.mean() ** 2 + pred.mean() ** 2 + c1) * (vr + vp + c2)))
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    return {
        "g_ssim": float(ssim),
        "rie": float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-12)),
        "cc": float(cov / np.sqrt(vr * vp)) if vr * vp > 0 else float("nan"),
        "psnr_db": float(20 * np.log10(1 / max(rmse, 1e-12))),
    }
