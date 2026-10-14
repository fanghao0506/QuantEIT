# Implementation notes

The default demo uses one simulated lung sample with original frame ID 20.
The complete 20-frame voltage and reference sequence is also included separately
for independent use, with the same operator, coordinate map and prepared normalization.
The analytic circuit expectations, signed sigmoid output, Adam optimizer and
MSE/TV/Laplacian/L1 objective follow the authors' archived 3D implementation.

Each reconstruction starts from the same fixed initial state. Reference fields
never enter the reconstruction loss or gradient. The compact coordinate map
rebuilds the original sparse CSR projection, with volume axes in z, y, x order.

Deterministic cuDNN convolution is enabled. Different devices and PyTorch
versions can still produce floating-point differences. The supplied numerical
inputs and example result have checksums in data/manifest.json and SHA256SUMS.txt.

Saved coefficients and volumes retain signed amplitudes. Visualization and
descriptive metrics use positive clipping and independent peak normalization.
G-SSIM is global volume SSIM; RIE compares the normalized volumes. Metrics use
all voxels and do not apply the 3D display threshold.

The default isosurface level is 0.4. A virtual zero-valued border closes the
display mesh at grid boundaries; it does not modify the saved volume. Maximum
intensity projections retain unthresholded values. Reference and reconstruction
use the same camera and spatial axes.

Optimization timings synchronize CUDA before and after the timed loop. They
exclude preparation, output saving, metrics and rendering. Checked package
versions and local validation are recorded in examples/verification.json.
