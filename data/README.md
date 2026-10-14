# Simulated 3D lung EIT data

The data are the original 20-frame conductivity-change sequence and simulated
EIT voltage vectors from the authors' archived Thorax_3D simulation. The complete
sequence is provided separately; the default demo inputs still contain frame 20.

| File | Contents |
|---|---|
| operator_01.npz, operator_02.npz | Ordered float32 blocks of the `(328, 28153)` sensitivity matrix |
| measurements.npz | `voltages`: `(1, 328)` float32; `frame_ids`: `[20]` |
| geometry.npz | FEM-to-grid indices and weights; `grid_shape=(32,32,40)` in z,y,x order |
| ground_truth.npz | `conductivity_changes`: `(1, 28153)` float32; `frame_ids`: `[20]` |
| measurements_20frames.npz | `voltages`: `(20, 328)` float32; `frame_ids`: `1,...,20` |
| ground_truth_20frames.npz | `conductivity_changes`: `(20, 28153)` float32; `frame_ids`: `1,...,20` |
| manifest.json | Dimensions, source-file checksums and distributed-file hashes |

The voltage and operator retain their original prepared normalization. The
demo applies no further input scaling or sign reversal. The two operator files
are concatenated in order without changing values; each is below 20 MB.

Row `i` of both full-sequence arrays corresponds to frame ID `i + 1`.
All 20 frames use the same operator and coordinate map. The frame-20 rows in
the full sequence exactly match the default demo inputs. The reference array
stores signed FEM conductivity changes, not a normalized display image.
No additional noise, normalization or sign reversal is applied to the sequence.
Use NumPy with `allow_pickle=False` to read the files; examples are in the root README.

The compact geometry map replaces the dense projection. It is one-to-one:
each FEM coefficient maps to one volume voxel with the supplied weight.
Unassigned background voxels are zero. Stored volumes have axis order z,y,x.

For another dataset, keep operator columns, FEM coefficients and mapping indices
in the same order, and keep operator rows aligned with voltage entries.
`quanteit_lung.data.LungDataset` holds these arrays; `Reconstructor` accepts a
prepared dataset and a measurement vector. Reference data are required only
for the supplied comparison figures and optional metrics.

These are simulated conductivity-change data, not patient recordings. Inputs
use the root MIT license; please preserve attribution and cite QuantEIT.
