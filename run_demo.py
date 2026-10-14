"""Reconstruct the included frame-20 lung sample, evaluate and visualize it."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from quanteit_lung.data import DEFAULT_DATA, ROOT, load_dataset, load_ground_truth
from quanteit_lung.metrics import evaluate_volume
from quanteit_lung.reconstruction import Config, Reconstructor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames", type=int, nargs="+", help="Input frame IDs; the included sample is frame 20")
    parser.add_argument("--epochs", type=int, help="Override the fixed 1000-epoch configuration")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config_dict = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    config = Config(**config_dict)
    dataset = load_dataset(args.data_dir)
    ids = dataset.frame_ids.tolist()
    requested = args.frames if args.frames is not None else ids
    if len(set(requested)) != len(requested) or any(i not in ids for i in requested):
        parser.error(f"--frames must contain unique IDs from the supplied dataset: {ids}")
    rows = [ids.index(i) for i in requested]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "reconstruction.npz"
    if result_file.exists():
        parser.error("Output already contains reconstruction.npz; select a different --output directory")
    started = perf_counter()
    reconstructor = Reconstructor(dataset, config, args.device)
    preparation_seconds = perf_counter() - started
    print(f"Device: {reconstructor.device}; operator: {dataset.sensitivity.shape}; grid (z,y,x): {dataset.grid_shape}", flush=True)
    print(f"Reconstructing {len(rows)} frame(s), {config.epochs} epochs per frame; no temporal warm start.", flush=True)
    vectors, times, histories = [], [], []
    for row in rows:
        frame = ids[row]
        vector, details = reconstructor.reconstruct(dataset.voltages[row])
        vectors.append(vector)
        times.append(details["optimization_seconds"])
        histories.append(details["loss_history"])
        print(f"Frame {frame:02d}: {times[-1]:.3f} s; objective {histories[-1][0][1]:.6g} → {histories[-1][-1][1]:.6g}", flush=True)
    vectors = np.stack(vectors)
    volumes = dataset.to_volume(vectors)
    np.savez_compressed(result_file, coefficients=vectors, volumes=volumes,
                        frame_ids=np.asarray(requested, np.int32), optimization_seconds=np.asarray(times))
    metadata = dict(config=asdict(config), frame_ids=requested, device=str(reconstructor.device),
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    cpu_threads=args.threads, preparation_seconds=preparation_seconds,
                    timing_scope="synchronized optimization loop; excludes preparation, IO, metrics and visualization",
                    initialization="same fixed random initial state independently for every frame",
                    total_reconstruction_wall_seconds=perf_counter() - started,
                    loss_history=histories,
                    sensitivity_sha256=hashlib.sha256(dataset.sensitivity.tobytes()).hexdigest(),
                    voltages_sha256=hashlib.sha256(dataset.voltages.tobytes()).hexdigest())
    if reconstructor.device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(reconstructor.device)
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Reference data are opened only after fitting is complete.
    if not args.no_metrics:
        truth = dataset.to_volume(load_ground_truth(args.data_dir, requested))
        metrics = []
        for frame, prediction, reference, row, seconds in zip(requested, volumes, truth, rows, times):
            record = dict(frame=frame, optimization_seconds=seconds, **evaluate_volume(reference, prediction))
            pred_v = dataset.sensitivity @ vectors[len(metrics)]
            record["relative_measurement_residual"] = float(np.linalg.norm(pred_v - dataset.voltages[row]) / max(np.linalg.norm(dataset.voltages[row]), 1e-12))
            metrics.append(record)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Mean G-SSIM: {np.mean([m['g_ssim'] for m in metrics]):.4f}; mean RIE: {np.mean([m['rie'] for m in metrics]):.4f}", flush=True)
    if not args.no_visualize:
        from quanteit_lung.visualization import visualize
        visualize(result_file, data_dir=args.data_dir)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()

