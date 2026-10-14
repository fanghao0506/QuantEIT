"""Offline 3D viewer and orthogonal projections of saved reconstructions."""
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

from .data import DEFAULT_DATA, load_dataset, load_ground_truth
from .metrics import normalize_positive

COLOR = "#37BEC8"
DEFAULT_THRESHOLD = 0.4
DEFAULT_CAMERA_EYE = dict(x=-0.4, y=2.3, z=0.8)
VIEWS = ((1, "Coronal MIP (x–z)"), (0, "Axial MIP (x–y)"), (2, "Sagittal MIP (y–z)"))


def surface(volume, threshold):
    """Return a display-only isosurface; stored predictions remain untouched."""
    # Marching cubes cannot close a surface that intersects the sampled grid
    # boundary without an exterior sample. A zero-valued virtual border closes
    # the DISPLAY mesh; it does not fill or modify the reconstruction volume.
    padded = np.pad(volume, 1, mode="constant", constant_values=0)
    if not float(padded.min()) < threshold < float(padded.max()):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    vertices, faces, _, _ = marching_cubes(padded, level=threshold)
    return (vertices - 1)[:, [2, 1, 0]], faces


def mesh_trace(volume, threshold, name):
    vertices, faces = surface(volume, threshold)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name,
        color=COLOR, opacity=0.96, flatshading=False, showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.25, roughness=0.45),
        hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra>" + name + "</extra>",
    )


def write_html(predictions, references, frame_ids, output, threshold):
    sequence = len(frame_ids) > 1
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Simulation reference", "QuantEIT reconstruction"))
    fig.add_trace(mesh_trace(references[0], threshold, "Reference"), row=1, col=1)
    fig.add_trace(mesh_trace(predictions[0], threshold, "QuantEIT"), row=1, col=2)
    frames = [go.Frame(name=str(int(frame)), traces=[0, 1], data=[
        mesh_trace(gt, threshold, "Reference"), mesh_trace(pred, threshold, "QuantEIT")
    ]) for gt, pred, frame in zip(references, predictions, frame_ids)]
    z, y, x = predictions.shape[-3:]
    scene = dict(
        xaxis=dict(title="x (voxel)", range=[-1, x]),
        yaxis=dict(title="y (voxel)", range=[-1, y]),
        zaxis=dict(title="z (voxel)", range=[-1, z]),
        aspectmode="data", bgcolor="white",
        camera=dict(eye=DEFAULT_CAMERA_EYE), uirevision="keep-camera",
    )
    fig.update_layout(
        title=dict(text="QuantEIT · 3D lung sequence" if sequence else f"QuantEIT · 3D lung reconstruction · Frame {int(frame_ids[0])}",
                   x=0.5, font=dict(size=23, color="#152637")),
        height=660 if sequence else 600, autosize=True, template="plotly_white", showlegend=False,
        font=dict(family="Arial, sans-serif", size=12, color="#152637"),
        scene=scene, scene2=scene, margin=dict(l=10, r=10, t=110, b=110 if sequence else 20),
        annotations=list(fig.layout.annotations) + [dict(
            text=f"{'Simulated sequence' if sequence else 'Simulated sample'} · positive peak-normalized display · isosurface {threshold:g}",
            x=0.5, y=1.13 if sequence else 1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=11, color="#607080"))],
    )
    if sequence:
        fig.frames = frames
        fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.10, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=350, redraw=True),
                transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                transition=dict(duration=0), mode="immediate")]),
        ])],
        sliders=[dict(x=0.22, len=0.75, y=-0.07, currentvalue=dict(prefix="Frame "), steps=[
            dict(label=str(int(frame)), method="animate", args=[[str(int(frame))], dict(
                frame=dict(duration=0, redraw=True), transition=dict(duration=0), mode="immediate")])
            for frame in frame_ids])],
        )
    fig.write_html(str(output), include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(responsive=True, displaylogo=False))


def _mip_axis(ax, volume, projection_axis, title):
    ax.imshow(volume.max(axis=projection_axis), cmap="magma", vmin=0, vmax=1, origin="lower")
    ax.set_title(title, fontsize=10, color="#172838", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _surface_axis(ax, volume, title, threshold):
    vertices, faces = surface(volume, threshold)
    if len(faces):
        collection = Poly3DCollection(vertices[faces], facecolor=COLOR, edgecolor="none", alpha=0.95)
        ax.add_collection3d(collection)
    z, y, x = volume.shape
    ax.set(xlim=(-1, x), ylim=(-1, y), zlim=(-1, z))
    ax.set_box_aspect((x, y, z))
    ax.view_init(elev=19, azim=100)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, color="#172838", pad=1)


def write_summary(predictions, references, frame_ids, output, threshold):
    selected = sorted(set([0, len(frame_ids) // 2, len(frame_ids) - 1]))
    fig = plt.figure(figsize=(13, 2.65 * len(selected)), facecolor="white")
    gs = fig.add_gridspec(len(selected), 6, left=0.025, right=0.98, bottom=0.06, top=0.91,
                         wspace=0.09, hspace=0.22)
    for row, index in enumerate(selected):
        prefix = f"Frame {int(frame_ids[index]):02d}"
        _surface_axis(fig.add_subplot(gs[row, 0], projection="3d"), references[index], prefix + " · Reference", threshold)
        _surface_axis(fig.add_subplot(gs[row, 1], projection="3d"), predictions[index], "QuantEIT · 3D", threshold)
        _mip_axis(fig.add_subplot(gs[row, 2]), references[index], 1, "Reference · coronal")
        _mip_axis(fig.add_subplot(gs[row, 3]), predictions[index], 1, "QuantEIT · coronal")
        _mip_axis(fig.add_subplot(gs[row, 4]), references[index], 0, "Reference · axial")
        _mip_axis(fig.add_subplot(gs[row, 5]), predictions[index], 0, "QuantEIT · axial")
    title = "QuantEIT reconstruction of a 3D simulated lung sequence" if len(frame_ids) > 1 else f"QuantEIT · 3D simulated lung reconstruction · Frame {int(frame_ids[0])}"
    fig.suptitle(title, fontsize=16,
                 color="#172838", fontweight="bold", y=0.975)
    fig.text(0.5, 0.015, f"Independent positive peak normalization · 3D isosurface level {threshold:g} · MIP color range [0, 1]",
             ha="center", fontsize=9, color="#617080")
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def projection_figure(prediction, reference, frame):
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.1), facecolor="white")
    for col, (projection_axis, title) in enumerate(VIEWS):
        _mip_axis(axes[0, col], reference, projection_axis, "Reference · " + title)
        _mip_axis(axes[1, col], prediction, projection_axis, "QuantEIT · " + title)
    fig.suptitle(f"3D lung reconstruction · Frame {int(frame):02d}",
                 fontsize=14, color="#172838", y=0.99)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.91, bottom=0.06, wspace=0.15, hspace=0.26)
    fig.text(0.5, 0.013, "Positive peak-normalized display", ha="center", fontsize=8, color="#617080")
    return fig


def write_gif(predictions, references, frame_ids, output):
    frames = []
    for prediction, reference, frame in zip(predictions, references, frame_ids):
        fig = projection_figure(prediction, reference, frame)
        stream = BytesIO()
        fig.savefig(stream, format="png", dpi=110)
        stream.seek(0)
        frames.append(Image.open(stream).convert("RGB"))
        plt.close(fig)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=300, loop=0,
                   optimize=False, disposal=2)


def visualize(result_file, output_dir=None, data_dir=DEFAULT_DATA, threshold=DEFAULT_THRESHOLD):
    if not 0 < threshold < 1:
        raise ValueError("Display threshold must lie strictly between 0 and 1")
    result_file = Path(result_file)
    output = Path(output_dir) if output_dir else result_file.parent
    output.mkdir(parents=True, exist_ok=True)
    with np.load(result_file, allow_pickle=False) as arrays:
        predictions = arrays["volumes"].astype(np.float32, copy=True)
        frame_ids = arrays["frame_ids"].copy()
    dataset = load_dataset(data_dir)
    references = dataset.to_volume(load_ground_truth(data_dir, frame_ids))
    if references.shape != predictions.shape:
        raise ValueError("Reference and reconstruction volume dimensions do not match")
    predictions = normalize_positive(predictions)
    references = normalize_positive(references)
    write_summary(predictions, references, frame_ids, output / "summary.png", threshold)
    if len(frame_ids) > 1:
        write_gif(predictions, references, frame_ids, output / "sequence.gif")
    else:
        fig = projection_figure(predictions[0], references[0], frame_ids[0])
        fig.savefig(output / "projections.png", dpi=150, facecolor="white")
        plt.close(fig)
    write_html(predictions, references, frame_ids, output / "viewer_3d.html", threshold)
    (output / "display_settings.json").write_text(json.dumps(dict(
        isosurface_threshold=threshold,
        frame_count=len(frame_ids), frame_ids=frame_ids.tolist(),
        camera_eye=DEFAULT_CAMERA_EYE,
        normalization="independent positive peak per frame",
        mip_uses_unthresholded_values=True,
        surface_virtual_zero_border=True,
        stored_reconstruction_unchanged=True), indent=2), encoding="utf-8")
    return output

