# QuantEIT：单帧 3D 肺重建 Demo

本包只包含仿真肺的**第 20 帧**、对应的算子和几何信息，以及 QuantEIT
重建与可视化代码。无需量子硬件或量子模拟器。

论文：[QuantEIT: Ultra-Lightweight Quantum-Inspired Inference for Electrical
Impedance Tomography](https://ieeexplore.ieee.org/abstract/document/11718876)，
IEEE Transactions on Industrial Informatics，2026。

![3D 重建](examples/viewer_preview.png)

## 直接运行

使用 Python 3.10 或更新版本，在仓库根目录打开终端：

```bash
python -m pip install -r requirements.txt
python run_demo.py
```

默认只重建第 20 帧，进行 1000 次 Adam 优化；有 CUDA 就使用 GPU，否则使用 CPU。
真值只在重建完成后用于指标和显示，不参与重建损失。

运行后用浏览器打开 `results/viewer_3d.html`，即可旋转和缩放 3D 图，无需联网。
同时生成重建 NPZ、对比 PNG、指标 CSV 和运行记录。固定配置见 `configs/default.json`。

3D 显示阈值默认为正值峰值的 0.4。它只影响等值面显示，不改变重建数值和指标。

## 直接查看预存结果

用浏览器打开 `examples/viewer_3d.html`。也可以只生成可视化，不需要 PyTorch 或 GPU：

```bash
python -m pip install -r requirements-visualization.txt
python visualize.py
```

## 常用选项

```bash
# 使用 CPU。
python run_demo.py --device cpu --output results/cpu

# 快速检查安装；10 次迭代只验证流程，不代表完整重建质量。
python run_demo.py --epochs 10 --output results/smoke

# 修改已有结果的显示阈值。
python visualize.py --result results/reconstruction.npz --threshold 0.5 --output results/view05

# 检查代码和数据。
python -m unittest discover -s tests -v
```

重复运行时请指定新的输出目录，程序不会覆盖已有重建。CPU 完整重建比 GPU 慢。

## 接入自己的数据

核心代码位于 `quanteit_lung/reconstruction.py`。可直接调用：

```python
from quanteit_lung.data import load_dataset
from quanteit_lung.reconstruction import Reconstructor

dataset = load_dataset()
engine = Reconstructor(dataset, device="auto")
coefficients, info = engine.reconstruct(dataset.voltages[0])
volume = dataset.to_volume(coefficients)
```

如果使用同一个算子，将输入替换为形状 `(328,)` 的 float32 电压向量即可，
但测量顺序和归一化必须与算子一致。更换传感器或几何时，需要同时提供匹配的
敏感矩阵、测量和 FEM 到体网格的映射，见 [data/README.md](data/README.md)。
`Reconstructor` 不需要真值。

示例算子为 `328 × 28153`，体网格按 z、y、x 为 `32 × 32 × 40`。
默认损失包含电压 MSE、3D TV、Laplacian 和 L1。更多实现说明见
[REPRODUCIBILITY.md](REPRODUCIBILITY.md)，引用见 [README.md](README.md)。

代码、仿真样本及预存结果均采用 [MIT 许可证](LICENSE)。

## 引用

如果你的工作使用本代码或 QuantEIT 方法，请引用上面的论文，
DOI：[10.1109/TII.2026.3731100](https://doi.org/10.1109/TII.2026.3731100)。
可复制的 BibTeX 见 [README.md](README.md#citation-and-license) 和 `citation.bib`。
GitHub 的 **Cite this repository** 入口也会优先给出这篇论文的引用。
