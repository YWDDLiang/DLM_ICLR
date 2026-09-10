# Environment

The implementation is tested on Python 3.10 and NVIDIA A800 GPUs. Data-only CLI commands do not import PyTorch. CUDA model inference requires the model extras and matching PyG binary extensions.

The retained numerical environment is:

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch / CUDA | 2.4.0 / 12.1 |
| Transformers | 4.54.0 |
| PEFT | 0.16.0 |
| safetensors | 0.5.3 |
| torch-scatter | 2.1.2, built for PyTorch 2.4 / CUDA 12.1 |
| torch-geometric | 1.7.2 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| pymatgen | 2025.6.14 |
| CHGNet package | 0.4.2 |
| CHGNet weights | 0.3.0 |
| ASE | 3.28.0 |
| SMACT | 3.1.0 |

The CHGNet package version and weight version are different quantities. The evaluation uses the 0.3.0 weights through the 0.4.2 library.

Install PyTorch for your CUDA runtime, then install `torch-scatter` from the [official PyG wheel index](https://data.pyg.org/whl/). Select the wheel family matching both PyTorch and CUDA. For the retained environment:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install -e '.[models,refiner,physics]'
```

The ranges in `pyproject.toml` support installation; they are not a claim that every permitted library combination has numerical parity with the recorded environment. Use the table above when reproducing the reported results. The DataLoader adapter supports both the older `torch_geometric.data` location and the newer `torch_geometric.loader` location.

Every model worker uses one CPU thread and deterministic PyTorch execution, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each F request uses batch size one and resets its own Python, NumPy and PyTorch random streams before creating the DataLoader iterator. Independent processes can share a GPU without changing the per-request sampling definition. The default is one refiner worker per GPU; `--refiner-workers` changes process concurrency.

Shell and Slurm scripts call the same installed package. Slurm partition names, GPU count, CPU allocation and wall time are cluster settings supplied to `sbatch`; the Python package does not require Slurm.
