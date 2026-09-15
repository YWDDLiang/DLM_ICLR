# Installation

Use Python 3.10–3.12. Install PyTorch for your CUDA driver, then matching PyG and torch-scatter packages following the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
python -m pip install -e '.[train,refine,physics,direct]'
```

The reference environment used PyTorch 2.4.1/CUDA 12.1, Transformers 4.54.0, PEFT 0.16.0, CHGNet 0.4.2 with model 0.3.0, ASE 3.28.0, pymatgen 2025.6.14 and SMACT 3.1.0. CUDA-specific libraries are installed for the target machine.

Composition/structure metrics alone use `pip install -e '.[direct]'`.

Foundation-model access follows provider terms. Set local model directories or authenticate with Hugging Face before using Meta-Llama-3-8B. Materials Project access uses `MP_API_KEY`, a private `--api-key-file`, or a concealed interactive prompt. Query reports contain reference metadata, not credentials.
