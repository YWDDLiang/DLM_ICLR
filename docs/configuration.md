# Configuration and execution

Every command accepts `--config` and repeatable `--set section.key=value`. Values are JSON when possible, otherwise strings. Relative paths resolve beside the configuration; `@run/` resolves below `output`; `hf:` identifies a Hugging Face repository.

[defaults.json](../src/dlm_iclr/defaults.json) is the single source of hyperparameter defaults. `configs/mp20.json` selects the saved independent C2 training Plan pool. `configs/custom.json` uses the supplied dataset's training conditions.

```bash
bash scripts/train.sh planner --config configs/local.json
bash scripts/train.sh b0 --config configs/local.json
bash scripts/train.sh c1 --config configs/local.json
bash scripts/train.sh diffusion --config configs/local.json
bash scripts/train.sh c2 --config configs/local.json
```

C2 exposes `warmup`, `collect`, `label`, `compile`, `editor`, `light`, `value` and `risk` stages. Collection reuses saved raw/refined structures. Physical labels are shared by geometry and protocol. Value features use the final editor.

Training progress and settings are stored inside each module's output. `--resume` restores an interrupted supported stage and recognizes completed C2 stage receipts. B0 supports `torchrun` and derives gradient accumulation from effective batch size and process count.

```text
outputs/DATASET/
├── data/          # prepared views and tokenizer
├── planner/       # epoch1, epoch2
├── b0/            # best, last, final adapter
├── c1/            # periodic head
├── diffusion/     # best and last states
├── c2/            # warmup, collection, labels, editor, light, value, risk
├── hull/          # versioned MP entries
├── cache/         # physical labels and structure matches
└── samples/       # plans, raw, refined, edited and evaluation
```

Sampling outputs may be placed elsewhere with `--output`. Completed requests are reused when their Plan sequence, model files and settings match. A new model/configuration experiment uses a new output directory.

`runtime.devices` lists visible PyTorch devices. Generation, diffusion and physics worker counts are separate; C2 also has `c2.batch_size`. Workers use independent request seeds, and results are joined in saved Plan order.
