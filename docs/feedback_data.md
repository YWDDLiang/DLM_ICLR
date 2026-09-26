# Physical Feedback data

Generate feedback data locally with the supplied code and scripts.

| Input | Configuration |
| --- | --- |
| TRAIN structures | `dataset.splits.train` |
| Trained constructor and periodic head | `models.constructor`, `models.periodic` |
| Diffusion checkpoint | `models.diffusion` |
| Materials Project access for physical labels | `MP_API_KEY` |

```bash
bash scripts/05_collect_feedback.sh --config configs/local.json
bash scripts/06_train_feedback.sh --config configs/local.json
```

The collection contains each training source's Plan, reference structure, candidate changes, and paired physical labels: convergence, energy above hull, and novelty. Failed and unresolved evaluations remain recorded.

Collection and training share the configured run directory. The training script validates that source IDs and evaluated geometries match before fitting. Use `--resume` to continue an interrupted stage.

Individual operations are also available through `crystaldlm train feedback --stage ...`; run `crystaldlm train feedback --help` for the interface.
