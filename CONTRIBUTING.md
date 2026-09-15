# Contributing

Changes should target the corresponding module API. Keep shell scripts thin and put defaults in `src/dlm_iclr/defaults.json`. Dataset paths and column names belong in configuration rather than model code.

Run `python -m pytest` for numerical and interface checks. Tests focus on probability calculations, continuous patching, ordered metrics and reusable input/configuration behavior. Model-scale validation should record the checkpoint, data split and execution settings in the experiment output.

Provide a short explanation of the affected scientific behavior and how it was verified. Derived third-party numerical code retains its original notice and citation.
