# Release validation

Validation targets the moved numerical kernels and public interfaces:

- Ten numerical/interface checks cover tree partition functions, marginals and gradients; risk tilting and KL budget; response masking; continuous patching; data mappings; MP subsystem caching; and all-request metrics.
- Real MP-20 preparation was compared on 8 TRAIN, 4 VAL and 4 TEST rows using the actual tokenizers. Planner prompts/answers and B0 prompts/answers matched the existing prepared records.
- The retained C1 checkpoint, 8192-input value model and 13-feature risk model loaded through the new modules. Eight recorded risk scores were reproduced with zero absolute difference.
- Full Direct, including fingerprints and coverage, ran on a saved crystal/reference fixture.
- The supplied diffusion weights ran a real-crystal forward/backward pass with finite gradients and a one-step sampling check. CHGNet ran a one-step CPU relaxation through the moved evaluator; the shortened check retained its nonconverged status.
- Wheel packaging, the installed CLI, shell syntax and local documentation links were checked.

Run portable checks with `python -m pytest -q`. The reference 1050 experiment supplies artifact provenance and default settings; a new full training or generation run produces its own outputs and measurements.
