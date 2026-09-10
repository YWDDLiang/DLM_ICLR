# Release validation

The release was exercised with the actual model assets in the environment described in [environment.md](environment.md). Machine-readable results are in [validation.json](validation.json).

The extracted G implementation reproduced all token outputs for four fixed H1A2 requests. E reproduced the proposal tokens, forward counts and 8192-dimensional value features for 16 fixed inputs with eight candidate streams each. A file-access check confirmed that E did not read physical-label inputs during selection. F reproduced the 800-step output exactly for the matching-seed fixture. Eight independent F processes on one GPU preserved that fixture's output and improved aggregate throughput; the timing is a single-fixture execution benchmark.

After real G and E updates, saving and loading the compact checkpoints preserved G logits and E logits plus all four decision-head outputs bitwise. The exported G and E checkpoints were approximately 2.18 GB and 2.21 GB respectively, with their required input/output tables retained.

A complete engineering run used 16 TRAIN conditions and four separate evaluation requests. It generated and physically evaluated TRAIN candidates, independently evaluated full-token G teachers, compiled feedback, updated G and E, fitted a value model with the updated E features, and completed S1 inference and evaluation. G performed three optimizer updates on 10 sources; E performed four content and decision updates on 15 sources; the value model performed eight updates on 120 rows from 15 sources, visiting every row twice. All three models had nonzero parameter changes. One TRAIN condition lacked a Yb reference and did not provide a reliable stability target.

This integration run intentionally used one actor epoch and two value epochs. Both four-request snapshots had SUN 0 and MSUN 3. It establishes that the complete workflow runs; it is not evidence of scientific improvement or a substitute for the formal experiments.

All 20 unit tests passed without skips in the reference environment. They include exact Plan selection and seed preservation, TRAIN exclusions, continuous KEEP and local-coordinate preservation, consistent atom permutations, missing reference coverage, unknown metric bounds and portable value-model loading. CIF-directory, CIF-CSV and structure-JSONL imports were also exercised with preserved source splits and explicit out-of-representation failures.
