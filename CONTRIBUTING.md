# Contributing

Corrections, portability improvements, and carefully scoped follow-up
experiments are welcome.

## Before opening a pull request

Run the dependency-free release checks:

```bash
python3 scripts/validate_repository.py
python3 -m compileall -q scripts
git diff --check
```

If you change corpus derivation, generated supervision, splitting, or model
evaluation, also run the stage-specific validators documented in
[docs/REPRODUCING.md](docs/REPRODUCING.md). State which raw PDF hash, model
revision, seed, hardware, and command produced new results.

## Preserve the experimental record

- Do not edit generated JSON by hand to make metrics or counts match.
- Keep source-unit grouping intact across train, development, and test.
- Select model checkpoints on development data only.
- Preserve raw predictions and failed generation attempts under ignored
  `artifacts/` paths.
- Regenerate `RESULTS.md` and `results/whole-child-results.json` with
  `scripts/freeze_results.py` when detailed result artifacts change.
- Distinguish an exact reproduction change from an extension or ablation.

## Large files and licensing

Do not commit model weights, raw PDFs, embeddings, generated-question audit
logs, or detailed prediction files. The `.gitignore` keeps them under
`artifacts/`, local model caches, and `data/raw/`.

Code contributions are accepted under MIT. Textbook-derived data remain under
CC BY-NC-SA 4.0; review [DATA_LICENSE.md](DATA_LICENSE.md) before modifying or
redistributing data files.

## Style

The project favors small, explicit Python scripts with deterministic paths and
human-readable JSON/JSONL artifacts. Keep commands runnable from the repository
root, pin externally downloaded model revisions, and add a validator whenever
new generated state becomes part of a scientific claim.
