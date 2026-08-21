# Focused PreVal diagnostic experiment

This directory is independent of the production estimator and original benchmark results. It never edits `aeon`, `tsml-eval`, or `data/`.

## Local example

From the repository root, with a directory containing `<dataset>/<dataset>_TRAIN.ts` and `_TEST.ts` files:

```text
.venv/Scripts/python.exe analysis/ucr112_resample0/failure_investigation/diagnostic_experiment/run_diagnostic.py ArrowHead MultiRocket --data-dir C:/path/to/Data --design minimal --save-predictions
```

Raw transformed matrices are cached under `feature_cache/<transform>/<dataset>/`. Re-running the command skips the transform and memory-maps the cache.

## Iridis

Edit the configuration block in `iridis_submit.sh`, especially `repo_dir`, `data_dir`, environment and memory, then run:

```text
bash iridis_submit.sh
```

The generated Slurm array runs the full factorial for `cases.csv`. Completed cases contain a `COMPLETE` marker and are skipped on resubmission. The Thorax MultiRocket tasks drive the 32 GB memory request; smaller cases need much less.

After copying or completing results, run:

```text
python summarise_results.py --results-dir results --output-dir summary
```

## Status

The four first-wave cases have completed and are analysed in `first_wave_report.md`. Their copied raw results remain under the repository-level `data/` tree; reproducible processed tables are generated under `summary/`. The first wave identifies uncentred float32 eigendecomposition as the shared numerical failure and clipped-softmax tie-breaking as an additional Thorax1 failure. No production estimator changes have been made.
