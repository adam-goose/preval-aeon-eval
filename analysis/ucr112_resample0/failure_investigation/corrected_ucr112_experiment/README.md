# Corrected PreVal UCR112 experiment

This is a separate resample-0 rerun of the six PreVal architectures on the same 112-dataset development collection. It does not overwrite or rename the original `*PreVal` variants or their results.

## Registered corrected variants

| Architecture | Original | Corrected experimental variant |
|---|---|---|
| MRHydra | `MRHydraPreVal` | `MRHydraPreValCorrected` |
| RDST | `RDSTPreVal` | `RDSTPreValCorrected` |
| MultiRocket | `MultiRocketPreVal` | `MultiRocketPreValCorrected` |
| Hydra | `HydraPreVal` | `HydraPreValCorrected` |
| MiniRocket | `MiniRocketPreVal` | `MiniRocketPreValCorrected` |
| Rocket | `RocketPreVal` | `RocketPreValCorrected` |

The corrected class lives in `tsml_eval._wip.preval`. The original names continue to instantiate aeon's original `PrevalidatedRidgeClassifier`.

## Iridis preparation

The array script runs exactly 672 tasks: six corrected classifiers by 112 datasets, all at resample 0. Results are written to a new `CorrectedPreValUCR112` root. Existing RidgeCV and original PreVal results can therefore be loaded alongside these results for matched baseline/original/corrected comparisons; they do not need to be rerun.

Before submission:

1. Push the tsml-eval commit containing the corrected class, tests, and registrations.
2. Push this repository's commit containing this manifest and submission script.
3. On Iridis, pull both repositories at those exact commits and update/install the editable tsml-eval environment if necessary.
4. Edit the configuration block in `iridis_submit.sh`, particularly repository, data, result, environment, queue, memory, and concurrency settings.
5. Confirm the dataset folder contains all names in `ucr112_datasets.txt`.
6. From this directory run `DRY_RUN=1 bash iridis_submit.sh`; inspect the generated submission file and verify it reports 672 tasks.
7. Run `bash iridis_submit.sh` to submit. Do not add `-pr`: resample 0 must use the same original train/test split semantics as the first benchmark.

The 32 GB request follows the diagnostic work and should be checked against the original UCR112 job records. Internal centring retains float32 but creates a centred working matrix, so peak memory should be monitored on the first large MiniRocket/MultiRocket jobs before opening full concurrency.

## Pre-launch smoke checks

Run the focused corrected-estimator tests explicitly because tsml-eval's default pytest configuration ignores `_wip`. On Iridis/Linux:

```text
python -m pytest -c /dev/null --confcutdir=tsml_eval/_wip/preval/tests tsml_eval/_wip/preval/tests/test_corrected_prevalidated_ridge.py -q
```

On Windows, replace `/dev/null` with `NUL`.

Also instantiate all six names once in the Iridis environment before submission. The tests verify both corrected registration and preservation of the original registration.

## Post-run comparison

Keep result folders under the corrected classifier names above. The next analysis should join each corrected result to the existing architecture-matched RidgeCV and original PreVal rows by dataset and resample, then repeat the UCR112 matched-pair accuracy, runtime, probability-output, and failure/outlier analyses. The held-out 30 datasets remain untouched until the corrected UCR112 development analysis is accepted.
