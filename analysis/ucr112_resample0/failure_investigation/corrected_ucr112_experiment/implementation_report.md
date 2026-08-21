# Corrected experimental PreVal implementation report

## Isolation and scope

The corrected estimator is implemented only in the tsml-eval experimental namespace as `tsml_eval._wip.preval.CorrectedPrevalidatedRidgeClassifier`. No aeon estimator file, standalone `preval-full` source, original classifier registration, or original result file has been modified.

The class is a deliberately close copy of the benchmarked PreVal algorithm. Lambda validation and defaults, low-variance filtering, target encoding and centring, float32 decomposition, LOOCV calculation, calibration objective/optimizer, fitted scale, coefficients, and clipped-softmax probability calculation are unchanged.

The two evidence-supported corrections are:

1. After low-variance filtering, retained training columns are centred in float32 before adding the intercept column and constructing the Gram matrix. Full-width training means are stored in `feature_means_`; prediction subtracts exactly those means before applying the fitted coefficients.
2. `_predict` takes the argmax of the final calibrated linear logits. `_predict_proba` separately applies the original clipped softmax to those same logits. Probability clipping can therefore no longer create class-label ties.

## Registration

Six new names are registered in tsml-eval:

- `MRHydraPreValCorrected`
- `RDSTPreValCorrected`
- `MultiRocketPreValCorrected`
- `HydraPreValCorrected`
- `MiniRocketPreValCorrected`
- `RocketPreValCorrected`

The corresponding original `*PreVal` names still instantiate aeon's original `PrevalidatedRidgeClassifier`. RDST retains its original external `StandardScaler(with_mean=True)` and 20-value `10^-4` to `10^4` lambda grid; internal centring is harmlessly redundant there and makes the corrected class consistent across architectures.

## Local verification

Completed checks:

- Corrected focused suite: **8 passed, 2 skipped**. The skips are only the Hydra and MRHydra registration instantiations because Torch is not installed in the local lightweight environment. The other four registrations and preservation of the original registration pass.
- Original aeon PreVal unit suite: **9 passed**. This verifies the preserved original implementation still behaves as before.
- Existing tsml-eval full classifier-name factory test: **passed**.
- Existing tsml-eval invalid-classifier factory test: **passed**.
- Python compilation and `git diff --check`: passed.
- Manifest validation: 112 unique datasets, six classifiers, 672 tasks.
- Iridis submission script: Git Bash syntax check and local `DRY_RUN=1` both passed; the generated array is `0-671` and no job was submitted.

The focused numerical regression constructs a high-dimensional float32 design with large per-feature offsets, confirms that its uncentred Gram matrix has non-positive eigenvalues, and confirms that stored internal centring produces a positive spectrum and finite fitted outputs. The prediction regression constructs calibrated logits above the softmax clipping bound and confirms that probabilities tie on the first class while corrected label prediction retains the true logit ordering. A translation test confirms that fitted means are applied at prediction time.

## Still to verify before full launch

1. In the actual Iridis environment, run the focused suite with Torch installed; all ten tests should pass rather than eight passing/two skipping.
2. Instantiate all six names and run one small smoke dataset through the standard experiment runner, writing to a disposable result root. This verifies environment/package wiring and tsml result-file output, not statistical performance.
3. Confirm the aeon and tsml-eval versions/commits match the original benchmark environment apart from the new tsml-eval commit.
4. Submit a small initial subset including one MultiRocket/Hydra case and inspect peak memory. Internal centring keeps float32 but creates a centred working matrix; the 32 GB request is expected to be sufficient but has not been measured in the full architecture pipelines.
5. Confirm Iridis's dataset directory contains exactly the 112 manifest names and that resample 0 uses the original train/test split without `-pr`.

No UCR112 classifier experiment was run locally.

## Repository handoff

Two separate commits are required:

1. **tsml-eval repository:** commit and push `tsml_eval/_wip/preval/` and the changes to `tsml_eval/experiments/_get_classifier.py`.
2. **preval-aeon-eval repository:** commit and push `analysis/ucr112_resample0/failure_investigation/corrected_ucr112_experiment/` together with the completed diagnostic report/summariser updates on which it depends.

On Iridis, pull both commits explicitly, reinstall or refresh the editable tsml-eval environment, run the pre-launch checks above, edit the submission configuration, perform a dry run, and then submit. The generated `.sub`, logs, results, transformed features, and diagnostic raw artefacts remain ignored and must not be committed.
