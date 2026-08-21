# Diagnostic experiment status and interpretation

## What has been implemented

The experiment is fully isolated beneath this directory. `diagnostic_core.py` is an independent diagnostic implementation and does not patch or import the production `PrevalidatedRidgeClassifier`. `run_diagnostic.py` fits each MiniRocket/MultiRocket transform once, caches raw train/test features, and reuses them for every intervention. A full case produces 48 PreVal configurations, two RidgeCV references, lambda-path diagnostics and optional predictions.

The code passed a synthetic high-dimensional (`p > n`) smoke test in float32 and float64. In that controlled smoke test, centring reduced the positive-eigenvalue condition estimate from approximately 861 to 9, demonstrating that the recorded diagnostics respond correctly to the proposed conditioning mechanism. This synthetic result is a pipeline validation, not evidence about UCR performance.

## What each intervention changes

| Intervention | Changed component | Unchanged components | Mechanism tested |
|---|---|---|---|
| Centred scaling | Subtracts training feature means before variance scaling | Raw transform, labels, split and downstream algorithm | Mean/intercept confounding and Gram conditioning |
| Float64 | Linear algebra, coefficients and score arithmetic use double precision | Scaling choice, transform and model equations | Loss of small eigendirections in float32 |
| Raw-score argmax | Selects class before calibration and clipping | Fitted ridge solution and selected lambda | Whether calibration/clipped softmax corrupts otherwise-correct labels |
| Positive-constrained scale | Optimises log-scale, guaranteeing `scale > 0` | Ridge solution, lambda grid and clipping | Negative or pathological unconstrained calibration |
| RidgeCV-selected lambda | Uses RidgeCV’s alpha on the identical scaled feature matrix | PreVal fit/calibration and prediction machinery | Lambda-selection criterion |
| Fixed lambda 1 | Removes data-dependent lambda selection | Everything else | General regularisation sensitivity |
| Combined repair | Centring + float64 + positive scale + raw-score labels | Transform and data | Whether the leading mechanisms jointly account for the failure |

## Evidence status

Existing UCR112 evidence already strongly supports clipped-softmax saturation as the **proximate** Thorax1 MiniRocket mechanism: all test rows have multiway maximum-probability ties and the production `_predict` selects from those clipped probabilities. The architecture pattern supports non-centring as the leading **root-cause** hypothesis.

The focused target experiment has not been run locally. The current editable aeon environment remained CPU-bound while importing the transform stack, and the Thorax data were not cached locally. Consequently there are not yet new target-case intervention results, selected lambdas, calibration scales or condition estimates to report. The Iridis array is prepared to obtain them without repeated transformations.

The result interpretation will be:

- Raw-score recovery with unchanged coefficients/lambda: direct confirmation of clipped-softmax label corruption.
- Centring recovery accompanied by lower condition estimate, smaller scale and fewer clipped logits: confirmation of scaling mismatch as root cause.
- Float64-only recovery: precision/eigendecomposition is primary.
- Positive-scale-only recovery: unconstrained calibration is primary.
- Ridge/fixed-lambda-only recovery: lambda selection is primary.
- No one-factor recovery but combined repair succeeds: interacting numerical and decision effects.
- RidgeCV reference mismatch against the original stored accuracy: stop and resolve transform/data/version reproducibility before interpreting any diagnostic contrast.

## Recommended execution order

Run the array as prepared. If cluster capacity is limited, execute tasks in this order: MiniRocket Thorax1, MiniRocket Thorax2, MultiRocket ArrowHead, MultiRocket HandOutlines, MultiRocket Thorax1/2, then EOGVerticalSignal and FaceFour. The first four are enough to determine whether clipping and centring generalise before paying for both large MultiRocket Thorax transforms.
