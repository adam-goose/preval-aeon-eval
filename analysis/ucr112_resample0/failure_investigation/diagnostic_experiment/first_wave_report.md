# First-wave diagnostic results

## Scope and validity

The first wave completed all 50 configurations and all 80 lambda-path rows for each of four cases: MiniRocket on `NonInvasiveFetalECGThorax1/2`, and MultiRocket on `ArrowHead` and `HandOutlines`. No configuration failed. The RidgeClassifierCV references reproduce the stored benchmark accuracies (exactly at the reported precision for the four reference cases), so the intervention contrasts are directly interpretable.

This is a mechanistic diagnostic on the 112-dataset development set. It does not justify selecting a production change without a broader confirmation wave.

## Headline results

Accuracy for the production-equivalent diagnostic configuration and the principal interventions is:

| Transform / dataset | Current: uncentred float32 + clipped softmax | Same fit, raw-score argmax | Centred float32 | Uncentred float64 | RidgeCV reference |
|---|---:|---:|---:|---:|---:|
| MiniRocket / Thorax1 | 0.156 | 0.513 | 0.948 | 0.949 | 0.949 |
| MiniRocket / Thorax2 | 0.949 | 0.949 | 0.965 | 0.965 | 0.964 |
| MultiRocket / ArrowHead | 0.303 | 0.303 | 0.869 | 0.869 | 0.869 |
| MultiRocket / HandOutlines | 0.641 | 0.641 | 0.943 | 0.943 | 0.946 |

Two independent one-factor interventions recover near-reference accuracy in every case: (1) centre before performing the float32 solve, or (2) retain uncentred scaling but perform the solve in float64. Their equivalence is the most important result of this wave.

## Mechanism 1: uncentred float32 linear algebra is the general failure

The uncentred transforms have extremely large feature means and Gram spectra. In float32, the computed Gram matrices consequently contain negative eigenvalues even though a Gram matrix is positive semidefinite. These are numerical artefacts, not properties of the data.

| Transform / dataset | Train × retained features | Uncentred float32 non-positive eigenvalues | Uncentred float32 raw minimum eigenvalue | Uncentred float64 raw minimum | Centred float32 raw minimum | Condition estimate: uncentred float32 → centred float32 |
|---|---:|---:|---:|---:|---:|---:|
| MiniRocket / Thorax1 | 1,800 × 9,917 | 792 | -33,398 | 95.2 | 95.2 | 7.23e11 → 5.95e4 |
| MiniRocket / Thorax2 | 1,800 × 9,928 | 547 | -1,512 | 64.2 | 64.2 | 1.12e12 → 8.98e4 |
| MultiRocket / ArrowHead | 36 × 79,155–79,157 | 12 | -200,736 | 14,644 | 36.0 | 9.77e8 → 3.63e4 |
| MultiRocket / HandOutlines | 1,000 × 79,683 | 489 | -16,165,060 | 4,743 | 1,001 | 5.72e11 → 2.55e4 |

The prediction evidence agrees with the spectral evidence. For the production-equivalent uncentred float32 fit, raw prediction agreement with RidgeCV is only 52.8% (Thorax1), 97.6% (Thorax2), 32.0% (ArrowHead), and 66.2% (HandOutlines). With uncentred float64 it rises to 99.9%, 99.9%, 100%, and 100%; centred float32 gives 100%, 100%, 100%, and 99.7%.

Therefore non-centring alone is not intrinsically incompatible with ridge classification: sklearn's float64 RidgeCV references are accurate with either scaling choice. The failure is specifically the interaction of non-centring, very large/high-dimensional transformed features, and the diagnostic PreVal float32 eigensolve. Centring makes float32 numerically safe by removing the large mean component; float64 preserves enough precision to solve the uncentred problem.

Lambda selection changes dramatically when the solve is corrupted, but it is mainly a symptom rather than the root cause. The current PreVal configuration selects 1000 in all four cases. Substituting RidgeCV's lambda without repairing the float32 solve does not recover Thorax1 or HandOutlines and only partly improves ArrowHead. Conversely, centred float32 and uncentred float64 recover accuracy even when their selected lambdas differ. Fixed-lambda rows show the same broad precision/centring split.

## Mechanism 2: clipped-softmax argmax is a secondary Thorax1 failure

For Thorax1's current fit, clipped probabilities have a tied maximum on 99.7% of test rows. Changing only the decision rule from clipped-probability argmax to raw-score argmax raises accuracy from 0.156 to 0.513. This directly confirms that clipping and subsequent tie-breaking corrupt many labels.

However, raw-score prediction does not restore the RidgeCV solution: it agrees with RidgeCV on only 52.8% of Thorax1 rows and remains far below RidgeCV accuracy (0.513 versus 0.949). Thus the ridge scores themselves are already badly damaged by the float32 uncentred solve; clipped-softmax then adds a second failure on top.

The other cases demonstrate why raw-score prediction is not the general explanation:

- Thorax2 has no material maximum ties at the current selected lambda, and raw-score and clipped-softmax accuracies are identical (0.949).
- ArrowHead and HandOutlines also have no maximum ties in the current configuration; raw-score argmax changes neither accuracy nor predictions materially.
- Once centring or float64 repairs the score calculation, raw and clipped-softmax prediction are almost identical. Small residual differences occur where clipping creates ties, but they are not catastrophic.

Raw-score argmax is therefore logically preferable for class labels because a monotone calibration/softmax should not change their ordering, while clipping can. But this wave shows it is a guard against secondary corruption, not a sufficient repair for the main numerical problem.

## Mechanisms not supported

- **Negative calibration scale:** all headline scales are positive. Positive-constrained calibration does not improve the current configuration in any of the four cases and slightly worsens Thorax2 because it selects a different lambda. There is no first-wave evidence that a negative scale causes these failures.
- **Calibration alone:** MultiRocket collapses occur without probability ties, and raw-score prediction remains collapsed. Calibration is not the shared cause.
- **A single bad lambda:** RidgeCV-selected and fixed common lambdas do not consistently repair uncentred float32. Stable arithmetic repairs performance across several lambdas.
- **Centring as a statistical-model requirement:** uncentred float64 matches RidgeCV. Centring is effective here primarily as numerical preconditioning, although it may also improve intercept handling.

## Remaining uncertainty and second wave

The evidence is decisive for these four cases but does not yet establish prevalence across the full MultiRocket degradation pattern. In particular:

1. The two MultiRocket examples span tiny (`n=36`) and moderate (`n=1000`) training sets, but neither is a 42-class Thorax MultiRocket case.
2. We have not tested whether less severely degraded or unaffected MultiRocket datasets show the same spectral pathology without a large accuracy effect.
3. Centring and float64 both repair all four cases, so this wave cannot choose between them as the eventual production strategy; that choice also depends on memory, speed, intercept semantics, and broad accuracy stability.
4. The diagnostic uses one resample and the development collection. The result identifies a deterministic numerical mechanism but does not estimate the frequency or performance impact of a repair across datasets/resamples.

The second wave is warranted, but it can remain focused. MultiRocket Thorax1 and Thorax2 are the highest-value cases because they combine the extreme class structure with the architecture-wide degradation. EOGVerticalSignal and FaceFour are useful controls to test whether the same numerical signatures track the severity of degradation. The existing second-wave cases should be run unchanged before modifying the estimator. A successful confirmation would be: uncentred float32 has spurious non-positive eigenvalues and poor RidgeCV agreement, while either centred float32 or uncentred float64 removes them and restores accuracy. If a second-wave failure persists under both repairs, another mechanism remains.

## Reproducible outputs

Run `summarise_results.py` against the copied result root to regenerate `summary/`. The most useful files are:

- `summary/headline_configuration_comparison.csv`: compact accuracies and fitted diagnostics for the decisive configurations.
- `summary/numerical_stability_summary.csv`: feature dimensions, Gram spectra, condition estimates, and decomposition time.
- `summary/prediction_agreement_with_ridgecv.csv`: exact class-prediction agreement with the matching RidgeCV reference.
- `summary/decision_rule_pairs.csv`: every raw-score versus clipped-softmax matched pair.
- `summary/intervention_contrasts.csv`: one-factor accuracy changes from the production-equivalent configuration.
- `summary/all_configuration_results.csv` and `summary/all_lambda_path_diagnostics.csv`: complete processed results.

