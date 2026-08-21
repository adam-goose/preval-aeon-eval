# Minimal diagnostic design

## Identification logic

The expensive time-series transform is fit once for each dataset–architecture pair and its raw float32 train/test feature matrices are cached as `.npy` files. Every classifier intervention uses those exact matrices, labels and split. This removes transform randomness and avoids recomputing MiniRocket/MultiRocket.

The staged minimal design contains the production-equivalent PreVal configuration, six one-factor interventions, and a combined repair:

| Intervention | What is held fixed | Distinguishes |
|---|---|---|
| Raw-score rather than clipped-softmax argmax | Matrix, lambda, scale and coefficients | Whether clipping/calibration directly changes labels |
| Centre features | Transform, dtype, lambda rule, calibration and decision | Non-centred intercept/conditioning mismatch |
| Float64 | Scaling and model rules | Precision/eigendecomposition instability |
| Positive-constrained scale | Matrix, lambda criterion and decision | Negative/extreme unconstrained calibration |
| RidgeCV-selected lambda | Matrix, calibration and decision | Lambda-selection criterion |
| Fixed lambda 1 | Matrix, calibration and decision | Sensitivity to selected regularisation |
| Centred + float64 + positive scale + raw score | Transform only | Whether the combined numerical/decision repair recovers discrimination |

The full design evaluates all `2 × 2 × 2 × 2 × 3 = 48` PreVal combinations, but it reuses only four decompositions per cached feature matrix. Two RidgeCV references—production uncentred scaling and centred scaling—are also included.

## Cases

MiniRocket uses Thorax1 as the directly observed clipped-tie failure and Thorax2 as a matched 42-class non-collapse control. MultiRocket uses both Thorax datasets, plus ArrowHead (small three-class failure), HandOutlines (binary failure), EOGVerticalSignal (12-class broader failure), and FaceFour (positive-effect control). This spans class count and sample size without initially transforming the entire archive.

## Recorded diagnostics

The runner records selected lambda, scale, calibrated LOOCV loss for every lambda, optimiser success/message, coefficient norm, raw Gram eigenvalue range and positive-eigenvalue condition estimate, non-positive eigenvalue count, leverage range, feature mean magnitude, upper/lower logit clipping fractions, tied-maximum frequency, prediction concentration, accuracy and log loss. Optional compressed prediction arrays allow exact confusion reconstruction.

## Interpretation order

1. Compare current clipped labels to raw-score labels. A recovery with the same fitted ridge solution proves clipping/calibration is the proximate label failure.
2. Compare centred to uncentred features. Recovery plus reduced condition estimate, scale and clipping supports the pipeline mismatch as root cause.
3. Compare float64 to float32 within each scaling condition. Recovery only in float64 supports numerical precision.
4. Compare positive to unconstrained calibration. This tests optimiser sign/extremity independently of ridge fitting.
5. Compare lambda sources. A recovery isolated to RidgeCV/fixed lambda implicates prevalidated lambda selection rather than calibration or conditioning.

The held-out 30 datasets are not included.
