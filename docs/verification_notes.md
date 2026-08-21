# PreVal implementation verification notes

## Overview

The aeon `PrevalidatedRidgeClassifier` was verified by direct parity testing
against the original standalone `PreVal` implementation. In every test, both implementations
were fitted using separate copies of the same input arrays and the same lambda
grid. Their fitted parameters and prediction outputs were then compared using
exact equality.

Verification was split into two complementary scripts:

- `scripts/compare_preval_implementations.py` uses deterministic synthetic data
  to exercise specific numerical branches, label types, shapes, and feature edge
  cases in a controlled way.
- `scripts/compare_preval_real_datasets.py` uses bundled aeon datasets to confirm
  that parity also holds on genuine time-series classification data after a
  simple deterministic tabular transformation.

The synthetic tests isolate behaviours that may be uncommon in a small sample of
real datasets. The real-dataset tests complement them by checking realistic data
distributions, class structures, and feature values on official train/test
splits.

## Synthetic parity cases

All synthetic cases use NumPy's seeded random-number generator and `float32`
features. Class labels are assigned from quantiles of a deterministic linear
score, ensuring that every class is represented and that the optimisation has a
stable signal. Several valid lambda grids are used, including logarithmic,
narrow, wide, and unevenly spaced grids.

1. **`seed_0_binary_small_n_ge_p_int`** — 18 cases and 4 features, with two
   integer-labelled classes. This is a small baseline binary problem exercising
   the `n >= p` eigendecomposition branch.
2. **`seed_1_multiclass_n_ge_p_string`** — 36 cases, 7 features, and three string
   classes. This checks multiclass encoding, string-label recovery, and the
   `n >= p` branch with a different seed and lambda grid.
3. **`seed_2_binary_n_lt_p_string`** — 12 cases and 24 features, with binary
   string labels. This checks binary classification and label recovery when the
   feature count exceeds the number of cases.
4. **`seed_7_multiclass_n_lt_p_int`** — 15 cases, 30 features, and three integer
   classes. This checks multiclass behaviour in the `n < p` branch using an
   uneven lambda grid.
5. **`seed_19_binary_medium_int`** — 64 cases and 9 features, with two integer
   classes. This adds a medium-sized binary problem and another random seed.
6. **`seed_31_multiclass_medium_string`** — 75 cases, 12 features, and five
   string classes. This broadens the multiclass and dataset-size coverage while
   checking a larger string label set.
7. **`constant_duplicate_near_constant_n_ge_p`** — 40 cases and 8 features,
   including one constant column, one duplicated column, and one near-constant
   column. This checks low-variance feature removal, redundant features, and the
   `n >= p` numerical path.
8. **`constant_duplicate_near_constant_n_lt_p`** — 14 cases and 28 features with
   the same three feature pathologies and three integer classes. This verifies
   their handling in the alternative `n < p` path.
9. **`highly_correlated_binary`** — 52 cases and 10 features generated mainly
   from two latent variables with small independent noise. This checks binary
   parity with strongly correlated features and a wide lambda grid.
10. **`highly_correlated_multiclass_n_lt_p`** — 16 cases, 32 highly correlated
    features, and four string classes. This combines multicollinearity,
    multiclass encoding, and the `n < p` branch.
11. **`mildly_ill_conditioned_binary`** — 48 cases and 6 almost-collinear
    features, with two string classes. This directly checks behaviour on a
    mildly ill-conditioned binary design matrix.
12. **`mildly_ill_conditioned_multiclass`** — 54 cases, 11 almost-collinear
    features, and three integer classes. This repeats the ill-conditioning check
    for multiclass output and a larger feature matrix.

## Real-dataset parity cases

Six small datasets bundled with aeon were used, so the script requires no dataset
download. Each equal-length time-series array was flattened from `(cases,
channels, time points)` to a two-dimensional `float32` matrix with one row per
case. Both implementations were fitted on identical copies of the official
training split, and predictions and probabilities were compared on the official
test split.

- **GunPoint** — binary; 50 training cases and 150 transformed features. It was
  included as a well-known small UCR problem and exercises `n < p`.
- **ItalyPowerDemand** — binary; 67 training cases and 24 transformed features.
  It provides the real-data `n >= p` case and a shorter series representation.
- **ArrowHead** — three classes; 36 training cases and 251 transformed features.
  It adds a small multiclass UCR problem with substantially more features than
  cases.
- **BasicMotions** — four classes; 40 training cases and 600 transformed
  features. Its multivariate series produce a higher-dimensional tabular input.
- **OSULeaf** — six classes; 200 training cases and 427 transformed features. It
  adds more cases, more classes, and a moderately large feature representation.
- **ACSF1** — ten classes; 100 training cases and 1,460 transformed features. It
  provides the largest class count and transformed feature count in the suite.

## Compared outputs

For every synthetic and real-data case, the following were compared between the
standalone reference and the aeon classifier:

- the selected ridge penalty (`lambda_`);
- the fitted scale factor;
- predicted class labels;
- predicted class probabilities;
- the complete fitted coefficient array; and
- the fitted intercept array.

Exact equality was required for every comparison. This was appropriate because
the port is intended to reproduce the same `float32` algorithm and both models
receive identical copied inputs and lambda grids. Matching the selected lambda
and scale verifies the model-selection and calibration stages. Matching
coefficients and intercept verifies the fitted model itself. Matching
probabilities checks the full numerical prediction path, while matching labels
also confirms consistent class encoding, decoding, and decision behaviour.

## Results and conclusion

All 12 synthetic parity cases passed with exact equality for every compared
output. All 6 real-dataset parity cases also passed with exact equality. No
mismatches were observed, and no tolerance-based comparison was required.

The aeon implementation can therefore be considered verified against the
standalone reference implementation over the behaviours covered by these suites.
It is ready to be frozen before experimental evaluation against baseline
classifiers. Further algorithm changes should be limited to corrections for a
genuine identified bug.
