# PreVal runtime verification notes

## Objective

The runtime comparison was performed to check that porting the standalone
`PreVal` implementation into aeon's `PreValClassifier` did not introduce a
significant computational regression. This was a parity-oriented timing check,
not a general performance benchmark or comparison against other classifiers.

## Method

The benchmark used two complementary groups of data:

- four deterministic synthetic cases covering binary and multiclass problems,
  integer and string labels, and both `n >= p` and `n < p`;
- the six real datasets previously used for correctness verification, spanning
  2 to 10 classes and 24 to 1,460 transformed features.

For each case, both implementations were fitted on separate copies of identical
tabular inputs using the same lambda grid. Timing used `time.perf_counter` and
included warm-up runs followed by seven measured repetitions. Execution order
was alternated to reduce ordering bias. The median fit time was used for each
implementation so that occasional timing outliers had limited influence on the
comparison. For example, one original GunPoint measurement was much larger than
the remaining measurements, but did not materially affect its median.

## Results

Positive percentage differences mean that the aeon implementation was slower.

### Synthetic datasets

| Dataset/case | Shape/classes | Original (s) | Aeon (s) | Aeon/original | Difference |
|---|---:|---:|---:|---:|---:|
| `seed_0_binary_small_n_ge_p_int` | 18x4, 2 classes | 0.014615 | 0.015141 | 1.036 | +3.6% |
| `seed_1_multiclass_n_ge_p_string` | 36x7, 3 classes | 0.008870 | 0.009518 | 1.073 | +7.3% |
| `seed_2_binary_n_lt_p_string` | 12x24, 2 classes | 0.016516 | 0.016834 | 1.019 | +1.9% |
| `seed_7_multiclass_n_lt_p_int` | 15x30, 3 classes | 0.010116 | 0.010725 | 1.060 | +6.0% |

### Real datasets

| Dataset | Shape/classes | Original (s) | Aeon (s) | Aeon/original | Difference |
|---|---:|---:|---:|---:|---:|
| GunPoint | 50x150, 2 classes | 0.013932 | 0.015089 | 1.083 | +8.3% |
| ItalyPowerDemand | 67x24, 2 classes | 0.014976 | 0.015732 | 1.050 | +5.0% |
| ArrowHead | 36x251, 3 classes | 0.011683 | 0.012659 | 1.084 | +8.4% |
| BasicMotions | 40x600, 4 classes | 0.013415 | 0.014237 | 1.061 | +6.1% |
| OSULeaf | 200x427, 6 classes | 0.018899 | 0.020162 | 1.067 | +6.7% |
| ACSF1 | 100x1460, 10 classes | 0.015283 | 0.016570 | 1.084 | +8.4% |

Across all ten cases, the median aeon/original runtime ratio was **1.064** and
the median percentage difference was **+6.4%**. The largest measured slowdown
was ACSF1 at **1.084x (+8.4%)**. No median speed-up was observed; the smallest
slowdown was the synthetic binary `n < p` case at **1.019x (+1.9%)**.

## Interpretation and conclusion

The aeon implementation showed a small and relatively consistent runtime
overhead across the tested shapes, class counts, and datasets. An overall median
overhead of approximately 6%, with no per-case median slowdown greater than
8.4%, provides no evidence of a substantial computational regression relative
to the standalone reference implementation.

Together with the completed correctness parity checks, this runtime verification
supports freezing the implementation before experimental evaluation. This
conclusion is limited to the cases and local timing procedure described above.
