#### Reference Issues/PRs

Closes #3419.

#### What does this implement/fix? Explain your changes.

This PR adds `PrevalidatedRidgeClassifier` to
`aeon.classification.sklearn`. The classifier is based on Dempster, Webb and Schmidt,
*Prevalidated Ridge Regression is a Highly-Efficient Drop-In Replacement for Logistic Regression for High-Dimensional Data*
([arXiv:2401.15610](https://arxiv.org/abs/2401.15610)), and adapts the authors' [reference implementation](https://github.com/angus924/preval) to aeon's estimator interface.

- supports binary and multiclass targets, including non-numeric labels;
- selects a ridge penalty from a configurable positive lambda grid using
  efficient leave-one-out predictions;
- calibrates the ridge predictions with a fitted scale factor and returns class
  probabilities through `predict_proba`;
- uses the smaller of the feature-space and case-space eigendecompositions for
  `n >= p` and `n < p` respectively; and
- removes constant and very low-variance feature columns before fitting.

The classifier exposes the usual aeon fitted state and prediction interface,
including learned classes, selected lambda, scale, coefficients, intercept,
and probability estimates. It has also been added to the generated
classification API reference.

##### Verification

The implementation was checked directly against the standalone reference using
identical copied inputs and lambda grids. Exact equality was obtained for the
selected lambda, scale, coefficients, intercept, predicted probabilities, and
predicted labels in all cases:

- **12 deterministic synthetic cases**, covering multiple seeds, binary and
  multiclass targets, integer and string labels, `n >= p` and `n < p`, varied
  dataset dimensions and lambda grids, constant/duplicated/near-constant
  columns, and correlated or mildly ill-conditioned features;
- **6 bundled real datasets** using official train/test splits: GunPoint,
  ItalyPowerDemand, ArrowHead, BasicMotions, OSULeaf, and ACSF1.
  These cover binary through 10-class problems, 24 to 1,460 transformed
  features, and both the n >= p and n < p computation paths.

The aeon unit tests cover binary and multiclass estimator lifecycles, string and
integer labels, both shape branches, low-variance feature removal, fitted
attribute dimensions, probability validity, prediction/probability agreement,
and invalid lambda grids. All **7 focused unit tests** pass, as do aeon's
centralized estimator checks.

A repeated fit-time comparison used warm-up runs, alternating execution order,
and seven measured repetitions on four synthetic and the six real datasets.
The median aeon/reference runtime ratio across the ten cases was **1.064**
(**+6.4%**). The largest median slowdown was **1.084x (+8.4%)**. This indicates
a small, consistent interface overhead with no substantial computational
regression relative to the reference implementation in the tested cases.

#### Does your contribution introduce a new dependency? If yes, which one?

No. The implementation uses `scipy.optimize.minimize` and `sklearn.preprocessing.LabelBinarizer`;
SciPy and scikit-learn are  lready core aeon dependencies.

#### Any other comments?

The detailed correctness verification and runtime scripts are maintained separately from the aeon PR
so that the upstream contribution remains focused on the estimator, its unit tests, and API documentation.

### PR checklist

##### For all contributions
- [x] I've added myself to the [list of contributors](https://github.com/aeon-toolkit/aeon/blob/main/.all-contributorsrc). Alternatively, you can use the [@all-contributors](https://allcontributors.org/docs/en/bot/usage) bot to do this for you **after** the PR has been merged.
- [x] The PR title starts with either [ENH], [MNT], [DOC], [BUG], [REF], [DEP] or [GOV] indicating whether the PR topic is related to enhancement, maintenance, documentation, bugs, refactoring, deprecation or governance.

##### For new estimators and functions
- [x] I've added the estimator/function to the online [API documentation](https://www.aeon-toolkit.org/en/latest/api_reference.html).
- [ ] (OPTIONAL) I've added myself as a `__maintainer__` at the top of relevant files and want to be contacted regarding its maintenance. Unmaintained files may be removed. This is for the full file, and you should not add yourself if you are just making minor changes or do not want to help maintain its contents.

##### For developers with write access
- [ ] (OPTIONAL) I've updated aeon's [CODEOWNERS](https://github.com/aeon-toolkit/aeon/blob/main/CODEOWNERS) to receive notifications about future changes to these files.
