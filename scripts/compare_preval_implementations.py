"""Deterministic correctness checks for the standalone and aeon PreVal APIs."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_implementations(aeon_repo):
    """Load the aeon classifier and its local read-only reference module."""
    sys.path.insert(0, str(aeon_repo))
    from aeon.classification.sklearn import PreValClassifier

    reference_path = aeon_repo / "aeon" / "classification" / "sklearn" / "preval.py"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference implementation not found: {reference_path}")

    spec = importlib.util.spec_from_file_location("preval_reference", reference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reference implementation: {reference_path}")
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    return reference_module.PreVal, PreValClassifier


def _make_case(name, seed, n, p, n_classes, labels, lambdas, feature_kind="random"):
    """Create one reproducible classification problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)

    if feature_kind == "constant_duplicate_near_constant":
        X[:, 0] = 2.5
        X[:, 1] = X[:, 2]
        X[:, 3] = 0.25 + np.linspace(-2e-6, 2e-6, n, dtype=np.float32)
    elif feature_kind == "correlated":
        latent = rng.normal(size=(n, 2)).astype(np.float32)
        weights = rng.normal(size=(2, p)).astype(np.float32)
        X = latent @ weights + np.float32(2e-3) * X
    elif feature_kind == "ill_conditioned":
        base = rng.normal(size=n).astype(np.float32)
        X[:, 0] = base
        for column in range(1, p):
            X[:, column] = (
                np.float32(1 + column * 1e-4) * base
                + np.float32(2e-5) * X[:, column]
            )

    # Balanced quantile labels guarantee every class is represented and give
    # the optimizer a stable, non-degenerate signal for every seeded dataset.
    score = X @ rng.normal(size=p).astype(np.float32)
    order = np.argsort(score)
    y = np.empty(n, dtype=np.int64)
    y[order] = np.arange(n, dtype=np.int64) * n_classes // n
    if labels == "string":
        y = np.asarray([f"class_{value}" for value in y])

    return {
        "name": name,
        "seed": seed,
        "X": X,
        "y": y,
        "lambdas": np.asarray(lambdas, dtype=np.float32),
    }


def _cases():
    """Return cases spanning shapes, labels, grids, and feature pathologies."""
    log_grid = np.logspace(-2, 2, 5, dtype=np.float32)
    narrow_grid = np.array([0.03, 0.2, 1.0, 7.0], dtype=np.float32)
    wide_grid = np.logspace(-4, 4, 7, dtype=np.float32)
    uneven_grid = np.array([0.005, 0.07, 0.9, 2.5, 30.0], dtype=np.float32)

    return [
        _make_case("seed_0_binary_small_n_ge_p_int", 0, 18, 4, 2, "integer", log_grid),
        _make_case("seed_1_multiclass_n_ge_p_string", 1, 36, 7, 3, "string", narrow_grid),
        _make_case("seed_2_binary_n_lt_p_string", 2, 12, 24, 2, "string", wide_grid),
        _make_case("seed_7_multiclass_n_lt_p_int", 7, 15, 30, 3, "integer", uneven_grid),
        _make_case("seed_19_binary_medium_int", 19, 64, 9, 2, "integer", narrow_grid),
        _make_case("seed_31_multiclass_medium_string", 31, 75, 12, 5, "string", log_grid),
        _make_case(
            "constant_duplicate_near_constant_n_ge_p",
            43,
            40,
            8,
            2,
            "string",
            uneven_grid,
            "constant_duplicate_near_constant",
        ),
        _make_case(
            "constant_duplicate_near_constant_n_lt_p",
            47,
            14,
            28,
            3,
            "integer",
            log_grid,
            "constant_duplicate_near_constant",
        ),
        _make_case(
            "highly_correlated_binary",
            59,
            52,
            10,
            2,
            "integer",
            wide_grid,
            "correlated",
        ),
        _make_case(
            "highly_correlated_multiclass_n_lt_p",
            61,
            16,
            32,
            4,
            "string",
            narrow_grid,
            "correlated",
        ),
        _make_case(
            "mildly_ill_conditioned_binary",
            71,
            48,
            6,
            2,
            "string",
            uneven_grid,
            "ill_conditioned",
        ),
        _make_case(
            "mildly_ill_conditioned_multiclass",
            89,
            54,
            11,
            3,
            "integer",
            wide_grid,
            "ill_conditioned",
        ),
    ]


def _difference_details(actual, expected):
    """Describe an exact-comparison failure without hiding float differences."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    details = f"aeon={actual!r}, reference={expected!r}"
    if actual.shape == expected.shape and np.issubdtype(actual.dtype, np.number):
        if actual.size:
            max_difference = np.max(np.abs(actual.astype(float) - expected.astype(float)))
            details += f", max_abs_difference={max_difference:.17g}"
    return details


def _check_exact(name, actual, expected):
    """Require exact equality and return a diagnostic on failure."""
    try:
        np.testing.assert_equal(actual, expected)
    except AssertionError:
        return f"{name}: {_difference_details(actual, expected)}"
    return None


def _run_case(case, reference_class, port_class):
    """Fit and compare both implementations for one case."""
    X, y, lambdas = case["X"], case["y"], case["lambdas"]
    reference = reference_class(lambdas=lambdas.copy())
    port = port_class(lambdas=lambdas.copy())

    # Separate copies ensure neither implementation can affect the other's input.
    reference.fit(X.copy(), y.copy())
    port.fit(X.copy(), y.copy())

    checks = (
        ("selected lambda", port.lambda_, reference.lambda_),
        ("scale factor", port.scale_, reference.c),
        ("predictions", port.predict(X.copy()), reference.predict(X.copy())),
        (
            "predicted probabilities",
            port.predict_proba(X.copy()),
            reference.predict_proba(X.copy()),
        ),
        ("coefficients", port.coef_, reference.B),
        ("intercept", port.intercept_, reference.B0),
    )
    return [failure for check in checks if (failure := _check_exact(*check))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aeon-repo",
        type=Path,
        required=True,
        help="Path to the aeon checkout containing both implementations.",
    )
    args = parser.parse_args()

    reference_class, port_class = _load_implementations(args.aeon_repo.resolve())
    cases = _cases()
    failed_cases = []

    for case in cases:
        try:
            failures = _run_case(case, reference_class, port_class)
        except Exception as error:  # Keep the case identity visible for fit failures.
            failures = [f"fit or prediction raised {type(error).__name__}: {error}"]

        if failures:
            failed_cases.append(case["name"])
            print(f"FAIL {case['name']} (seed={case['seed']})")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print(
                f"PASS {case['name']} (seed={case['seed']}, "
                f"n={case['X'].shape[0]}, p={case['X'].shape[1]}): "
                "all outputs match exactly"
            )

    passed = len(cases) - len(failed_cases)
    print(f"\nSUMMARY: {passed}/{len(cases)} cases passed exactly.")
    if failed_cases:
        print(f"Failed cases: {', '.join(failed_cases)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
