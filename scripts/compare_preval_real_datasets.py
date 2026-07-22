"""Compare standalone and aeon PreVal on small, bundled real datasets."""

import argparse
from pathlib import Path

import numpy as np

from compare_preval_implementations import _load_implementations


DATASETS = (
    "GunPoint",
    "ItalyPowerDemand",
    "ArrowHead",
    "BasicMotions",
    "OSULeaf",
    "ACSF1",
)
LAMBDAS = np.logspace(-2, 2, 5, dtype=np.float32)


def _as_tabular(X):
    """Flatten equal-length time series from (cases, channels, time) to 2D."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected equal-length 3D data, found shape {X.shape}")
    return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)


def _failure(name, actual, expected):
    """Return a concise exact-equality diagnostic, or None when equal."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if np.array_equal(actual, expected):
        return None

    detail = f"{name}: aeon shape={actual.shape}, reference shape={expected.shape}"
    if actual.shape == expected.shape and np.issubdtype(actual.dtype, np.number):
        difference = np.abs(actual.astype(float) - expected.astype(float))
        index = np.unravel_index(np.argmax(difference), difference.shape)
        detail += (
            f", max_abs_difference={difference[index]:.17g} at {index}, "
            f"aeon={actual[index]!r}, reference={expected[index]!r}"
        )
    else:
        detail += f", aeon={actual!r}, reference={expected!r}"
    return detail


def _run_dataset(name, load_classification, reference_class, port_class):
    """Fit on the official train split and compare outputs on the test split."""
    X_train, y_train = load_classification(name, split="train")
    X_test, _ = load_classification(name, split="test")
    X_train = _as_tabular(X_train)
    X_test = _as_tabular(X_test)

    reference = reference_class(lambdas=LAMBDAS.copy())
    port = port_class(lambdas=LAMBDAS.copy())
    reference.fit(X_train.copy(), y_train.copy())
    port.fit(X_train.copy(), y_train.copy())

    checks = (
        ("selected lambda", port.lambda_, reference.lambda_),
        ("scale factor", port.scale_, reference.c),
        ("predictions", port.predict(X_test.copy()), reference.predict(X_test.copy())),
        (
            "predicted probabilities",
            port.predict_proba(X_test.copy()),
            reference.predict_proba(X_test.copy()),
        ),
        ("coefficients", port.coef_, reference.B),
        ("intercept", port.intercept_, reference.B0),
    )
    failures = [failure for check in checks if (failure := _failure(*check))]
    return X_train.shape, len(np.unique(y_train)), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aeon-repo",
        type=Path,
        required=True,
        help="Path to the local aeon checkout.",
    )
    args = parser.parse_args()
    aeon_repo = args.aeon_repo.resolve()

    reference_class, port_class = _load_implementations(aeon_repo)
    from aeon.datasets import load_classification

    failed = []
    for name in DATASETS:
        try:
            shape, n_classes, failures = _run_dataset(
                name, load_classification, reference_class, port_class
            )
        except Exception as error:
            shape, n_classes = ("?", "?"), "?"
            failures = [f"raised {type(error).__name__}: {error}"]

        if failures:
            failed.append(name)
            print(f"FAIL {name} (n={shape[0]}, p={shape[1]}, classes={n_classes})")
            for failure in failures:
                print(f"  - {failure}")
        else:
            branch = "n >= p" if shape[0] >= shape[1] else "n < p"
            print(
                f"PASS {name} (n={shape[0]}, p={shape[1]}, classes={n_classes}, "
                f"{branch}): all outputs match exactly"
            )

    passed = len(DATASETS) - len(failed)
    print(f"\nSUMMARY: {passed}/{len(DATASETS)} real datasets passed exactly.")
    if failed:
        print(f"Failed datasets: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
