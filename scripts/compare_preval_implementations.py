"""Brief parity checks between the original PreVal and the aeon port."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_implementations(aeon_repo):
    """Load the installed aeon port and the local read-only reference module."""
    sys.path.insert(0, str(aeon_repo))
    from aeon.classification.sklearn import PreValClassifier

    reference_path = (
        aeon_repo / "aeon" / "classification" / "sklearn" / "preval.py"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference implementation not found: {reference_path}")

    spec = importlib.util.spec_from_file_location("preval_reference", reference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reference implementation: {reference_path}")
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    return reference_module.PreVal, PreValClassifier


def _cases():
    return [
        (
            "binary_n_ge_p_constant_col",
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.2, 0.1, 1.0],
                    [0.9, 0.8, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(["a", "a", "b", "b", "a", "b"]),
        ),
        (
            "multiclass_n_ge_p",
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(["c0", "c1", "c2", "c0", "c1", "c2"]),
        ),
        (
            "binary_n_lt_p",
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.5, 1.0],
                    [1.0, 0.0, 1.0, 0.3, 1.0],
                    [0.2, 0.8, 0.2, 0.7, 1.0],
                    [0.8, 0.2, 0.9, 0.4, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array(["x", "y", "x", "y"]),
        ),
    ]


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
    lambdas = np.logspace(-2, 2, 5).astype(np.float32)

    for name, X, y in _cases():
        reference = reference_class(lambdas=lambdas.copy())
        port = port_class(lambdas=lambdas.copy())
        reference.fit(X.copy(), y.copy())
        port.fit(X.copy(), y.copy())

        reference_predictions = reference.predict(X.copy())
        port_predictions = port.predict(X.copy())
        reference_probabilities = reference.predict_proba(X.copy())
        port_probabilities = port.predict_proba(X.copy())

        np.testing.assert_equal(port.lambda_, reference.lambda_)
        np.testing.assert_equal(port.scale_, reference.c)
        np.testing.assert_array_equal(port_predictions, reference_predictions)
        np.testing.assert_array_equal(port_probabilities, reference_probabilities)
        np.testing.assert_array_equal(port.coef_, reference.B)
        np.testing.assert_array_equal(port.intercept_, reference.B0)

        print(
            f"PASS {name}: lambda={float(port.lambda_):.8g}, "
            f"predictions/probabilities/coefficients/intercept match exactly"
        )

    print("All brief parity checks passed.")


if __name__ == "__main__":
    main()
