"""Fast synthetic test of the independent diagnostic core."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostic_core import (evaluate_configuration, evaluate_preval_path,
                             evaluate_ridge_reference, prepare_decomposition)


def main():
    rng = np.random.default_rng(20260821)
    # Deliberately p > n and non-centred to exercise the dual path.
    X_train = (rng.normal(size=(48, 160)) + 2.0).astype(np.float32)
    X_test = (rng.normal(size=(24, 160)) + 2.0).astype(np.float32)
    y_train = np.repeat(np.arange(4), 12); y_test = np.tile(np.arange(4), 6)
    lambdas = np.logspace(-3, 3, 10)
    output = {"cases": []}
    for centered in [False, True]:
        for dtype in ["float32", "float64"]:
            dec = prepare_decomposition(X_train, X_test, y_train, y_test, centered, dtype)
            path, models = evaluate_preval_path(dec, lambdas, ["unconstrained", "positive"])
            assert len(path) == 20 and dec.diagnostics["solve_space"] == "dual"
            best = min((x for x in path if x["calibration_constraint"] == "positive"), key=lambda x: x["nll"])
            metrics, _ = evaluate_configuration(dec, models[(best["lambda"], "positive")], "raw_score")
            output["cases"].append({**dec.diagnostics, "selected_lambda": best["lambda"], **metrics})
    ridge, _ = evaluate_ridge_reference(X_train, X_test, y_train, y_test, False, lambdas)
    output["ridge_reference"] = ridge
    target = Path(__file__).resolve().parent / "smoke_test_results.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Smoke test passed; wrote {target}")


if __name__ == "__main__":
    main()
