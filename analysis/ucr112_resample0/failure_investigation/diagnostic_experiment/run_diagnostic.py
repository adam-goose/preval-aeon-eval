"""Cache ROCKET features once and run independent diagnostic ridge interventions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from diagnostic_core import (evaluate_configuration, evaluate_preval_path,
                             evaluate_ridge_reference, prepare_decomposition)

HERE = Path(__file__).resolve().parent
LAMBDAS = np.logspace(-3, 3, 10)


def load_data(dataset, data_dir):
    from aeon.datasets import load_classification
    X_train, y_train = load_classification(dataset, split="train", extract_path=data_dir)
    X_test, y_test = load_classification(dataset, split="test", extract_path=data_dir)
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    if X_train.ndim == 2:
        X_train, X_test = X_train[:, None, :], X_test[:, None, :]
    enc = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    return X_train, X_test, enc.transform(y_train), enc.transform(y_test), enc.classes_


def cache_features(dataset, transform_name, data_dir, cache_dir, n_jobs):
    target = cache_dir / transform_name / dataset
    target.mkdir(parents=True, exist_ok=True)
    train_path, test_path, meta_path = target / "train.npy", target / "test.npy", target / "metadata.json"
    labels_path = target / "labels.npz"
    if all(p.exists() for p in [train_path, test_path, meta_path, labels_path]):
        labels = np.load(labels_path, allow_pickle=True)
        return np.load(train_path, mmap_mode="r"), np.load(test_path, mmap_mode="r"), labels["y_train"], labels["y_test"], json.loads(meta_path.read_text())
    X_train, X_test, y_train, y_test, classes = load_data(dataset, data_dir)
    if transform_name == "MiniRocket":
        from aeon.transformations.collection.convolution_based import MiniRocket
        transform = MiniRocket(n_kernels=10_000, max_dilations_per_kernel=32, n_jobs=n_jobs, random_state=0)
    elif transform_name == "MultiRocket":
        from aeon.transformations.collection.convolution_based import MultiRocket
        transform = MultiRocket(n_kernels=10_000, max_dilations_per_kernel=32,
                                n_features_per_kernel=4, n_jobs=n_jobs, random_state=0)
    else:
        raise ValueError(transform_name)
    started = perf_counter(); transform.fit(X_train, y_train)
    raw_train = np.asarray(transform.transform(X_train), dtype=np.float32)
    raw_test = np.asarray(transform.transform(X_test), dtype=np.float32)
    np.save(train_path, raw_train); np.save(test_path, raw_test)
    np.savez(labels_path, y_train=y_train, y_test=y_test, classes=classes)
    meta = {"dataset": dataset, "transform": transform_name, "random_state": 0,
            "n_train": len(y_train), "n_test": len(y_test), "n_classes": int(np.unique(y_train).size),
            "n_features": raw_train.shape[1], "series_shape_train": list(X_train.shape),
            "transform_and_cache_seconds": perf_counter() - started}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return np.load(train_path, mmap_mode="r"), np.load(test_path, mmap_mode="r"), y_train, y_test, meta


def choose_design(name):
    if name == "full":
        return None
    # Current, one-factor interventions, and a combined repair. Lambda source is added later.
    return {
        (False, "float32", "unconstrained", "preval", "clipped_softmax"),
        (False, "float32", "unconstrained", "preval", "raw_score"),
        (True, "float32", "unconstrained", "preval", "clipped_softmax"),
        (False, "float64", "unconstrained", "preval", "clipped_softmax"),
        (False, "float32", "positive", "preval", "clipped_softmax"),
        (False, "float32", "unconstrained", "ridge", "clipped_softmax"),
        (False, "float32", "unconstrained", "fixed", "clipped_softmax"),
        (True, "float64", "positive", "preval", "raw_score"),
        (True, "float64", "positive", "preval", "clipped_softmax"),
    }


def run_case(args):
    out = Path(args.output_dir) / args.transform / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test, y_train, y_test, meta = cache_features(
        args.dataset, args.transform, args.data_dir, Path(args.cache_dir), args.n_jobs)
    results, paths, predictions = [], [], {}
    # Ridge reference exactly matches production centring choice, plus centred comparator.
    ridge_selected = {}
    for centered in [False, True]:
        r, pred = evaluate_ridge_reference(raw_train, raw_test, y_train, y_test, centered, LAMBDAS)
        ridge_selected[centered] = r["selected_lambda"]
        results.append({"model": "RidgeClassifierCV", "centered": centered, "dtype": "float64/sklearn",
                        "calibration_constraint": "none", "lambda_source": "ridge_cv",
                        "selected_lambda": r.pop("selected_lambda"), "decision_rule": "raw_score", **r})
        predictions[f"RidgeCV_centered={centered}"] = pred

    minimal = choose_design(args.design)
    for centered in [False, True]:
        for dtype in ["float32", "float64"]:
            if minimal is not None and not any(x[0] == centered and x[1] == dtype for x in minimal):
                continue
            dec = prepare_decomposition(raw_train, raw_test, y_train, y_test, centered, dtype)
            per_lambda, models = evaluate_preval_path(dec, LAMBDAS, ["unconstrained", "positive"])
            for x in per_lambda:
                paths.append({"centered": centered, "dtype": dtype, **dec.diagnostics, **x})
            for constraint in ["unconstrained", "positive"]:
                subset = [x for x in per_lambda if x["calibration_constraint"] == constraint]
                preval_lambda = min(subset, key=lambda x: x["nll"])["lambda"]
                selected = {"preval": preval_lambda, "ridge": ridge_selected[centered], "fixed": args.fixed_lambda}
                for lambda_source, lam in selected.items():
                    available = np.array([k[0] for k in models if k[1] == constraint])
                    actual_lam = float(available[np.argmin(np.abs(available - float(lam)))])
                    model = models[(actual_lam, constraint)]
                    for decision in ["raw_score", "clipped_softmax"]:
                        key = (centered, dtype, constraint, lambda_source, decision)
                        if minimal is not None and key not in minimal:
                            continue
                        metrics, pred = evaluate_configuration(dec, model, decision)
                        row = {"model": "DiagnosticPreVal", "centered": centered, "dtype": dtype,
                               "calibration_constraint": constraint, "lambda_source": lambda_source,
                               "selected_lambda": actual_lam, "decision_rule": decision, **metrics,
                               **{k: v for k, v in model.items() if k not in {"raw_scores", "scaled_scores", "probabilities"}}}
                        results.append(row); predictions["|".join(map(str, key))] = pred
            pd.DataFrame(paths).to_csv(out / "lambda_path_diagnostics.csv", index=False)
    pd.DataFrame(results).to_csv(out / "configuration_results.csv", index=False)
    (out / "case_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.save_predictions:
        np.savez_compressed(out / "predictions.npz", y_test=y_test, **predictions)
    (out / "COMPLETE").write_text("ok\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset"); p.add_argument("transform", choices=["MiniRocket", "MultiRocket"])
    p.add_argument("--data-dir", default=os.environ.get("AEON_DATA"))
    p.add_argument("--cache-dir", default=str(HERE / "feature_cache"))
    p.add_argument("--output-dir", default=str(HERE / "results"))
    p.add_argument("--design", choices=["minimal", "full"], default="minimal")
    p.add_argument("--fixed-lambda", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=1); p.add_argument("--save-predictions", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_case(parse_args())
