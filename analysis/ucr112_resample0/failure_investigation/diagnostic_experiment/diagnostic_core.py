"""Independent diagnostic ridge implementation; does not modify aeon estimators."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelBinarizer, StandardScaler


@dataclass
class Decomposition:
    dtype: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    classes: np.ndarray
    Y: np.ndarray
    target_mean: np.ndarray
    V: np.ndarray
    S2: np.ndarray
    R: np.ndarray
    R2: np.ndarray
    RTY: np.ndarray
    mask: np.ndarray
    diagnostics: dict


def clipped_softmax(scores: np.ndarray, dtype: np.dtype) -> np.ndarray:
    eps = np.finfo(dtype).eps
    bound = -np.log(eps)
    z = np.clip(scores, -bound, bound)
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _calibrate(Y, loo_scores, target_mean, dtype, constraint):
    def objective(x):
        scale = np.exp(x[0]) if constraint == "positive" else x[0]
        p = clipped_softmax(scale * loo_scores + target_mean, dtype)
        return -np.log(np.maximum((Y * p).max(axis=1), np.finfo(dtype).tiny)).sum()

    x0 = np.array([0.0 if constraint == "positive" else 1.0], dtype=float)
    result = minimize(objective, x0=x0, method="BFGS", jac="2-point")
    scale = float(np.exp(result.x[0]) if constraint == "positive" else result.x[0])
    return scale, float(result.fun), bool(result.success), str(result.message), int(result.nit)


def prepare_decomposition(raw_train, raw_test, y_train, y_test, centered, dtype_name):
    dtype = np.dtype(dtype_name)
    started = perf_counter()
    scaler = StandardScaler(with_mean=centered, with_std=True)
    X_train = scaler.fit_transform(raw_train).astype(dtype, copy=False)
    X_test = scaler.transform(raw_test).astype(dtype, copy=False)
    feature_mean_abs = float(np.mean(np.abs(X_train.mean(axis=0, dtype=np.float64))))
    feature_mean_max_abs = float(np.max(np.abs(X_train.mean(axis=0, dtype=np.float64))))
    feature_std = X_train.std(axis=0, dtype=dtype)
    mask = feature_std < dtype.type(1e-6)
    X_train = X_train[:, ~mask]
    X_test = X_test[:, ~mask]
    X_design = np.hstack((np.ones((len(X_train), 1), dtype=dtype), X_train))

    lb = LabelBinarizer(neg_label=-1)
    Y = lb.fit_transform(y_train).astype(dtype)
    if Y.shape[1] == 1:
        Y = np.hstack((-Y, Y))
    target_mean = Y.mean(axis=0)
    Y = Y - target_mean
    n, p = X_design.shape
    eps = np.finfo(dtype).eps
    if n >= p:
        G = X_design.T @ X_design
        eigen_raw, V = np.linalg.eigh(G)
        eigen_nonpositive = int((eigen_raw <= 0).sum())
        S2 = np.clip(eigen_raw, eps, None)
        S = np.sqrt(S2)
        U = (X_design @ V) / S
        solve_space = "primal"
    else:
        G = X_design @ X_design.T
        eigen_raw, U = np.linalg.eigh(G)
        eigen_nonpositive = int((eigen_raw <= 0).sum())
        S2 = np.clip(eigen_raw, eps, None)
        S = np.sqrt(S2)
        V = (X_design.T @ U) / S
        solve_space = "dual"
    R = U * S
    positive = eigen_raw[eigen_raw > 0]
    diagnostics = {
        "centered": centered, "dtype": dtype_name, "n_train": n, "n_features_raw": raw_train.shape[1],
        "n_features_used": X_train.shape[1], "n_low_variance_dropped": int(mask.sum()),
        "feature_mean_abs_average": feature_mean_abs, "feature_mean_max_abs": feature_mean_max_abs,
        "solve_space": solve_space, "gram_min_eigenvalue_raw": float(eigen_raw.min()),
        "gram_max_eigenvalue_raw": float(eigen_raw.max()), "gram_nonpositive_eigenvalues": eigen_nonpositive,
        "gram_condition_positive": float(positive.max() / positive.min()) if len(positive) else np.inf,
        "decomposition_seconds": perf_counter() - started,
    }
    return Decomposition(dtype_name, X_train, X_test, np.asarray(y_train), np.asarray(y_test),
                         lb.classes_, Y, target_mean, V, S2, R, R**2, R.T @ Y, mask, diagnostics)


def evaluate_preval_path(dec: Decomposition, lambdas, calibration_constraints):
    dtype = np.dtype(dec.dtype)
    per_lambda, models = [], {}
    for lam in np.asarray(lambdas, dtype=dtype):
        alpha_hat = dec.RTY / (dec.S2[:, None] + lam)
        fitted = dec.R @ alpha_hat
        residual = dec.Y - fitted
        diag_h = (dec.R2 / (dec.S2 + lam)).sum(axis=1)
        loo_residual = residual / np.clip(1 - diag_h[:, None], np.finfo(dtype).eps, None)
        loo_scores = fitted - (loo_residual - residual)
        reference_coef = dec.V @ alpha_hat
        raw_intercept = dec.target_mean + reference_coef[0]
        raw_coef = reference_coef[1:].T
        raw_test_scores = dec.X_test @ raw_coef.T + raw_intercept
        for constraint in calibration_constraints:
            scale, nll, success, message, nit = _calibrate(
                dec.Y, loo_scores, dec.target_mean, dtype, constraint
            )
            scaled_scores = scale * (raw_test_scores - dec.target_mean) + dec.target_mean
            probabilities = clipped_softmax(scaled_scores, dtype)
            tied = np.isclose(probabilities, probabilities.max(axis=1, keepdims=True), rtol=1e-7, atol=1e-12).sum(axis=1)
            clipped_bound = -np.log(np.finfo(dtype).eps)
            models[(float(lam), constraint)] = {
                "raw_scores": raw_test_scores, "scaled_scores": scaled_scores,
                "probabilities": probabilities, "scale": scale, "nll": nll,
                "optimizer_success": success, "optimizer_message": message, "optimizer_iterations": nit,
                "diag_h_min": float(diag_h.min()), "diag_h_max": float(diag_h.max()),
                "fraction_logits_clipped_upper": float((scaled_scores >= clipped_bound).mean()),
                "fraction_logits_clipped_lower": float((scaled_scores <= -clipped_bound).mean()),
                "fraction_rows_tied_max": float((tied > 1).mean()),
                "mean_tied_max": float(tied.mean()), "max_tied_max": int(tied.max()),
                "coef_l2": float(np.linalg.norm(raw_coef)),
            }
            per_lambda.append({"lambda": float(lam), "calibration_constraint": constraint,
                               **{k: v for k, v in models[(float(lam), constraint)].items()
                                  if k not in {"raw_scores", "scaled_scores", "probabilities"}}})
    return per_lambda, models


def evaluate_configuration(dec, model, decision_rule):
    if decision_rule == "raw_score":
        idx = model["raw_scores"].argmax(axis=1)
    elif decision_rule == "clipped_softmax":
        idx = model["probabilities"].argmax(axis=1)
    else:
        raise ValueError(decision_rule)
    pred = dec.classes[idx]
    return {
        "accuracy": float(accuracy_score(dec.y_test, pred)),
        "log_loss": float(log_loss(dec.y_test, model["probabilities"], labels=dec.classes)),
        "unique_predicted_classes": int(np.unique(pred).size),
        "dominant_prediction_fraction": float(np.unique(pred, return_counts=True)[1].max() / len(pred)),
    }, pred


def evaluate_ridge_reference(raw_train, raw_test, y_train, y_test, centered, lambdas):
    scaler = StandardScaler(with_mean=centered, with_std=True)
    X_train = scaler.fit_transform(raw_train)
    X_test = scaler.transform(raw_test)
    ridge = RidgeClassifierCV(alphas=lambdas).fit(X_train, y_train)
    pred = ridge.predict(X_test)
    return {"accuracy": float(accuracy_score(y_test, pred)), "selected_lambda": float(ridge.alpha_),
            "unique_predicted_classes": int(np.unique(pred).size),
            "dominant_prediction_fraction": float(np.unique(pred, return_counts=True)[1].max() / len(pred))}, pred
