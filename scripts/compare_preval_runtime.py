"""Benchmark standalone PreVal against the aeon PrevalidatedRidgeClassifier."""

import argparse
import time
from pathlib import Path

import numpy as np

from compare_preval_implementations import _cases, _load_implementations
from compare_preval_real_datasets import DATASETS, LAMBDAS, _as_tabular


SYNTHETIC_CASES = (
    "seed_0_binary_small_n_ge_p_int",
    "seed_1_multiclass_n_ge_p_string",
    "seed_2_binary_n_lt_p_string",
    "seed_7_multiclass_n_lt_p_int",
)

AEON_REPO = Path(r"C:\Users\Adam\Desktop\Aeon\aeon")


def _fit_time(estimator_class, lambdas, X, y):
    """Time only fit; estimator construction and input copies are excluded."""
    estimator = estimator_class(lambdas=lambdas.copy())
    X_copy, y_copy = X.copy(), y.copy()
    start = time.perf_counter()
    estimator.fit(X_copy, y_copy)
    return time.perf_counter() - start


def _benchmark(X, y, lambdas, reference_class, port_class, warmups, repeats):
    """Warm up and measure both implementations with alternating run order."""
    original_times = []
    aeon_times = []

    for iteration in range(warmups + repeats):
        measured = iteration >= warmups
        order = (
            (("original", reference_class), ("aeon", port_class))
            if iteration % 2 == 0
            else (("aeon", port_class), ("original", reference_class))
        )
        for implementation, estimator_class in order:
            elapsed = _fit_time(estimator_class, lambdas, X, y)
            if measured:
                if implementation == "original":
                    original_times.append(elapsed)
                else:
                    aeon_times.append(elapsed)

    original_median = float(np.median(original_times))
    aeon_median = float(np.median(aeon_times))
    ratio = aeon_median / original_median
    return {
        "original_times": original_times,
        "aeon_times": aeon_times,
        "original_median": original_median,
        "aeon_median": aeon_median,
        "ratio": ratio,
        "percent_difference": (ratio - 1.0) * 100.0,
    }


def _synthetic_data():
    """Return four deterministic cases covering class count and shape branch."""
    selected = {case["name"]: case for case in _cases()}
    for name in SYNTHETIC_CASES:
        case = selected[name]
        yield name, case["X"], case["y"], case["lambdas"]


def _real_data(load_classification):
    """Return the official training split of each existing parity dataset."""
    for name in DATASETS:
        X, y = load_classification(name, split="train")
        yield name, _as_tabular(X), y, LAMBDAS


def _run_group(
    heading, datasets, reference_class, port_class, warmups, repeats
):
    """Benchmark and print one group of datasets."""
    results = []
    for name, X, y, lambdas in datasets:
        timing = _benchmark(
            X, y, lambdas, reference_class, port_class, warmups, repeats
        )
        results.append(
            {
                "group": heading,
                "name": name,
                "n": X.shape[0],
                "p": X.shape[1],
                "classes": len(np.unique(y)),
                **timing,
            }
        )

    print(f"\n{heading}")
    print(
        f"{'Dataset/case':<43} {'Shape/classes':<21} "
        f"{'Original (s)':>12} {'Aeon (s)':>10} {'Ratio':>8} {'Difference':>12}"
    )
    print("-" * 112)
    for result in results:
        shape = f"{result['n']}x{result['p']}, {result['classes']} cls"
        print(
            f"{result['name']:<43} {shape:<21} "
            f"{result['original_median']:>12.6f} "
            f"{result['aeon_median']:>10.6f} "
            f"{result['ratio']:>8.3f} "
            f"{result['percent_difference']:>+11.1f}%"
        )
    return results


def _interpret(results):
    """Give a short interpretation based on the aggregate median ratio."""
    median_ratio = float(np.median([result["ratio"] for result in results]))
    largest_slowdown = max(results, key=lambda result: result["ratio"])

    if median_ratio > 1.25:
        conclusion = "The aeon port shows a substantial overall runtime regression."
    elif median_ratio < 0.80:
        conclusion = "The aeon port is substantially faster overall."
    elif largest_slowdown["ratio"] > 1.50:
        conclusion = (
            "Runtime is broadly comparable overall, but at least one case has a "
            "notable slowdown that may merit investigation."
        )
    else:
        conclusion = (
            "Runtime is broadly comparable, with no substantial regression evident."
        )
    return conclusion


def _print_summary(results):
    """Print raw measurements and aggregate statistics."""
    print("\nRaw per-case measurements (seconds)")
    for result in results:
        original = ", ".join(f"{value:.6f}" for value in result["original_times"])
        aeon = ", ".join(f"{value:.6f}" for value in result["aeon_times"])
        print(f"{result['name']}: original=[{original}]; aeon=[{aeon}]")

    ratios = [result["ratio"] for result in results]
    differences = [result["percent_difference"] for result in results]
    largest_slowdown = max(results, key=lambda result: result["ratio"])
    speedups = [result for result in results if result["ratio"] < 1.0]

    print("\nOverall summary")
    print(f"Median runtime ratio: {np.median(ratios):.3f}")
    print(f"Median percentage difference: {np.median(differences):+.1f}%")
    print(
        f"Largest slowdown: {largest_slowdown['name']} "
        f"({largest_slowdown['ratio']:.3f}x, "
        f"{largest_slowdown['percent_difference']:+.1f}%)"
    )
    if speedups:
        largest_speedup = min(speedups, key=lambda result: result["ratio"])
        print(
            f"Largest speed-up: {largest_speedup['name']} "
            f"({largest_speedup['ratio']:.3f}x, "
            f"{largest_speedup['percent_difference']:+.1f}%)"
        )
    else:
        closest = min(results, key=lambda result: result["ratio"])
        print(
            "Largest speed-up: none observed; smallest slowdown was "
            f"{closest['name']} ({closest['ratio']:.3f}x, "
            f"{closest['percent_difference']:+.1f}%)"
        )
    print(f"Interpretation: {_interpret(results)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aeon-repo",
        type=Path,
        default=Path(r"C:\Users\Adam\Desktop\Aeon\aeon"),
        help="Path to the local aeon checkout (default: %(default)s)",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 3:
        parser.error("use at least one warm-up and three measured repetitions")

    aeon_repo = args.aeon_repo.resolve()
    reference_class, port_class = _load_implementations(aeon_repo)
    from aeon.datasets import load_classification

    synthetic_results = _run_group(
        "Synthetic datasets",
        _synthetic_data(),
        reference_class,
        port_class,
        args.warmups,
        args.repeats,
    )
    real_results = _run_group(
        "Real datasets",
        _real_data(load_classification),
        reference_class,
        port_class,
        args.warmups,
        args.repeats,
    )
    _print_summary(synthetic_results + real_results)


if __name__ == "__main__":
    main()
