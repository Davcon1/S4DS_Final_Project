"""XGBoost in the same rolling-origin loop as the baseline (David).

Compares seasonal naive, tuned ridge and tuned XGBoost for 24-hour-ahead hourly
bike-share demand. Everything that is not XGBoost-specific is imported from
baseline.py, so the three models share rows, features, folds and metrics.

Run:  python david.py            all six test months (Jul-Dec 2012)
      python david.py 1          first test month only (development check)
Needs hour.csv and baseline.py in the same folder.
Out:  a printed summary, david_results.json, david_predictions.csv, david.png
"""

import csv
import json
import math
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime

import baseline
from baseline import (load, rmse, mae, time_ordered_folds, percentile_interval,
                      ALPHAS, K_FOLDS, N_BOOT, SEED)

TEST_MONTHS = [(2012, m) for m in range(7, 13)]     # each month is forecast once
N_TRIALS = 30                                       # XGBoost random-search budget
SEARCH_SEED = SEED + 1                              # separate stream from the bootstrap
N_THREADS = 2
MODEL_NAMES = ["naive", "ridge", "xgboost"]
PAIRS = [("xgboost", "ridge"), ("ridge", "naive"), ("xgboost", "naive")]   # a minus b


# ---------------------------------------------------------------- features
def matrix(rows):
    """The same 58 columns ridge sees (baseline.feature_row). Trees do not need
    scaling, so no statistic of any kind is fitted here."""
    return np.array([baseline.feature_row(r) for r in rows], dtype=np.float32)


def log_target(rows):
    return np.array([math.log(1 + r["cnt"]) for r in rows], dtype=np.float32)


# ---------------------------------------------------------------- XGBoost
def sample_configs(n, seed):
    """n random hyperparameter settings, drawn once and reused at every origin
    (the counterpart of ridge's fixed penalty grid)."""
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append({
            "n_estimators": int(rng.integers(100, 601)),
            "learning_rate": float(10 ** rng.uniform(-2, math.log10(0.3))),
            "max_depth": int(rng.integers(2, 9)),
            "min_child_weight": float(10 ** rng.uniform(0, 1.5)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_lambda": float(10 ** rng.uniform(-1, 2)),
        })
    return configs


def xgb_fit_predict(X, y, X_test, config):
    """Fit on log(1 + cnt), predict rentals (same back-transform as ridge)."""
    model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             random_state=SEED, n_jobs=N_THREADS, **config)
    model.fit(X, y)
    return [max(math.exp(v) - 1, 0.0) for v in model.predict(X_test)]


def tuned_xgboost(train_rows, test_rows, configs):
    """Pick one of the configs by the same time-ordered CV as ridge (mean fold RMSE
    on the rental scale, training rows only), refit on all training rows, predict."""
    X, y = matrix(train_rows), log_target(train_rows)
    actual = [r["cnt"] for r in train_rows]
    folds = time_ordered_folds(len(train_rows), K_FOLDS)
    cv_score = []
    for config in configs:
        fold_scores = []
        for train_end, valid_end in folds:
            pred = xgb_fit_predict(X[:train_end], y[:train_end], X[train_end:valid_end], config)
            fold_scores.append(rmse(actual[train_end:valid_end], pred))
        cv_score.append(sum(fold_scores) / len(fold_scores))
    best = cv_score.index(min(cv_score))
    print("  xgboost: chose trial %d (CV RMSE %.1f)  %s" % (
        best, cv_score[best], json.dumps(configs[best])))
    pred = xgb_fit_predict(X, y, matrix(test_rows), configs[best])
    return pred, {"trial": best, "cv_rmse": cv_score[best], "config": configs[best],
                  "cv_rmse_all_trials": cv_score}


def tuned_ridge(train_rows, test_rows):
    """baseline.tuned_ridge, but also returning the chosen alpha and CV score."""
    folds = time_ordered_folds(len(train_rows), K_FOLDS)
    cv_score = []
    for alpha in ALPHAS:
        fold_scores = []
        for train_end, valid_end in folds:
            valid = train_rows[train_end:valid_end]
            pred = baseline.ridge_fit_predict(train_rows[:train_end], valid, alpha)
            fold_scores.append(rmse([r["cnt"] for r in valid], pred))
        cv_score.append(sum(fold_scores) / len(fold_scores))
    best = cv_score.index(min(cv_score))
    print("  ridge: chose alpha = %.3g (CV RMSE %.1f)" % (ALPHAS[best], cv_score[best]))
    pred = baseline.ridge_fit_predict(train_rows, test_rows, ALPHAS[best])
    return pred, {"alpha": ALPHAS[best], "cv_rmse": cv_score[best]}


# ---------------------------------------------------------------- error bars
def day_sums(test_rows, predictions):
    """Per test day and model: sum of squared errors, sum of absolute errors, hours."""
    days = sorted(set(r["t"].date() for r in test_rows))
    position = {d: i for i, d in enumerate(days)}
    sums = {name: {"sq": np.zeros(len(days)), "abs": np.zeros(len(days))} for name in predictions}
    hours = np.zeros(len(days))
    for i, r in enumerate(test_rows):
        d = position[r["t"].date()]
        hours[d] += 1
        for name in predictions:
            e = r["cnt"] - predictions[name][i]
            sums[name]["sq"][d] += e * e
            sums[name]["abs"][d] += abs(e)
    return sums, hours


def boot_totals(sums, hours, draws):
    """For each bootstrap resample of days: total squared error, absolute error, hours.
    Every model uses the same resampled days, so differences are paired."""
    totals = {name: {"sq": sums[name]["sq"][draws].sum(axis=1),
                     "abs": sums[name]["abs"][draws].sum(axis=1)} for name in sums}
    return totals, hours[draws].sum(axis=1)


def intervals(totals, n):
    """95% percentile intervals for each model's RMSE and MAE and each paired
    RMSE difference, from bootstrap totals."""
    boot_rmse = {name: np.sqrt(totals[name]["sq"] / n) for name in totals}
    out = {name: {"rmse_ci": percentile_interval(boot_rmse[name]),
                  "mae_ci": percentile_interval(totals[name]["abs"] / n)} for name in totals}
    for a, b in PAIRS:
        out["%s_minus_%s" % (a, b)] = percentile_interval(boot_rmse[a] - boot_rmse[b])
    return out


def summarise(actual, predictions, ci):
    res = {}
    for name in MODEL_NAMES:
        res[name] = {"rmse": rmse(actual, predictions[name]),
                     "mae": mae(actual, predictions[name]),
                     "mean_error": sum(a - p for a, p in zip(actual, predictions[name])) / len(actual),
                     "rmse_ci": ci[name]["rmse_ci"], "mae_ci": ci[name]["mae_ci"]}
        print("  %-7s RMSE %6.1f [%6.1f, %6.1f]   MAE %5.1f [%5.1f, %5.1f]   mean error %+5.1f" % (
            name, res[name]["rmse"], *res[name]["rmse_ci"], res[name]["mae"],
            *res[name]["mae_ci"], res[name]["mean_error"]))
    for a, b in PAIRS:
        key = "%s_minus_%s" % (a, b)
        diff = res[a]["rmse"] - res[b]["rmse"]
        lo, hi = ci[key]
        verdict = "tie" if lo <= 0 <= hi else ("%s better" % (a if hi < 0 else b))
        res[key] = {"rmse_diff": diff, "ci": [lo, hi], "verdict": verdict}
        print("  %-7s - %-5s RMSE %+6.1f [%+6.1f, %+6.1f]  -> %s" % (a, b, diff, lo, hi, verdict))
    return res


# ---------------------------------------------------------------- rolling origin
def rolling_origin(rows, months, rng):
    configs = sample_configs(N_TRIALS, SEARCH_SEED)
    results, saved = [], []
    pooled_actual, pooled_pred = [], {name: [] for name in MODEL_NAMES}
    pooled_totals, pooled_n = None, 0
    for year, month in months:
        started = time.time()
        month_start = datetime(year, month, 1)
        train_rows = [r for r in rows if r["t"] < month_start]
        test_rows = [r for r in rows if r["t"].year == year and r["t"].month == month]
        actual = [r["cnt"] for r in test_rows]
        print("\n%d-%02d: train %d rows, test %d rows" % (year, month, len(train_rows), len(test_rows)))

        predictions = {"naive": baseline.naive(train_rows, test_rows)}
        predictions["ridge"], ridge_info = tuned_ridge(train_rows, test_rows)
        predictions["xgboost"], xgb_info = tuned_xgboost(train_rows, test_rows, configs)

        # Day-block bootstrap. One generator, months in order, so July uses exactly
        # the resamples of baseline.py and reproduces its intervals.
        sums, hours = day_sums(test_rows, predictions)
        draws = rng.integers(0, len(hours), size=(N_BOOT, len(hours)))
        totals, n = boot_totals(sums, hours, draws)

        res = summarise(actual, predictions, intervals(totals, n))
        res.update({"label": month_start.strftime("%b"), "month": "%d-%02d" % (year, month),
                    "n_train": len(train_rows), "n_test": len(test_rows), "n_test_days": len(hours),
                    "ridge_tuning": ridge_info, "xgboost_tuning": xgb_info,
                    "seconds": time.time() - started})
        results.append(res)

        # Pool across months: resample b of the pooled set is resample b of every
        # month put together (days are resampled within their own month).
        pooled_actual += actual
        for name in MODEL_NAMES:
            pooled_pred[name] += predictions[name]
        if pooled_totals is None:
            pooled_totals, pooled_n = totals, n
        else:
            for name in MODEL_NAMES:
                for k in ("sq", "abs"):
                    pooled_totals[name][k] = pooled_totals[name][k] + totals[name][k]
            pooled_n = pooled_n + n
        for i, r in enumerate(test_rows):
            saved.append([r["t"].isoformat(sep=" "), r["cnt"]] +
                         ["%.3f" % predictions[name][i] for name in MODEL_NAMES])

    print("\nall %d test months pooled (%d hours)" % (len(months), len(pooled_actual)))
    pooled = summarise(pooled_actual, pooled_pred, intervals(pooled_totals, pooled_n))
    pooled["n_test"] = len(pooled_actual)
    return results, pooled, saved


# ---------------------------------------------------------------- figure
def figure(results, pooled, path):
    labels = [r["label"] for r in results]
    xs = list(range(len(results)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8))

    for name, shift, colour, marker, label in [
            ("naive", -0.15, "gray", "o", "Seasonal naive (same hour last week)"),
            ("ridge", 0.0, "tab:blue", "s", "Tuned ridge"),
            ("xgboost", 0.15, "tab:orange", "^", "Tuned XGBoost")]:
        values = [r[name]["rmse"] for r in results]
        below = [r[name]["rmse"] - r[name]["rmse_ci"][0] for r in results]
        above = [r[name]["rmse_ci"][1] - r[name]["rmse"] for r in results]
        left.errorbar([x + shift for x in xs], values, yerr=[below, above],
                      fmt=marker + "-", capsize=3, markersize=4, linewidth=1, color=colour, label=label)
    left.set_ylim(0, None)
    left.set_ylabel("Test RMSE (rentals per hour)")
    left.set_title("24-hour-ahead forecast error")
    left.legend(frameon=False, fontsize=7, loc="lower left")
    left.set_xticks(xs)
    left.set_xticklabels(labels)

    key = "xgboost_minus_ridge"
    points = [(x, r[key]) for x, r in zip(xs, results)] + [(len(xs) + 0.3, pooled[key])]
    for x, d in points:
        tie = d["verdict"] == "tie"
        right.errorbar([x], [d["rmse_diff"]],
                       yerr=[[d["rmse_diff"] - d["ci"][0]], [d["ci"][1] - d["rmse_diff"]]],
                       fmt="o", capsize=3, markersize=5, color="tab:orange",
                       markerfacecolor="white" if tie else "tab:orange")
    right.axhline(0, color="black", linewidth=0.8, linestyle="--")
    right.axvline(len(xs) - 0.35, color="lightgray", linewidth=0.8)
    right.set_ylabel("Paired RMSE difference\n(XGBoost minus ridge)")
    right.set_title("Below zero: XGBoost is better")
    right.set_xticks(xs + [len(xs) + 0.3])
    right.set_xticklabels(labels + ["All"])

    for ax in (left, right):
        ax.set_xlabel("Test month (2012)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.text(0.5, 0.005, "Bars: 95% day-block bootstrap intervals (1,000 resamples). Hollow marker: interval "
             "contains zero (tie).\nData: UCI Bike Sharing Dataset (Fanaee-T & Gama, 2013), CC BY 4.0.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    n_months = int(sys.argv[1]) if len(sys.argv) > 1 else len(TEST_MONTHS)
    tag = "" if n_months == len(TEST_MONTHS) else "_dev"
    started = time.time()
    rows = load("hour.csv")
    results, pooled, saved = rolling_origin(rows, TEST_MONTHS[:n_months], np.random.default_rng(SEED))
    seconds = time.time() - started
    print("\ntotal run time: %.0f s" % seconds)

    with open("david_results%s.json" % tag, "w") as f:
        json.dump({"settings": {"seed": SEED, "search_seed": SEARCH_SEED, "n_trials": N_TRIALS,
                                "k_folds": K_FOLDS, "n_boot": N_BOOT, "n_threads": N_THREADS,
                                "xgboost_version": xgb.__version__, "numpy_version": np.__version__},
                   "months": results, "pooled": pooled, "seconds": seconds}, f, indent=2)
    with open("david_predictions%s.csv" % tag, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "cnt"] + MODEL_NAMES)
        writer.writerows(saved)
    figure(results, pooled, "david%s.png" % tag)
