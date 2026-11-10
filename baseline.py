"""Baseline checkpoint: seasonal naive and tuned ridge, 24-hour-ahead hourly bike-share demand.

Run:  python baseline_simple.py      (expects hour.csv in the same folder)
Out:  a printed summary and baseline.png
"""

import csv
import math
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
TEST_MONTHS = [(2012, 7)]     # final run: (2012, 7) ... (2012, 12)
LAGS = [24, 48, 168]          # hours, all >= 24
ALPHAS = [10 ** (-3 + 0.5 * i) for i in range(15)]   # 0.001 ... 10000
K_FOLDS = 5
N_BOOT = 1000
N_NUMERIC = 7                 # the last 7 columns of each feature row get scaled


# ---------------------------------------------------------------- data
def load(path):
    """Return a list of usable rows (dictionaries), sorted by time."""
    count_at = {}
    raw_rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = datetime.strptime(r["dteday"], "%Y-%m-%d") + timedelta(hours=int(r["hr"]))
            count_at[t] = int(r["cnt"])
            raw_rows.append((t, r))

    first_hour = datetime(2011, 1, 1, 0)
    rows = []
    dropped_first_week = 0
    dropped_missing_lag = 0
    for t, r in raw_rows:
        # Look up demand exactly 24, 48 and 168 clock hours earlier.
        lag_counts = [count_at.get(t - timedelta(hours=l)) for l in LAGS]
        if None in lag_counts:                       # a lag fell on a missing hour
            if t < first_hour + timedelta(hours=168):
                dropped_first_week += 1
            else:
                dropped_missing_lag += 1
            continue
        rows.append({
            "t": t,
            "cnt": count_at[t],
            "hr": int(r["hr"]),
            "workingday": int(r["workingday"]),
            "weathersit": int(r["weathersit"]),
            "numeric": [float(r["temp"]), float(r["atemp"]), float(r["hum"]), float(r["windspeed"])],
            "lag_counts": lag_counts,
        })
    rows.sort(key=lambda row: row["t"])

    all_hours = int((datetime(2013, 1, 1) - first_hour).total_seconds() // 3600)
    print("rows loaded:", len(raw_rows))
    print("hours in 2011-2012:", all_hours, " missing:", all_hours - len(raw_rows))
    print("dropped (first week):", dropped_first_week)
    print("dropped (lag on a missing hour):", dropped_missing_lag)
    print("rows used:", len(rows))
    return rows


# ---------------------------------------------------------------- features
def feature_row(row):
    """Turn one row into a list of 58 numbers."""
    x = [0.0] * 48                                   # one-hot: hour x working day
    x[row["hr"] + 24 * row["workingday"]] = 1.0
    for k in (2, 3, 4):                              # one-hot weather (1 is the reference)
        x.append(1.0 if row["weathersit"] == k else 0.0)
    x += row["numeric"]                              # temp, atemp, hum, windspeed
    x += [math.log(1 + c) for c in row["lag_counts"]]   # lags on the log scale
    return x


# ---------------------------------------------------------------- models
def rmse(actual, predicted):
    total = 0.0
    for a, p in zip(actual, predicted):
        total += (a - p) ** 2
    return math.sqrt(total / len(actual))


def mae(actual, predicted):
    total = 0.0
    for a, p in zip(actual, predicted):
        total += abs(a - p)
    return total / len(actual)


def ridge_fit_predict(train_rows, test_rows, alpha):
    """Fit ridge on log(1 + cnt) using train_rows, predict rentals for test_rows."""
    X = np.array([feature_row(r) for r in train_rows])
    y = np.array([math.log(1 + r["cnt"]) for r in train_rows])
    X_test = np.array([feature_row(r) for r in test_rows])

    # Scale the numeric columns using the TRAINING mean and std only.
    mean = X[:, -N_NUMERIC:].mean(axis=0)
    std = X[:, -N_NUMERIC:].std(axis=0)
    X[:, -N_NUMERIC:] = (X[:, -N_NUMERIC:] - mean) / std
    X_test[:, -N_NUMERIC:] = (X_test[:, -N_NUMERIC:] - mean) / std

    # Ridge solution: w = (X'X + alpha*I)^-1 X'y, with the intercept left unpenalised
    # (done by centring X and y first).
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean))
    intercept = y_mean - x_mean @ w

    log_pred = X_test @ w + intercept
    return [max(math.exp(v) - 1, 0.0) for v in log_pred]      # back to rentals


def time_ordered_folds(n, k):
    """K folds where each fold trains on rows [0, start) and validates on the
    next block. Returns a list of (train_end, valid_end) positions."""
    size = n // (k + 1)
    folds = []
    for i in range(k):
        start = n - (k - i) * size
        folds.append((start, start + size))
    return folds


def tuned_ridge(train_rows, test_rows):
    """Pick alpha by time-ordered CV inside the training rows, then predict."""
    folds = time_ordered_folds(len(train_rows), K_FOLDS)
    cv_score = []
    for alpha in ALPHAS:
        fold_scores = []
        for train_end, valid_end in folds:
            fit_part = train_rows[:train_end]                 # the past
            valid_part = train_rows[train_end:valid_end]      # the block right after it
            pred = ridge_fit_predict(fit_part, valid_part, alpha)
            fold_scores.append(rmse([r["cnt"] for r in valid_part], pred))
        cv_score.append(sum(fold_scores) / len(fold_scores))
    best = cv_score.index(min(cv_score))
    print("  ridge: chose alpha = %.3g (CV RMSE %.1f)" % (ALPHAS[best], cv_score[best]))
    return ridge_fit_predict(train_rows, test_rows, ALPHAS[best])


def naive(train_rows, test_rows):
    """Same hour last week (the 168-hour lag is the third one)."""
    return [r["lag_counts"][2] for r in test_rows]


MODELS = {"naive": naive, "ridge": tuned_ridge}


# ---------------------------------------------------------------- error bars
def percentile_interval(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def block_bootstrap(test_rows, predictions, rng):
    """95% intervals by resampling whole test days with replacement."""
    # Group the errors by day: errors_by_day[model][day] = list of that day's errors.
    days = sorted(set(r["t"].date() for r in test_rows))
    errors_by_day = {}
    for name in predictions:
        errors_by_day[name] = {d: [] for d in days}
        for r, p in zip(test_rows, predictions[name]):
            errors_by_day[name][r["t"].date()].append(r["cnt"] - p)

    # Each row of draws is one pretend month: a list of day positions.
    draws = rng.integers(0, len(days), size=(N_BOOT, len(days)))

    boot_rmse = {name: [] for name in predictions}
    boot_mae = {name: [] for name in predictions}
    for draw in draws:
        for name in predictions:                     # same days for every model = paired
            errs = []
            for i in draw:
                errs += errors_by_day[name][days[i]]
            boot_rmse[name].append(math.sqrt(sum(e * e for e in errs) / len(errs)))
            boot_mae[name].append(sum(abs(e) for e in errs) / len(errs))

    out = {}
    for name in predictions:
        out[name] = {"rmse": percentile_interval(boot_rmse[name]),
                     "mae": percentile_interval(boot_mae[name])}
    diffs = [r - n for r, n in zip(boot_rmse["ridge"], boot_rmse["naive"])]
    out["ridge_minus_naive"] = percentile_interval(diffs)
    return out


# ---------------------------------------------------------------- rolling origin
def rolling_origin(rows, rng):
    results = []
    for year, month in TEST_MONTHS:
        month_start = datetime(year, month, 1)
        train_rows = [r for r in rows if r["t"] < month_start]
        test_rows = [r for r in rows if r["t"].year == year and r["t"].month == month]
        actual = [r["cnt"] for r in test_rows]
        print("\n%d-%02d: train %d rows, test %d rows" % (year, month, len(train_rows), len(test_rows)))

        predictions = {}
        for name, model in MODELS.items():
            predictions[name] = model(train_rows, test_rows)
        intervals = block_bootstrap(test_rows, predictions, rng)

        res = {"label": month_start.strftime("%b %Y")}
        for name in MODELS:
            res[name] = {"rmse": rmse(actual, predictions[name]),
                         "mae": mae(actual, predictions[name]),
                         "rmse_ci": intervals[name]["rmse"],
                         "mae_ci": intervals[name]["mae"]}
            print("  %-5s RMSE %.1f [%.1f, %.1f]   MAE %.1f [%.1f, %.1f]" % (
                name, res[name]["rmse"], *res[name]["rmse_ci"], res[name]["mae"], *res[name]["mae_ci"]))
        res["diff"] = res["ridge"]["rmse"] - res["naive"]["rmse"]
        res["diff_ci"] = intervals["ridge_minus_naive"]
        print("  ridge - naive RMSE %.1f [%.1f, %.1f]" % (res["diff"], *res["diff_ci"]))
        results.append(res)
    return results


# ---------------------------------------------------------------- figure
def figure(results, path):
    labels = [r["label"] for r in results]
    xs = list(range(len(results)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8))

    for name, shift, colour, label in [("naive", -0.08, "gray", "Seasonal naive (same hour last week)"),
                                       ("ridge", 0.08, "tab:blue", "Tuned ridge")]:
        values = [r[name]["rmse"] for r in results]
        below = [r[name]["rmse"] - r[name]["rmse_ci"][0] for r in results]
        above = [r[name]["rmse_ci"][1] - r[name]["rmse"] for r in results]
        left.errorbar([x + shift for x in xs], values, yerr=[below, above],
                      fmt="o-", capsize=4, color=colour, label=label)
    left.set_ylim(0, None)
    left.set_ylabel("Test RMSE (rentals per hour)")
    left.set_title("24-hour-ahead forecast error")
    left.legend(frameon=False, fontsize=8)

    values = [r["diff"] for r in results]
    below = [r["diff"] - r["diff_ci"][0] for r in results]
    above = [r["diff_ci"][1] - r["diff"] for r in results]
    right.errorbar(xs, values, yerr=[below, above], fmt="o-", capsize=4, color="tab:blue")
    right.axhline(0, color="black", linewidth=0.8, linestyle="--")
    right.set_ylabel("Paired RMSE difference\n(ridge minus naive)")
    right.set_title("Below zero: ridge is better")

    for ax in (left, right):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.set_xlabel("Test month")
    fig.text(0.5, 0.005, "Bars: 95% day-block bootstrap intervals (1,000 resamples). "
             "Data: UCI Bike Sharing Dataset (Fanaee-T & Gama, 2013), CC BY 4.0.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    rows = load("hour.csv")
    results = rolling_origin(rows, np.random.default_rng(SEED))
    figure(results, "baseline.png")