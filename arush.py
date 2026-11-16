"""Ridge residuals by hour, and an hour-by-lag interaction test decided by the
inner time-ordered CV only (training window before the first test month).

Run:  python arush.py      (expects hour.csv in the same folder)
Out:  a printed summary and arush.png. No test-month rows are used.
"""

import math
from contextlib import contextmanager
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline
from baseline import load, rmse, time_ordered_folds, ALPHAS, K_FOLDS, LAGS, TEST_MONTHS

FIGURE_PATH = "arush.png"
baseline_feature_row = baseline.feature_row


# ---------------------------------------------------------------- features
def interaction_feature_row(row):
    """Baseline 58 columns plus 72 hour-by-lag columns (hour indicator times log lag).
    The new columns sit before the numeric block, so baseline's last-N_NUMERIC
    scaling still applies to exactly the same 7 columns."""
    x = baseline_feature_row(row)
    head, numeric = x[:-baseline.N_NUMERIC], x[-baseline.N_NUMERIC:]
    inter = [0.0] * (24 * len(LAGS))
    for j, c in enumerate(row["lag_counts"]):
        inter[24 * j + row["hr"]] = math.log(1 + c)
    return head + inter + numeric


@contextmanager
def features(fn):
    """baseline.ridge_fit_predict looks up feature_row at call time; swapping it
    reuses the baseline fitting and fold-wise scaling unchanged."""
    original = baseline.feature_row
    baseline.feature_row = fn
    try:
        yield
    finally:
        baseline.feature_row = original


# ---------------------------------------------------------------- inner CV
def cv_ridge(train_rows, feature_fn):
    """Same folds, grid and selection rule as baseline.tuned_ridge. Returns the best
    alpha, its per-fold RMSE, and its out-of-fold predictions."""
    folds = time_ordered_folds(len(train_rows), K_FOLDS)
    best = None
    with features(feature_fn):
        for alpha in ALPHAS:
            pred, scores = [], []
            for train_end, valid_end in folds:
                valid = train_rows[train_end:valid_end]
                p = baseline.ridge_fit_predict(train_rows[:train_end], valid, alpha)
                pred += p
                scores.append(rmse([r["cnt"] for r in valid], p))
            mean = sum(scores) / len(scores)
            if best is None or mean < best["cv_rmse"]:      # first minimum, as in baseline
                best = {"alpha": alpha, "cv_rmse": mean, "fold_rmse": scores, "pred": pred}
    best["rows"] = [r for s, e in folds for r in train_rows[s:e]]
    return best


def by_hour(rows, pred):
    """(mean residual, RMSE, n) per hour of day; residual = actual minus predicted."""
    out = []
    for h in range(24):
        res = [r["cnt"] - p for r, p in zip(rows, pred) if r["hr"] == h]
        out.append((sum(res) / len(res), math.sqrt(sum(e * e for e in res) / len(res)), len(res)))
    return out


# ---------------------------------------------------------------- figure
def figure(hourly, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8))
    hours = list(range(24))
    for (name, table), colour in zip(hourly.items(), ["tab:blue", "tab:orange"]):
        left.plot(hours, [m for m, _, _ in table], "o-", color=colour, label=name, markersize=3)
        right.plot(hours, [e for _, e, _ in table], "o-", color=colour, label=name, markersize=3)
    left.axhline(0, color="black", linewidth=0.8, linestyle="--")
    left.set_ylabel("Mean residual (rentals per hour)")
    left.set_title("Above zero: model predicts too low")
    right.set_ylim(0, None)
    right.set_ylabel("RMSE (rentals per hour)")
    right.set_title("Error by hour")
    right.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day")
    fig.text(0.5, 0.005, "Out-of-fold residuals from the 5 time-ordered CV folds inside the "
             "training window (no test-month data). Data: UCI Bike Sharing Dataset, CC BY 4.0.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    rows = load("hour.csv")
    train_rows = [r for r in rows if r["t"] < datetime(*TEST_MONTHS[0], 1)]
    print("\ninner CV on %d training rows (before the first test month)" % len(train_rows))

    fits = {"ridge": cv_ridge(train_rows, baseline_feature_row),
            "ridge + hour x lag": cv_ridge(train_rows, interaction_feature_row)}
    for name, f in fits.items():
        print("  %-18s alpha = %.3g  CV RMSE %.1f  folds %s" % (
            name, f["alpha"], f["cv_rmse"], " ".join("%.1f" % s for s in f["fold_rmse"])))

    base, inter = fits["ridge"], fits["ridge + hour x lag"]
    diffs = [b - a for a, b in zip(base["fold_rmse"], inter["fold_rmse"])]
    print("  interaction minus ridge, per fold: %s  (better in %d of %d folds)" % (
        " ".join("%+.1f" % d for d in diffs), sum(d < 0 for d in diffs), len(diffs)))
    print("  inner-CV choice: %s" % min(fits, key=lambda n: fits[n]["cv_rmse"]))

    hourly = {name: by_hour(f["rows"], f["pred"]) for name, f in fits.items()}
    print("\nout-of-fold residuals by hour (mean residual / RMSE)")
    print("  hour     n   " + "   ".join("%-20s" % n for n in hourly))
    for h in range(24):
        n = hourly["ridge"][h][2]
        cells = "   ".join("%+7.1f / %6.1f      " % hourly[name][h][:2] for name in hourly)
        print("  %4d %5d   %s" % (h, n, cells))
    figure(hourly, FIGURE_PATH)
