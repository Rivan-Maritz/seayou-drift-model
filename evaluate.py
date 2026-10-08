"""Show how accurate the SeaYou drift model is on the held-out test scenarios.

    python evaluate.py

Prints a table and saves evaluation.png. Errors are straight-line distance in metres between the
predicted and true end position. "Physics only" is the interpolated node current advected for the horizon,
"last seen only" assumes the person does not move at all.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from seayou_common import L_X, HORIZONS_MIN
from train_model import predict_end, errors

HERE = Path(__file__).resolve().parent


def find(*names):
    for n in names:
        p = HERE / n
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {names} found next to evaluate.py")


def main():
    bundle = joblib.load(find("model/drift_model.joblib", "drift_model.joblib"))
    test = pd.read_csv(find("data/test.csv", "test.csv"))
    X = test[bundle["features"]]
    hs = test["horizon_min"].values

    clf = bundle.get("beach_clf")
    ex, ey, beach_pred = predict_end(bundle["model_x"], bundle["model_y"], X, clf)
    err = errors(test, ex, ey)
    phys_x = np.minimum(test["last_x"] + test["base_dx"], L_X).values
    err_phys = errors(test, phys_x, (test["last_y"] + test["base_dy"]).values)
    err_stay = errors(test, test["last_x"].values, test["last_y"].values)
    radius = np.interp(hs, list(bundle["radius90"]), list(bundle["radius90"].values()))
    inside = err <= radius

    print(f"Test rows: {len(test)} from {test['scenario_id'].nunique()} unseen scenarios\n")
    print(f"{'Horizon':>8} {'Model':>8} {'Median':>8} {'P90':>8} {'<100m':>7} {'<250m':>7} "
          f"{'Physics':>8} {'Stay':>8} {'Gain':>6} {'90% rad':>8} {'Covered':>8}")
    for h in HORIZONS_MIN:
        m = hs == h
        print(f"{h:>6} m {err[m].mean():>7.0f}m {np.median(err[m]):>7.0f}m {np.quantile(err[m], .9):>7.0f}m "
              f"{(err[m] < 100).mean():>7.0%} {(err[m] < 250).mean():>7.0%} "
              f"{err_phys[m].mean():>7.0f}m {err_stay[m].mean():>7.0f}m "
              f"{1 - err[m].mean() / err_phys[m].mean():>6.0%} {radius[m][0]:>7.0f}m {inside[m].mean():>8.0%}")
    print(f"\n{'ALL':>8} {err.mean():>7.0f}m {np.median(err):>7.0f}m {np.quantile(err, .9):>7.0f}m "
          f"{(err < 100).mean():>7.0%} {(err < 250).mean():>7.0%} {err_phys.mean():>7.0f}m "
          f"{err_stay.mean():>7.0f}m {1 - err.mean() / err_phys.mean():>6.0%} {'':>8} {inside.mean():>8.0%}")

    r2x, r2y = r2_score(test["end_x"], ex), r2_score(test["end_y"], ey)
    true_b = test["beached"].values.astype(bool)
    tp = (beach_pred & true_b).sum()
    print(f"\nR2 of final position: offshore axis {r2x:.3f}, along-beach axis {r2y:.3f}")
    print(f"Beach prediction: accuracy {(beach_pred == true_b).mean():.1%}, "
          f"precision {tp / max(beach_pred.sum(), 1):.1%}, recall {tp / max(true_b.sum(), 1):.1%} "
          f"(only {true_b.mean():.1%} of cases beach, so accuracy alone is misleading)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(HORIZONS_MIN, [err[hs == h].mean() for h in HORIZONS_MIN], "o-", label="Model")
    ax[0].plot(HORIZONS_MIN, [err_phys[hs == h].mean() for h in HORIZONS_MIN], "s--", label="Physics only")
    ax[0].plot(HORIZONS_MIN, [err_stay[hs == h].mean() for h in HORIZONS_MIN], "^:", label="Last seen only")
    ax[0].set_xlabel("Horizon (min)"); ax[0].set_ylabel("Mean error (m)"); ax[0].set_title("Error by horizon"); ax[0].legend()
    for h in HORIZONS_MIN:
        e = np.sort(err[hs == h])
        ax[1].plot(e, np.arange(1, len(e) + 1) / len(e), label=f"{h} min")
    ax[1].set_xlim(0, 1200); ax[1].set_xlabel("Error (m)"); ax[1].set_ylabel("Share of cases within error")
    ax[1].set_title("Cumulative error"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].scatter(test["end_y"], ey, s=3, alpha=.15)
    lim = [test["end_y"].min(), test["end_y"].max()]
    ax[2].plot(lim, lim, "r--"); ax[2].set_xlabel("True along-beach position (m)")
    ax[2].set_ylabel("Predicted (m)"); ax[2].set_title(f"Along-beach position, R2 = {r2y:.2f}")
    fig.tight_layout(); fig.savefig(HERE / "evaluation.png", dpi=130)
    print("\nSaved evaluation.png")


if __name__ == "__main__":
    main()
