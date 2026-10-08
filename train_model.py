"""Train the SeaYou drift model and evaluate it on the held-out test scenarios.

Model design:
  1. Physics baseline: interpolate node currents to the last seen point and advect for the horizon.
  2. Gradient boosting learns the RESIDUAL between the true displacement and that baseline
     (wind/leeway, eddies, rip currents, time trends, shore effects).
  3. A separate classifier predicts whether the person reaches the beach within the horizon.
     If it says yes, the predicted position is snapped to the shoreline.
  4. Split conformal calibration gives a 90% search radius per horizon.
"""
import json, math, os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from seayou_common import L_X, HORIZONS_MIN, feature_names

COVERAGE = 0.90
BEACH_THRESHOLD = 0.5


def fit_reg():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)


def fit_clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def predict_end(model_x, model_y, X, clf=None, thr=BEACH_THRESHOLD):
    """Predicted end position (x, y) and a beached flag for each row of X."""
    ex = X["last_x"].values + X["base_dx"].values + model_x.predict(X)
    ey = X["last_y"].values + X["base_dy"].values + model_y.predict(X)
    if clf is not None:
        beach = clf.predict_proba(X)[:, 1] >= thr
        ex = np.where(beach, L_X, ex)
    else:
        beach = ex >= L_X
    return np.minimum(ex, L_X), ey, beach


def errors(df, ex, ey):
    return np.hypot(ex - df["end_x"].values, ey - df["end_y"].values)


def summarize(err):
    return dict(mean=float(err.mean()), median=float(np.median(err)), p90=float(np.quantile(err, 0.9)))


def main(data="data", out="model"):
    feats = feature_names()
    train = pd.read_csv(os.path.join(data, "train.csv"))
    test = pd.read_csv(os.path.join(data, "test.csv"))

    # hold out 15% of TRAIN scenarios for conformal calibration
    rng = np.random.default_rng(0)
    sids = train["scenario_id"].unique()
    calib_ids = set(rng.choice(sids, int(0.15 * len(sids)), replace=False))
    is_cal = train["scenario_id"].isin(calib_ids)
    fit_df, cal_df = train[~is_cal], train[is_cal]

    Xf = fit_df[feats]
    rx = fit_df["disp_x"] - fit_df["base_dx"]
    ry = fit_df["disp_y"] - fit_df["base_dy"]
    model_x, model_y = fit_reg().fit(Xf, rx), fit_reg().fit(Xf, ry)
    clf = fit_clf().fit(Xf, fit_df["beached"])

    # 90% radius per horizon (split conformal)
    radius = {}
    ex, ey, _ = predict_end(model_x, model_y, cal_df[feats], clf)
    cal_err = errors(cal_df, ex, ey)
    for h in HORIZONS_MIN:
        e = cal_err[cal_df["horizon_min"].values == h]
        q = min(1.0, math.ceil((len(e) + 1) * COVERAGE) / len(e))
        radius[h] = float(np.quantile(e, q))

    # evaluation on unseen scenarios
    Xt = test[feats]
    ex, ey, beach_pred = predict_end(model_x, model_y, Xt, clf)
    err_model = errors(test, ex, ey)
    base_x = np.minimum(test["last_x"] + test["base_dx"], L_X)
    err_phys = errors(test, base_x.values, (test["last_y"] + test["base_dy"]).values)
    err_stay = errors(test, test["last_x"].values, test["last_y"].values)
    hs = test["horizon_min"].values
    rad = np.interp(hs, list(radius), list(radius.values()))
    true_b = test["beached"].values.astype(bool)
    tp = int((beach_pred & true_b).sum())

    metrics = {
        "overall": {"model": summarize(err_model), "physics_only": summarize(err_phys),
                    "last_seen_only": summarize(err_stay)},
        "coverage_90pct_radius": float((err_model <= rad).mean()),
        "beach": {"accuracy": float((beach_pred == true_b).mean()),
                  "precision": tp / max(int(beach_pred.sum()), 1),
                  "recall": tp / max(int(true_b.sum()), 1),
                  "roc_auc": float(roc_auc_score(true_b, clf.predict_proba(Xt)[:, 1])),
                  "share_of_cases_that_beach": float(true_b.mean())},
        "by_horizon_min": {int(h): {"model_mean_m": float(err_model[hs == h].mean()),
                                    "physics_mean_m": float(err_phys[hs == h].mean()),
                                    "radius90_m": radius[h]} for h in HORIZONS_MIN},
    }

    os.makedirs(out, exist_ok=True)
    joblib.dump({"model_x": model_x, "model_y": model_y, "beach_clf": clf, "features": feats,
                 "radius90": radius}, os.path.join(out, "drift_model.joblib"))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
