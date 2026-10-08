"""Generate synthetic train/test data for the SeaYou drift model.

Usage: python generate_data.py --scenarios 1000 --bodies 6 --out data
Train and test are split BY SCENARIO so the test set contains ocean states the model never saw.
"""
import argparse, os
import numpy as np
import pandas as pd
from seayou_common import L_X, L_Y, HORIZONS_MIN, build_features
from simulator import sample_scenario, read_nodes, simulate, prev_time

GPS_NOISE = 5.0      # drone last-seen position error (m)
WIND_NOISE = 0.5     # wind measurement error (m/s)


def make_scenario_rows(sid, rng, n_bodies):
    sc = sample_scenario(rng)
    t0 = rng.uniform(1000, 6 * 3600)
    node_now = read_nodes(sc, t0, rng)
    node_prev = read_nodes(sc, prev_time(t0), rng)
    x0 = rng.uniform(100, L_X - 200, n_bodies)
    y0 = rng.uniform(100, L_Y - 100, n_bodies)
    hs = [m * 60 for m in HORIZONS_MIN]
    res = simulate(sc, x0, y0, t0, hs, rng)

    nh = len(HORIZONS_MIN)
    last_true = np.column_stack([x0, y0])
    last_obs = last_true + rng.normal(0, GPS_NOISE, last_true.shape)
    last_rows = np.repeat(last_obs, nh, axis=0)
    hor_rows = np.tile(HORIZONS_MIN, n_bodies)
    wind_obs = sc["wind"] + rng.normal(0, WIND_NOISE, 2)

    X = build_features(last_rows, hor_rows, node_now, node_prev, wind_obs)
    end_x = np.stack([res[h][0] for h in hs], axis=1).reshape(-1)
    end_y = np.stack([res[h][1] for h in hs], axis=1).reshape(-1)
    beached = np.stack([res[h][2] for h in hs], axis=1).reshape(-1)
    meta = pd.DataFrame({
        "scenario_id": sid,
        "body_id": np.repeat(np.arange(n_bodies), nh),
        "end_x": end_x, "end_y": end_y,
        "disp_x": end_x - last_rows[:, 0], "disp_y": end_y - last_rows[:, 1],
        "beached": beached.astype(int),
    })
    return pd.concat([meta, X], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=1000)
    ap.add_argument("--bodies", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    df = pd.concat([make_scenario_rows(s, rng, a.bodies) for s in range(a.scenarios)], ignore_index=True)

    ids = rng.permutation(a.scenarios)
    n_test = int(a.scenarios * a.test_frac)
    test_ids = set(ids[:n_test])
    is_test = df["scenario_id"].isin(test_ids)

    os.makedirs(a.out, exist_ok=True)
    df = df.round(4)
    df[~is_test].to_csv(os.path.join(a.out, "train.csv"), index=False)
    df[is_test].to_csv(os.path.join(a.out, "test.csv"), index=False)
    print(f"train rows: {(~is_test).sum()}  test rows: {is_test.sum()}  "
          f"beached share: {df['beached'].mean():.2%}")


if __name__ == "__main__":
    main()
