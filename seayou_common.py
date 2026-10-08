"""Shared constants and feature engineering for the SeaYou drift model.

Coordinate system (metres): x runs from open water (x=0) towards the beach
(shoreline at x=L_X), y runs along the beach. A real deployment should replace
the constants below with your local frame (e.g. UTM easting/northing rotated so
the shoreline is vertical) and your real node positions.
"""
import numpy as np
import pandas as pd

L_X = 2000.0          # shoreline position (m)
L_Y = 2000.0          # length of beach segment covered (m)

# 5 x 5 grid of current sensor nodes, 400 m apart
NODE_X = np.arange(200.0, 2000.0, 400.0)
NODE_Y = np.arange(200.0, 2000.0, 400.0)
NODE_XY = np.array([(x, y) for y in NODE_Y for x in NODE_X])
N_NODES = len(NODE_XY)

HORIZONS_MIN = [10, 20, 30, 60, 90, 120]   # prediction horizons used for training
TREND_LAG_S = 900.0                        # nodes are also read 15 min before "last seen"


def idw(pts_xy, node_vals, k=4, power=2.0):
    """Inverse distance weighted current (u, v) at points from the k nearest nodes."""
    d = np.linalg.norm(pts_xy[:, None, :] - NODE_XY[None, :, :], axis=2)
    idx = np.argsort(d, axis=1)[:, :k]
    dk = np.take_along_axis(d, idx, 1)
    w = 1.0 / np.maximum(dk, 1.0) ** power
    w /= w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * node_vals[idx]).sum(axis=1)


def feature_names():
    base = ["horizon_min", "last_x", "last_y", "dist_to_shore",
            "wind_u", "wind_v", "wind_disp_x", "wind_disp_y",
            "idw_u", "idw_v", "idw_du", "idw_dv", "base_dx", "base_dy"]
    now = [f"node{i:02d}_{c}" for i in range(N_NODES) for c in "uv"]
    trend = [f"dnode{i:02d}_{c}" for i in range(N_NODES) for c in "uv"]
    return base + now + trend


def build_features(last_xy, horizon_min, node_now, node_prev, wind_uv):
    """Build the model input table.

    last_xy      (n, 2)   last seen position from the drone (m)
    horizon_min  (n,)     how many minutes ahead to predict
    node_now     (25, 2)  node current readings (u east/x, v north/y, m/s) at last-seen time
    node_prev    (25, 2)  node readings 15 minutes earlier
    wind_uv      (2,)     wind vector (m/s), same frame
    """
    last_xy = np.asarray(last_xy, float).reshape(-1, 2)
    n = len(last_xy)
    T = np.broadcast_to(np.asarray(horizon_min, float), (n,)) * 60.0
    node_now = np.asarray(node_now, float).reshape(N_NODES, 2)
    node_prev = np.asarray(node_prev, float).reshape(N_NODES, 2)
    wind = np.asarray(wind_uv, float).reshape(2)

    cur = idw(last_xy, node_now)
    prev = idw(last_xy, node_prev)
    trend = cur - prev

    cols = [
        T / 60.0, last_xy[:, 0], last_xy[:, 1], L_X - last_xy[:, 0],
        np.full(n, wind[0]), np.full(n, wind[1]),
        wind[0] * T, wind[1] * T,
        cur[:, 0], cur[:, 1], trend[:, 0], trend[:, 1],
        cur[:, 0] * T, cur[:, 1] * T,
    ]
    X = np.column_stack(cols)
    X = np.hstack([X,
                   np.tile(node_now.reshape(-1), (n, 1)),
                   np.tile((node_now - node_prev).reshape(-1), (n, 1))])
    return pd.DataFrame(X, columns=feature_names())
