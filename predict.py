"""Use the trained SeaYou model.

  python predict.py --demo      simulate a fresh unseen scenario, predict, and save demo.png

In production call predict_landing() with the drone's last seen position and the latest node readings.
"""
import argparse
import joblib
import numpy as np
from seayou_common import L_X, NODE_XY, build_features
from simulator import sample_scenario, read_nodes, simulate, prev_time


def predict_landing(bundle, last_xy, horizon_min, node_now, node_prev, wind_uv):
    """Return predicted end position (m), whether it reaches the beach, and a 90% search radius (m).

    node_now / node_prev are (25, 2) arrays of (u, v) in m/s: latest readings and readings 15 min earlier.
    """
    X = build_features([last_xy], [horizon_min], node_now, node_prev, wind_uv)[bundle["features"]]
    ex = X["last_x"].values + X["base_dx"].values + bundle["model_x"].predict(X)
    ey = X["last_y"].values + X["base_dy"].values + bundle["model_y"].predict(X)
    hs = sorted(bundle["radius90"])
    radius = float(np.interp(horizon_min, hs, [bundle["radius90"][h] for h in hs]))
    return {"x": float(min(ex[0], L_X)), "y": float(ey[0]), "reaches_beach": bool(ex[0] >= L_X),
            "search_radius_90_m": radius}


def demo(seed=7, horizon_min=60):
    bundle = joblib.load("model/drift_model.joblib")
    rng = np.random.default_rng(seed)
    sc = sample_scenario(rng)
    t0 = 3 * 3600.0
    node_now, node_prev = read_nodes(sc, t0, rng), read_nodes(sc, prev_time(t0), rng)
    start = np.array([900.0, 1100.0])
    truth = simulate(sc, [start[0]], [start[1]], t0, [horizon_min * 60], rng)[horizon_min * 60]
    true_end = np.array([truth[0][0], truth[1][0]])
    pred = predict_landing(bundle, start, horizon_min, node_now, node_prev, sc["wind"])
    err = float(np.hypot(pred["x"] - true_end[0], pred["y"] - true_end[1]))
    print(f"start {start}, true end {true_end.round(1)}, predicted ({pred['x']:.1f}, {pred['y']:.1f}), "
          f"error {err:.1f} m, 90% radius {pred['search_radius_90_m']:.1f} m")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.quiver(NODE_XY[:, 0], NODE_XY[:, 1], node_now[:, 0], node_now[:, 1], color="tab:blue", label="node current")
        ax.axvline(L_X, color="tan", lw=6, label="beach")
        ax.plot(*start, "ko", label="last seen (drone)")
        ax.plot(*true_end, "g*", ms=14, label="true position")
        ax.plot(pred["x"], pred["y"], "rx", ms=12, mew=3, label="predicted")
        ax.add_patch(plt.Circle((pred["x"], pred["y"]), pred["search_radius_90_m"], fill=False, color="r", ls="--"))
        ax.set_xlim(0, 2100); ax.set_ylim(0, 2000); ax.set_aspect("equal")
        ax.set_xlabel("x, offshore to beach (m)"); ax.set_ylabel("y, along beach (m)")
        ax.set_title(f"{horizon_min} min drift prediction"); ax.legend(loc="lower left", fontsize=8)
        fig.savefig("demo.png", dpi=130, bbox_inches="tight")
    except ImportError:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--horizon", type=int, default=60)
    a = ap.parse_args()
    demo(horizon_min=a.horizon)
