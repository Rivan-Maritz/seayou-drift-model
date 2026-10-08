"""Physics-based ground truth simulator for drifting persons near a beach.

Used to create labelled training data. Each scenario is a random ocean state:
  - mean longshore + cross-shore current
  - divergence-free eddies (stream function modes)
  - optional rip current (offshore jet with feeder flow)
  - slow temporal variation (tide / weather)
  - wind driven surface current (seen by nodes) plus body leeway (NOT seen by nodes)
Bodies are advected with RK2 plus random-walk turbulence and stop when they reach the shore.
"""
import numpy as np
from seayou_common import L_X, L_Y, NODE_XY, TREND_LAG_S

N_EDDY = 4


def sample_scenario(rng):
    lam = rng.uniform(500, 2500, N_EDDY)
    theta = rng.uniform(0, 2 * np.pi, N_EDDY)
    wind_speed = rng.uniform(0, 12)
    wind_dir = rng.uniform(0, 2 * np.pi)
    return dict(
        cross_u=rng.uniform(-0.12, 0.12),          # + is towards the beach
        long_v=rng.uniform(-0.45, 0.45),
        eddy_vel=rng.uniform(0.02, 0.15, N_EDDY),
        eddy_k=2 * np.pi / lam,
        eddy_theta=theta,
        eddy_phase=rng.uniform(0, 2 * np.pi, N_EDDY),
        rip_on=bool(rng.random() < 0.5),
        rip_y=rng.uniform(300, 1700),
        rip_w=rng.uniform(60, 200),
        rip_strength=rng.uniform(0.2, 0.8),
        rip_len=rng.uniform(300, 900),
        tide_amp=rng.uniform(0.0, 0.3),
        tide_period=rng.uniform(1800, 7200),
        tide_phase=rng.uniform(0, 2 * np.pi),
        wind=np.array([wind_speed * np.cos(wind_dir), wind_speed * np.sin(wind_dir)]),
    )


def current_field(sc, x, y, t):
    """Water current (u, v) in m/s at positions (x, y) and time t (seconds)."""
    u = np.full_like(x, sc["cross_u"], dtype=float)
    v = np.full_like(x, sc["long_v"], dtype=float)
    for k in range(N_EDDY):
        kx = sc["eddy_k"][k] * np.cos(sc["eddy_theta"][k])
        ky = sc["eddy_k"][k] * np.sin(sc["eddy_theta"][k])
        kk = sc["eddy_k"][k]
        c = np.cos(kx * x + ky * y + sc["eddy_phase"][k])
        u += sc["eddy_vel"][k] * (ky / kk) * c
        v -= sc["eddy_vel"][k] * (kx / kk) * c
    if sc["rip_on"]:
        off = L_X - x
        lateral = np.exp(-((y - sc["rip_y"]) / sc["rip_w"]) ** 2)
        env = np.exp(-(np.maximum(off, 0) / sc["rip_len"]) ** 2)
        u -= sc["rip_strength"] * lateral * env
        v -= 0.4 * sc["rip_strength"] * np.tanh((y - sc["rip_y"]) / (3 * sc["rip_w"])) \
            * np.exp(-(np.maximum(off, 0) / (0.5 * sc["rip_len"])) ** 2)
    f = 1.0 + sc["tide_amp"] * np.sin(2 * np.pi * t / sc["tide_period"] + sc["tide_phase"])
    u = u * f + 0.01 * sc["wind"][0]     # wind driven surface current (visible to nodes)
    v = v * f + 0.01 * sc["wind"][1]
    return u, v


def read_nodes(sc, t, rng, noise=0.02):
    """Noisy node readings at time t, shape (n_nodes, 2)."""
    u, v = current_field(sc, NODE_XY[:, 0].copy(), NODE_XY[:, 1].copy(), t)
    return np.column_stack([u, v]) + rng.normal(0, noise, (len(u), 2))


def simulate(sc, x0, y0, t0, horizons_s, rng, dt=10.0, diffusivity=0.15):
    """Advect bodies from (x0, y0) starting at time t0. Returns {horizon_s: (x, y, beached)}."""
    x = np.array(x0, float)
    y = np.array(y0, float)
    n = len(x)
    leeway = rng.uniform(0.01, 0.03, n)            # hidden per-body wind drift fraction
    beached = np.zeros(n, bool)
    want = {int(round(h / dt)): h for h in horizons_s}
    out = {}
    sig = np.sqrt(2 * diffusivity * dt)
    for i in range(1, max(want) + 1):
        t = t0 + (i - 1) * dt
        u, v = current_field(sc, x, y, t)
        um, vm = current_field(sc, x + 0.5 * dt * u, y + 0.5 * dt * v, t + 0.5 * dt)
        dx = (um + leeway * sc["wind"][0]) * dt + sig * rng.standard_normal(n)
        dy = (vm + leeway * sc["wind"][1]) * dt + sig * rng.standard_normal(n)
        act = ~beached
        x = np.where(act, x + dx, x)
        y = np.where(act, np.clip(y + dy, 0, L_Y), y)
        hit = x >= L_X
        beached |= hit
        x = np.minimum(x, L_X)
        if i in want:
            out[want[i]] = (x.copy(), y.copy(), beached.copy())
    return out


def prev_time(t0):
    return t0 - TREND_LAG_S
