"""GPS in, GPS out interface for the SeaYou drift model.

    from drift_api import GeoFrame, load_model, predict_drift

    frame = GeoFrame(origin_lat=-29.85, origin_lon=31.05, onshore_bearing_deg=270)
    result = predict_drift(frame, last_seen_lat=-29.851, last_seen_lon=31.043,
                           node_readings=[{"node_id": 0, "speed_ms": 0.31, "direction_deg": 95}, ...],
                           horizon_min=60)
    print(result["latitude"], result["longitude"], result["search_radius_m"])

CONVENTIONS (please read)
  * Node direction_deg is the compass bearing the water is flowing TOWARDS (0 = north, 90 = east).
  * Wind direction is the usual weather convention: the bearing the wind blows FROM.
  * GeoFrame ties GPS to the model's local grid:
      origin_lat / origin_lon  = the open-water corner of the grid (local x=0, y=0)
      onshore_bearing_deg      = compass bearing you would sail from open water straight towards the beach
    The grid is 2000 m (offshore to beach) by 2000 m (along the beach), with the shoreline at x = 2000 m.
    Looking from open water towards the beach, local y increases to your LEFT.
  * Nodes are numbered 0-24 in the order of seayou_common.NODE_XY (see node_layout_latlon()).
"""
import math
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
from seayou_common import L_X, L_Y, NODE_XY, N_NODES, HORIZONS_MIN
from predict import predict_landing

HERE = Path(__file__).resolve().parent
MIN_NODES = 3


@dataclass(frozen=True)
class GeoFrame:
    origin_lat: float
    origin_lon: float
    onshore_bearing_deg: float

    def _m_per_deg(self):
        p = math.radians(self.origin_lat)
        m_lat = 111132.954 - 559.822 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
        m_lon = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p) + 0.118 * math.cos(5 * p)
        return m_lat, m_lon

    def en_to_xy(self, east, north):
        b = math.radians(self.onshore_bearing_deg)
        return east * math.sin(b) + north * math.cos(b), -east * math.cos(b) + north * math.sin(b)

    def xy_to_en(self, x, y):
        b = math.radians(self.onshore_bearing_deg)
        return x * math.sin(b) - y * math.cos(b), x * math.cos(b) + y * math.sin(b)

    def to_local(self, lat, lon):
        m_lat, m_lon = self._m_per_deg()
        return self.en_to_xy((lon - self.origin_lon) * m_lon, (lat - self.origin_lat) * m_lat)

    def to_latlon(self, x, y):
        m_lat, m_lon = self._m_per_deg()
        east, north = self.xy_to_en(x, y)
        return float(self.origin_lat + north / m_lat), float(self.origin_lon + east / m_lon)


def load_model(path=None):
    candidates = [Path(path)] if path else [HERE / "model" / "drift_model.joblib", HERE / "drift_model.joblib"]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("drift_model.joblib not found. Run train_model.py first.")


def node_layout_latlon(frame):
    """Where each of the 25 nodes should be placed, so readings match the model's layout."""
    return [{"node_id": i, "latitude": frame.to_latlon(x, y)[0], "longitude": frame.to_latlon(x, y)[1]}
            for i, (x, y) in enumerate(NODE_XY)]


def _readings_to_local(frame, readings, name):
    """List of {node_id, speed_ms, direction_deg} -> (25, 2) local (u, v). Missing nodes are interpolated."""
    arr = np.full((N_NODES, 2), np.nan)
    for r in readings:
        i = int(r["node_id"])
        if not 0 <= i < N_NODES:
            raise ValueError(f"{name}: node_id {i} is outside 0..{N_NODES - 1}")
        d = math.radians(r["direction_deg"])
        arr[i] = frame.en_to_xy(r["speed_ms"] * math.sin(d), r["speed_ms"] * math.cos(d))
    ok = ~np.isnan(arr[:, 0])
    if ok.sum() < MIN_NODES:
        raise ValueError(f"{name}: need readings from at least {MIN_NODES} nodes, got {int(ok.sum())}")
    for i in np.where(~ok)[0]:      # inverse distance fill for dead or missing nodes
        d = np.linalg.norm(NODE_XY[ok] - NODE_XY[i], axis=1)
        w = 1.0 / d ** 2
        arr[i] = (w[:, None] * arr[ok]).sum(0) / w.sum()
    return arr, int((~ok).sum())


def predict_drift(frame, last_seen_lat, last_seen_lon, node_readings, horizon_min=60,
                  prev_node_readings=None, wind_speed_ms=0.0, wind_from_deg=0.0, bundle=None):
    """Predict where a person last seen at (lat, lon) will be after horizon_min minutes.

    node_readings       latest node data, list of {"node_id", "speed_ms", "direction_deg"}
    prev_node_readings  same format, from about 15 min earlier (improves accuracy; if omitted, no trend is used)
    wind_speed_ms / wind_from_deg  wind at the scene; if unknown leave at 0 (accuracy drops in windy conditions)
    Returns a dict with latitude, longitude, search_radius_m (90% confidence), reaches_beach, beach_probability,
    drift_distance_m, drift_bearing_deg and a list of warnings.
    """
    bundle = bundle or load_model()
    warnings = []
    node_now, n_missing = _readings_to_local(frame, node_readings, "node_readings")
    if prev_node_readings is None:
        node_prev = node_now
        warnings.append("No earlier node readings given, so current trends were not used.")
    else:
        node_prev, _ = _readings_to_local(frame, prev_node_readings, "prev_node_readings")
    if n_missing:
        warnings.append(f"{n_missing} node(s) had no reading and were interpolated.")

    x0, y0 = frame.to_local(last_seen_lat, last_seen_lon)
    if not (0 <= x0 <= L_X and 0 <= y0 <= L_Y):
        warnings.append("Last seen position is outside the area covered by the node grid, so the result is unreliable.")
    if not (min(HORIZONS_MIN) <= horizon_min <= max(HORIZONS_MIN)):
        warnings.append(f"Horizon is outside the trained range of {min(HORIZONS_MIN)} to {max(HORIZONS_MIN)} minutes.")

    to_deg = math.radians((wind_from_deg + 180.0) % 360.0)       # direction the wind blows towards
    wind_xy = frame.en_to_xy(wind_speed_ms * math.sin(to_deg), wind_speed_ms * math.cos(to_deg))
    if wind_speed_ms == 0:
        warnings.append("Wind not provided, assumed calm.")

    p = predict_landing(bundle, [x0, y0], horizon_min, node_now, node_prev, wind_xy)
    lat, lon = frame.to_latlon(p["x"], p["y"])
    dx, dy = p["x"] - x0, p["y"] - y0
    east, north = frame.xy_to_en(dx, dy)
    return {
        "latitude": lat, "longitude": lon, "horizon_min": horizon_min,
        "search_radius_m": p["search_radius_90_m"],
        "reaches_beach": p["reaches_beach"], "beach_probability": p["beach_probability"],
        "drift_distance_m": math.hypot(dx, dy),
        "drift_bearing_deg": math.degrees(math.atan2(east, north)) % 360.0,
        "last_seen": {"latitude": last_seen_lat, "longitude": last_seen_lon},
        "warnings": warnings,
    }


def predict_drift_track(frame, last_seen_lat, last_seen_lon, node_readings, **kwargs):
    """Predictions at every trained horizon (10, 20, 30, 60, 90, 120 min), e.g. to draw a drift path."""
    bundle = kwargs.pop("bundle", None) or load_model()
    return [predict_drift(frame, last_seen_lat, last_seen_lon, node_readings, horizon_min=h,
                          bundle=bundle, **kwargs) for h in HORIZONS_MIN]


def _distance_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _demo():
    """End to end check: simulate a scenario, hand the model GPS and compass style data, compare to truth."""
    import json
    from simulator import sample_scenario, read_nodes, simulate, prev_time
    frame = GeoFrame(origin_lat=-29.85, origin_lon=31.05, onshore_bearing_deg=270)   # example location only
    rng = np.random.default_rng(11)
    sc = sample_scenario(rng)
    t0 = 3 * 3600.0

    def to_readings(arr):
        out = []
        for i, (u, v) in enumerate(arr):
            e, n = frame.xy_to_en(u, v)
            out.append({"node_id": i, "speed_ms": math.hypot(e, n),
                        "direction_deg": math.degrees(math.atan2(e, n)) % 360.0})
        return out

    now, prev = to_readings(read_nodes(sc, t0, rng)), to_readings(read_nodes(sc, prev_time(t0), rng))
    we, wn = frame.xy_to_en(*sc["wind"])
    wind_speed = math.hypot(we, wn)
    wind_from = (math.degrees(math.atan2(we, wn)) + 180.0) % 360.0

    start_xy = (900.0, 1100.0)
    start_lat, start_lon = frame.to_latlon(*start_xy)
    truth = simulate(sc, [start_xy[0]], [start_xy[1]], t0, [3600], rng)[3600]
    true_lat, true_lon = frame.to_latlon(truth[0][0], truth[1][0])

    res = predict_drift(frame, start_lat, start_lon, now, horizon_min=60, prev_node_readings=prev,
                        wind_speed_ms=wind_speed, wind_from_deg=wind_from)
    print(json.dumps(res, indent=2))
    print(f"\nTrue position after 60 min: {true_lat:.6f}, {true_lon:.6f}")
    print(f"Prediction error: {_distance_m(res['latitude'], res['longitude'], true_lat, true_lon):.0f} m "
          f"(90% search radius {res['search_radius_m']:.0f} m)")
    print("\nRound trip check of GPS conversion:", np.round(frame.to_local(*frame.to_latlon(*start_xy)), 3))


if __name__ == "__main__":
    _demo()
