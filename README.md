# SeaYou drift prediction model

Predicts where a person in the water will be after N minutes, from:
- last seen position (drone)
- current speed and direction from a grid of in-water sensor nodes (latest reading plus one from 15 min earlier)
- wind (optional but used)

## Run order
    pip install -r requirements.txt
    python generate_data.py --scenarios 1000 --bodies 6 --out data   # writes data/train.csv, data/test.csv
    python train_model.py                                            # writes model/ and prints metrics
    python predict.py --demo                                         # single prediction plus demo.png

## How it works
1. simulator.py is the ground truth physics: random ocean states (longshore/cross-shore current, eddies, rip currents,
   slow tide/weather variation, wind current, per-body leeway, turbulence). Bodies stop when they reach the beach.
2. Nodes read the true current plus sensor noise. Drone position gets 5 m GPS noise.
3. Physics baseline: interpolate node currents to the last seen point and advect. Gradient boosting learns the
   residual on top of that. Predictions are clipped at the shoreline.
4. Split conformal calibration gives a 90% search radius per horizon.
5. Train and test are split by scenario, so the test set contains ocean states the model has never seen.

## Results on held-out test scenarios (synthetic data)
| Method | Mean error | Median | 90th percentile |
|---|---|---|---|
| Last seen position | 732 m | 565 m | 1550 m |
| Physics only (interpolated current) | 598 m | 340 m | 1512 m |
| Model | 233 m | 159 m | 536 m |

The 90% radius covered 88% of test cases. Beach/no-beach was right 94.6% of the time.

## Important caveats
- These results are on SIMULATED data. Real performance depends on how close the simulator is to your site.
- The model is tied to the node layout in seayou_common.py (5x5 grid, 400 m spacing). Change it to match your
  real nodes and regenerate data and retrain.
- Before real use, replace or supplement the synthetic data with real drifter/GPS-buoy tracks from your beach,
  and compare against an established search and rescue drift tool such as leeway drift models.
