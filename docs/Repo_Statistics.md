```bash
find . -path './.venv' -prune -o -type f -name '*.py' -not -path '*__pycache__*' -print0 | xargs -0 wc -l
    99 ./MLDailyHourClock.py
    99 ./prepare_dataset.py
   215 ./src/baseline_advanced.py
   125 ./src/baseline_advanced_logreg.py
    97 ./src/baseline_rgb.py
   340 ./src/camera_hour_overlay.py
   332 ./src/camera_hour_overlay_advanced.py
   365 ./src/camera_hour_overlay_mlp.py
   581 ./src/camera_hour_overlay_mlp_rpi.py
   242 ./src/camera_hour_overlay_rpi.py
    89 ./src/explore_data.py
   105 ./src/load_data.py
   107 ./src/normalize_data.py
    92 ./src/precompute_features_advanced.py
    76 ./src/precompute_features_robust.py
    75 ./src/precompute_mean_rgb.py
    84 ./src/predict_hour.py
   174 ./src/rebuild_labels.py
    15 ./src/settings.py
   256 ./src/train_hour_cnn.py
   297 ./src/train_hour_nn_cyclic.py
   397 ./src/train_hour_nn_cyclic_2.py
   232 ./src/train_hour_regression_cyclic.py
   208 ./src/train_robust_time.py
    81 ./src/utils.py
   149 ./src/utils_robust.py
  4932 total
  ```