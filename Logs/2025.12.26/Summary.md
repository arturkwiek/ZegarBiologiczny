Z logów z tego biegu pipeline’u (Logs/2025.12.26) wychodzi:

precompute_mean_rgb: ok. 4:31:57 → 16 317 s
precompute_features_advanced: ok. 11:10:55 → 40 255 s
precompute_features_robust: ok. 22:38:59 → 81 539 s
baseline_rgb (trening): ok. 0:02:20 → 140 s
baseline_advanced (trening): ok. 1:33:14 → 5 594 s
train_robust_time: ok. 1:57:55 → 7 075 s
train_hour_regression_cyclic: 2 026.71 s
train_hour_nn_cyclic: 525.93 s
Sumarycznie dla całego pipeline’u z tego dnia:
~153 473 s ≈ 2 558 min ≈ 42.6 godziny czasu obliczeń (według tego, co skrypty same raportują w logach).

Jeśli chcesz, mogę to samo policzyć tylko dla “czystego trenowania” (pomijając precompute_*).