python3 -m src.baseline_advanced
2025-12-29 12:45:54,378 INFO: Rozpoczynam wczytywanie cech...
2025-12-29 12:45:54,379 INFO: Wczytywanie cech z pliku CSV...
2025-12-29 12:45:55,577 INFO: Wczytano cechy rozszerzone: (471031, 11)
2025-12-29 12:45:55,580 WARNING: Brak kolumn ['h_mean', 'h_std'] w features_advanced.csv – trenowanie tylko na: ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std', 's_mean', 'v_mean']
2025-12-29 12:45:55,608 INFO: Cechy wczytane.
2025-12-29 12:45:55,608 INFO: Podział na zbiór treningowy i testowy...
2025-12-29 12:45:55,729 INFO: Rozmiar zbioru treningowego: (329721, 8), testowego: (141310, 8)
2025-12-29 12:45:55,735 INFO: Trenowanie modelu: logreg

=== Trenowanie modelu: logreg ===
/mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny/.venv/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:1184: FutureWarning: 'n_jobs' has no effect since 1.8 and will be removed in 1.10. You provided 'n_jobs=-1', please leave it unspecified.
  warnings.warn(msg, category=FutureWarning)
2025-12-29 12:48:35,301 INFO: Model logreg wytrenowany.
Model logreg zapisano do pliku: /mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny/models/pc/baseline_advanced_logreg_model.pkl
2025-12-29 12:48:35,347 INFO: Predykcja zakończona dla modelu: logreg
Accuracy (logreg): 0.3361
2025-12-29 12:48:35,350 INFO: Trenowanie modelu: knn

=== Trenowanie modelu: knn ===
2025-12-29 12:48:35,987 INFO: Model knn wytrenowany.
Model knn zapisano do pliku: /mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny/models/pc/baseline_advanced_knn_model.pkl
2025-12-29 12:48:38,762 INFO: Predykcja zakończona dla modelu: knn
Accuracy (knn): 0.7657
2025-12-29 12:48:38,765 INFO: Trenowanie modelu: rf

=== Trenowanie modelu: rf ===
2025-12-29 12:48:43,575 INFO: Model rf wytrenowany.
Model rf zapisano do pliku: /mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny/models/pc/baseline_advanced_rf_model.pkl
2025-12-29 12:48:44,115 INFO: Predykcja zakończona dla modelu: rf
Accuracy (rf): 0.7441
2025-12-29 12:48:44,117 INFO: Trenowanie modelu: gb

=== Trenowanie modelu: gb ===
2025-12-29 13:25:30,347 INFO: Model gb wytrenowany.
Model gb zapisano do pliku: /mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny/models/pc/baseline_advanced_gb_model.pkl
2025-12-29 13:25:35,181 INFO: Predykcja zakończona dla modelu: gb
Accuracy (gb): 0.7018
2025-12-29 13:25:35,183 INFO: Najlepszy model: knn (accuracy = 0.7657)

################################################################################
Najlepszy model: knn (accuracy = 0.7657)
################################################################################

Pełny raport dla najlepszego modelu:

Classification report:
              precision    recall  f1-score   support

           0       0.55      0.53      0.54      5370
           1       0.37      0.43      0.40      5370
           2       0.31      0.36      0.33      5372
           3       0.25      0.26      0.26      5372
           4       0.27      0.28      0.27      5371
           5       0.46      0.42      0.44      5371
           6       0.88      0.76      0.81      5371
           7       0.99      0.99      0.99      5368
           8       0.99      1.00      0.99      5366
           9       0.99      0.99      0.99      5085
          10       0.99      0.99      0.99      5190
          11       0.99      0.99      0.99      5364
          12       0.99      0.99      0.99      5696
          13       0.99      0.99      0.99      6420
          14       0.99      0.99      0.99      6244
          15       1.00      0.99      0.99      6440
          16       0.89      0.88      0.88      7216
          17       0.75      0.79      0.77      7519
          18       0.78      0.79      0.79      7519
          19       0.78      0.71      0.75      7519
          20       0.81      0.83      0.82      6656
          21       0.84      0.84      0.84      5371
          22       0.84      0.80      0.82      5370
          23       0.64      0.59      0.61      5370

    accuracy                           0.77    141310
   macro avg       0.76      0.76      0.76    141310
weighted avg       0.77      0.77      0.77    141310

Macierz pomyłek:
[[2870  997  485  240  228  100   25    0    0    0    0    0    0    0
     0    0   11    0    0   33   47   35   19  280]
 [ 844 2310  860  515  343  104   21    0    0    0    0    0    0    0
     0    0    0    1    0   42    4   14   21  291]
 [ 354  903 1911  923  587  335   30    0    0    0    0    0    0    0
     0    0    0    0    0  112   64    2   45  106]
 [ 246  667 1033 1419 1121  473   47    0    0    0    0    0    0    0
     0    0    0    0    0   82   67    0   21  196]
 [ 271  461  722 1221 1480  918   65    0    0    0    0    0    0    0
     0    0    0    0    0   15    7    0   15  196]
 [ 157  189  512  699 1092 2273  318    0    0    0    0    0    0    0
     0    0    0    0    1   39    0    0    3   88]
 [  54  107  124  167  233  571 4060    4    0    0    0    0    0    0
     0    0    0    1    0    4    6    0    2   38]
 [   0    0    0    0    0    0    4 5341   11    0    0    1    0    0
    10    1    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0   16 5341    8    0    0    0    1
     0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0   13 5059   12    0    1    0
     0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    6   11 5157   13    2    1
     0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0    5   24 5317   15    3
     0    0    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    5    0    0    3    0   22 5631   23
     8    0    4    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    1    0    0   11    1    5   38 6340
    21    0    2    0    0    0    0    0    1    0]
 [   0    0    0    0    0    0    0    3    0    0    0   12    1   20
  6200    8    0    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0   10   10    0    0    0    5    0    6
    18 6389    2    0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    5    0    0    0    0    0    1    0
     0    7 6330  703  124   17    8   19    2    0]
 [   1    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0  705 5912  773  103    9   13    0    3]
 [   1    0    0    0    1    0    0    0    0    0    0    0    0    0
     0    0   37 1016 5975  464    3   13    1    8]
 [  41  112  206   70   35   75    1    0    0    0    0    0    0    0
     0    0    1  180  674 5361  654   90   10    9]
 [   6   22   84   68   11    4    0    0    0    0    0    0    0    0
     0    0    5   34   16  450 5511  386   33   26]
 [  23   25    8    0    0    0    2    0    0    0    0    0    0    0
     0    0   16    9   14   89  402 4524  247   12]
 [  10   32   74   27   31   13    0    0    0    0    0    0    0    0
     0    0    0    0    0    8   32  269 4320  554]
 [ 381  501  114  262  272  115   18    0    0    0    0    0    0    0
     0    0    0    0   76   16   26   35  379 3175]]
Czas trenowania: 0:39:40
(.venv) vision@Vision-DellLaPrv:/mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny$