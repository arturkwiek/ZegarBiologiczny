(.venv) vision@Vision-DellLaPrv:/mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny$ python3 -m src.train_hour_cnn --epochs 10 --batch-size 64 --img-size 224
[INFO] Używane urządzenie: cuda
[INFO] Wczytuję etykiety z: dataset/2025/labels.csv
[INFO] Katalog z danymi: dataset/2025
[INFO] Liczba rekordów w labels.csv: 471095
[INFO] Podział: train=423986, val=47109
[EPOCH 001] train_loss=1.7541 train_acc=0.351 val_loss=1.3234 val_acc=0.501 time=3519.5s
[INFO] Nowy najlepszy model (val_loss=1.3234)
[EPOCH 002] train_loss=1.1886 train_acc=0.545 val_loss=1.0480 val_acc=0.591 time=16192.3s
[INFO] Nowy najlepszy model (val_loss=1.0480)
[EPOCH 003] train_loss=1.0053 train_acc=0.611 val_loss=0.9199 val_acc=0.644 time=1747.2s
[INFO] Nowy najlepszy model (val_loss=0.9199)
[EPOCH 004] train_loss=0.9208 train_acc=0.642 val_loss=0.9039 val_acc=0.638 time=2621.9s
[INFO] Nowy najlepszy model (val_loss=0.9039)
[EPOCH 005] train_loss=0.8578 train_acc=0.665 val_loss=0.8495 val_acc=0.668 time=4669.9s
[INFO] Nowy najlepszy model (val_loss=0.8495)
[EPOCH 006] train_loss=0.8120 train_acc=0.681 val_loss=0.7882 val_acc=0.688 time=1794.3s
[INFO] Nowy najlepszy model (val_loss=0.7882)
[EPOCH 007] train_loss=0.7783 train_acc=0.692 val_loss=0.7585 val_acc=0.695 time=1820.0s
[INFO] Nowy najlepszy model (val_loss=0.7585)
[EPOCH 008] train_loss=0.7482 train_acc=0.703 val_loss=0.7662 val_acc=0.697 time=28951.8s
[EPOCH 009] train_loss=0.7245 train_acc=0.709 val_loss=0.7142 val_acc=0.711 time=6519.7s
[INFO] Nowy najlepszy model (val_loss=0.7142)
[EPOCH 010] train_loss=0.7038 train_acc=0.716 val_loss=0.6829 val_acc=0.725 time=6384.4s
[INFO] Nowy najlepszy model (val_loss=0.6829)
[INFO] Zapisano model CNN do: /mnt/c/Users/Dell/Desktop/Workplace/Repositories/ZegarBiologiczny/models/pc/best_cnn_hour.pt
[INFO] Całkowity czas trenowania: 74276.15s
