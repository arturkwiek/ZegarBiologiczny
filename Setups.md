# Wymagania środowiskowe

sudo apt update

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r r
pip install opencv-python numpy

sudo apt install -y libgl1 libglib2.0-0 libxcb1 libx11-6 libxext6 libxrender1 libxfixes3 libxi6 libxkbcommon0

sudo apt install -y libxcb-xinerama0 libxcb-randr0 libxcb-render0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-util1 libxcb-xkb1

python src\camera_hour_overlay.py --model models\baseline_rgb_model.pkl --cam 0
python src/camera_hour_overlay_advanced.py   --model models/baseline_advanced_logreg_model.pkl   --features_csv features_advanced.csv   --cam 0

