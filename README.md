# Driver Drowsiness Detection

Real-time drowsiness alarm using a webcam. A fine-tuned **MobileNetV3-Small** classifies each eye as open or closed, and an alarm sounds when the eyes stay closed for 2 seconds.

## Results

Tested on people the model never saw during training:

| Test set | Images | Accuracy | Closed-eye recall |
|---|---|---|---|
| MRL Eye (infrared, 7 unseen people) | 12,152 | **95.9%** | **98.9%** |
| CEW (normal camera photos) | 720 | **97.5%** | 97.7% |

## How it works

```
webcam → MediaPipe Face Landmarker → tilt-corrected eye crops (64×64 grayscale)
       → MobileNetV3-Small (open / closed) → eyes closed ≥ 2 s → alarm
```

- **Model:** MobileNetV3-Small pretrained on ImageNet, fine-tuned on 89,744 eye images
- **Data split:** by person, so no one appears in both training and test data (no data leakage)
- **Augmentation (GPU, Kornia):** flips, rotation, shift, zoom, brightness and contrast changes, blur, noise
- **Live app:** smoothed eye-state probability, a closed-eye timer, and an on-screen and audio alert (~30 FPS on a laptop)

## Project structure

| File | What it does |
|---|---|
| `01_prepare_data.ipynb` | Loads both datasets, splits by person, saves 64×64 grayscale arrays |
| `02_train_model.ipynb` | Trains and evaluates the model, saves `models/eye_model.pt` |
| `03_live_detection.ipynb` | Live webcam drowsiness detection |
| `models/eye_model.pt` | Trained model (lets you run the live demo without training) |

## Run it

**1. Install** (Python 3.13, NVIDIA GPU recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

**2. Live demo only:** run `03_live_detection.ipynb` and press **Q** to quit.

**3. Train it yourself:** download the datasets, then run notebooks 01 and 02.

| Dataset | Download | Put it in |
|---|---|---|
| MRL Eye Dataset | [Kaggle](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset) | `data/mrl/mrleyedataset/` |
| CEW (eye images 24×24) | [Official page](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html) | `data/cew/dataset_B_Eye_Images/` |

## Tech stack

PyTorch · torchvision · Kornia · MediaPipe · OpenCV · scikit-learn
