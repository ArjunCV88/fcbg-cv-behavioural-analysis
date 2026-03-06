# Computer Vision for Behavioural Analysis 

A computer vision pipeline for automated behavioural analysis from video, demonstrating body pose estimation and facial expression classification using MediaPipe. 

The pipeline processes two separate camera feeds:
- **Camera 1 (Body):** 33-point body pose estimation for movement tracking and activity segmentation
- **Camera 2 (Face):** 468-point facial mesh for expression and head movement classification (7 classes: neutral, smiling, mouth open, eyebrows raised, eyes closed, head turn, head tilt)

## Directory Structure

```
fcbg_demo/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup_env.sh                # Environment setup (conda)
├── setup_env_venv.sh           # Environment setup (venv)
├── notebooks/
│   └── FCBG_CV_Demo.ipynb      # Body pose estimation notebook
├── scripts/
│   └── face_classifier.py      # Facial expression + head movement classifier
├── videos/                     # Input videos (not tracked in git)
├── output/                     # Generated CSVs and annotated videos
│   ├── pose_landmarks.csv
│   ├── facial_features.csv
│   └── combined_behavioural_data.csv
└── figures/                    # Generated analysis figures
    ├── pose_estimation_samples.png
    ├── landmark_trajectories.png
    ├── movement_velocity.png
    ├── behavioural_segmentation.png
    ├── face_mesh_samples.png
    ├── feature_exploration.png
    ├── expression_features.png
    └── combined_behavioural_timeline.png
```

## Setup

### Option A: venv (recommended)

```bash
cd fcbg_demo
bash setup_env_venv.sh
source fcbg-env/bin/activate
```

### Option B: conda

```bash
cd fcbg_demo
bash setup_env.sh
conda activate fcbg-demo
```

### PyTorch (GPU)

Install PyTorch separately for your CUDA version:

```bash
# CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## How to Run

### Body Pose Analysis (notebook)

Place your movement video in `videos/`, then:

```bash
source fcbg-env/bin/activate
jupyter notebook notebooks/FCBG_CV_Demo.ipynb
```

Update `VIDEO_FILENAME` in the notebook to match your video file. Run all cells. Generates pose figures, landmark CSVs, and annotated video.

### Facial Expression & Head Movement Analysis (script)

Place your face video in `videos/`, then:

```bash
source fcbg-env/bin/activate
export TF_USE_LEGACY_KERAS=1
python scripts/face_classifier.py
```

The script extracts facial features (MAR, EAR, eyebrow height, head pose, nose-to-chin geometry), detects behavioural transitions, and classifies each frame into one of 7 expression/movement classes. Generates figures, CSV, and annotated video.

## Sample Outputs

The `figures/` directory contains publication-ready visualisations:

- **pose_estimation_samples.png** — Body landmark overlay examples
- **landmark_trajectories.png** — Body joint trajectories over time
- **movement_velocity.png** — Wrist velocity with activity segmentation
- **behavioural_segmentation.png** — Movement state classification
- **face_mesh_samples.png** — Facial mesh with expression labels (7 classes)
- **feature_exploration.png** — Raw facial features with phase annotations
- **expression_features.png** — Feature traces with classification thresholds
- **combined_behavioural_timeline.png** — Multi-camera behavioural overview

## Author

Arjun Vinayak Chikkankod
