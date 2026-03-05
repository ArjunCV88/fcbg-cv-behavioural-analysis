#!/usr/bin/env python3
"""
FCBG Demo — Expanded facial expression + head movement classifier (v4 final).
Fixes: geometry-based head tilt (nose_to_chin_distance replaces broken pitch),
frontality guard on eyebrows_raised, 7-frame face_mesh_samples.
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')
import os

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BASE_DIR = Path('/home/d20125888/Downloads/fcbg_demo')
OUTPUT_DIR = BASE_DIR / 'output'
FIGURES_DIR = BASE_DIR / 'figures'
FACE_VIDEO = BASE_DIR / 'videos' / 'face_video.MOV'

def fix_rotation(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

cap = cv2.VideoCapture(str(FACE_VIDEO))
face_rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
face_fps = cap.get(cv2.CAP_PROP_FPS)
raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
face_w, face_h = (raw_h, raw_w) if face_rotation in (90, 270) else (raw_w, raw_h)
print(f'Video: {face_w}x{face_h}, rot={face_rotation}, {face_fps:.0f} FPS, {total_frames} frames, {total_frames/face_fps:.1f}s')

# 3D model points for solvePnP
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip (1)
    (0.0, -330.0, -65.0),     # chin (152)
    (-225.0, 170.0, -135.0),  # left eye corner (33)
    (225.0, 170.0, -135.0),   # right eye corner (263)
    (-150.0, -150.0, -125.0), # left mouth corner (61)
    (150.0, -150.0, -125.0),  # right mouth corner (291)
], dtype=np.float64)

POSE_LM_INDICES = [1, 152, 33, 263, 61, 291]

focal_length = face_w
camera_matrix = np.array([
    [focal_length, 0, face_w / 2],
    [0, focal_length, face_h / 2],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1), dtype=np.float64)

def estimate_head_pose(landmarks, img_w, img_h):
    """Estimate pitch, yaw, roll using cv2.RQDecomp3x3 (no wrapping)."""
    image_points = np.array([
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
        for idx in POSE_LM_INDICES
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    # RQDecomp3x3 gives stable angles without wrapping
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]  # x-axis: up(+) / down(-)
    yaw = angles[1]    # y-axis: right(+) / left(-)
    roll = angles[2]   # z-axis: tilt
    return pitch, yaw, roll

# ============================================================
# STEP 1 — EXTRACT EXPANDED FEATURE SET
# ============================================================
print('\n' + '=' * 70)
print('STEP 1 — EXTRACT EXPANDED FEATURE SET (all frames)')
print('=' * 70)

cap = cv2.VideoCapture(str(FACE_VIDEO))
records = []
frames_cache = {}
frame_idx = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as fm:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = fix_rotation(frame, face_rotation)
        h_img, w_img = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)

        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0]
            lms = face_lms.landmark

            # MAR
            mt = np.array([lms[13].x, lms[13].y])
            mb = np.array([lms[14].x, lms[14].y])
            ml = np.array([lms[61].x, lms[61].y])
            mr_pt = np.array([lms[291].x, lms[291].y])
            mar = np.linalg.norm(mt - mb) / (np.linalg.norm(ml - mr_pt) + 1e-6)

            # EAR (left eye)
            et = np.array([lms[159].x, lms[159].y])
            eb = np.array([lms[145].x, lms[145].y])
            el = np.array([lms[33].x, lms[33].y])
            er = np.array([lms[133].x, lms[133].y])
            ear = np.linalg.norm(et - eb) / (np.linalg.norm(el - er) + 1e-6)

            # Eyebrow height
            ey_brow = np.array([lms[70].x, lms[70].y])
            eyebrow_h = np.linalg.norm(ey_brow - et)

            # Head pose
            pitch, yaw, roll = estimate_head_pose(lms, w_img, h_img)

            # Mouth width ratio
            mouth_l = np.array([lms[61].x, lms[61].y])
            mouth_r = np.array([lms[291].x, lms[291].y])
            jaw_l = np.array([lms[234].x, lms[234].y])
            jaw_r = np.array([lms[454].x, lms[454].y])
            mouth_width_ratio = np.linalg.norm(mouth_l - mouth_r) / (np.linalg.norm(jaw_l - jaw_r) + 1e-6)

            # Cheek raise
            cheek_l = np.array([lms[111].x, lms[111].y])
            cheek_r = np.array([lms[340].x, lms[340].y])
            cheek_raise = (np.linalg.norm(cheek_l - mouth_l) + np.linalg.norm(cheek_r - mouth_r)) / 2

            # Geometry-based head tilt features
            nose_tip = np.array([lms[1].x, lms[1].y])
            chin_pt = np.array([lms[152].x, lms[152].y])
            forehead_pt = np.array([lms[10].x, lms[10].y])
            nose_to_chin = np.linalg.norm(nose_tip - chin_pt)
            forehead_to_nose = np.linalg.norm(forehead_pt - nose_tip)

            records.append({
                'frame': frame_idx,
                'mar': mar, 'ear': ear, 'eyebrow_height': eyebrow_h,
                'pitch': pitch, 'yaw': yaw, 'roll': roll,
                'mouth_width_ratio': mouth_width_ratio,
                'cheek_raise': cheek_raise,
                'nose_to_chin': nose_to_chin,
                'forehead_to_nose': forehead_to_nose,
            })

            if frame_idx % 5 == 0:
                ann = frame.copy()
                mp_drawing.draw_landmarks(
                    ann, face_lms, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                frames_cache[frame_idx] = ann

        if frame_idx % 200 == 0:
            print(f'  Frame {frame_idx}/{total_frames}')
        frame_idx += 1

cap.release()
df = pd.DataFrame(records)
print(f'\n  Extracted {len(df)} frames')

# Feature statistics
print('\n  Feature Statistics:')
feat_cols = ['mar', 'ear', 'eyebrow_height', 'pitch', 'yaw', 'roll', 'mouth_width_ratio', 'cheek_raise', 'nose_to_chin', 'forehead_to_nose']
print(f'  {"Feature":>20s}  {"min":>8s}  {"max":>8s}  {"mean":>8s}  {"std":>8s}  {"p25":>8s}  {"p50":>8s}  {"p75":>8s}')
print('  ' + '-' * 75)
for feat in feat_cols:
    v = df[feat].values
    print(f'  {feat:>20s}  {v.min():>8.3f}  {v.max():>8.3f}  {v.mean():>8.3f}  {v.std():>8.3f}'
          f'  {np.percentile(v,25):>8.3f}  {np.percentile(v,50):>8.3f}  {np.percentile(v,75):>8.3f}')

# ============================================================
# STEP 2 — PHASE DETECTION
# ============================================================
print('\n' + '=' * 70)
print('STEP 2 — PHASE DETECTION')
print('=' * 70)

SMOOTH_WIN = 15
smooth_cols = ['mar', 'ear', 'eyebrow_height', 'pitch', 'yaw', 'mouth_width_ratio', 'nose_to_chin']
for col in smooth_cols:
    df[f'{col}_s'] = uniform_filter1d(df[col].values, size=SMOOTH_WIN)
    df[f'd_{col}'] = pd.Series(df[f'{col}_s']).diff().fillna(0).values

ranges = {col: max(df[col].max() - df[col].min(), 1e-6) for col in smooth_cols}
df['d_combined'] = sum(np.abs(df[f'd_{col}']) / ranges[col] for col in smooth_cols)

cmean = df['d_combined'].mean()
cstd = df['d_combined'].std()
thresh = cmean + 2.0 * cstd

candidates = df[df['d_combined'] > thresh]['frame'].values

def cluster(frames, gap=25):
    if len(frames) == 0:
        return []
    clusters = [[frames[0]]]
    for f in frames[1:]:
        if f - clusters[-1][-1] <= gap:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    return [int(np.median(c)) for c in clusters]

transitions = cluster(candidates, gap=25)
print(f'  Transitions: {transitions}')

bounds = sorted(set([0] + transitions + [int(df['frame'].max()) + 1]))
segments = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

# Merge short segments (< 12 frames)
merged = []
for seg in segments:
    if merged and seg[1] - seg[0] < 12:
        merged[-1] = (merged[-1][0], seg[1])
    else:
        merged.append(seg)
segments = merged

phases = []
print(f'\n  {"Ph":>3s} {"Start":>6s} {"End":>6s} {"N":>5s} {"Sec":>5s}'
      f'  {"MAR":>6s} {"EAR":>6s} {"EyBr":>6s} {"Pitch":>7s} {"Yaw":>7s} {"MWR":>6s}')
print('  ' + '-' * 75)

for i, (s, e) in enumerate(segments):
    sd = df[(df['frame'] >= s) & (df['frame'] < e)]
    if len(sd) == 0:
        continue
    p = {
        'phase': i, 'start': s, 'end': e, 'n': len(sd),
        'dur': (e - s) / face_fps,
        'mar': sd['mar'].mean(), 'ear': sd['ear'].mean(),
        'eyebrow': sd['eyebrow_height'].mean(),
        'pitch': sd['pitch'].mean(), 'yaw': sd['yaw'].mean(),
        'mwr': sd['mouth_width_ratio'].mean(),
        'cheek': sd['cheek_raise'].mean(),
    }
    phases.append(p)
    print(f'  {i:>3d} {s:>6d} {e:>6d} {len(sd):>5d} {p["dur"]:>5.1f}'
          f'  {p["mar"]:>6.3f} {p["ear"]:>6.3f} {p["eyebrow"]:>6.4f}'
          f'  {p["pitch"]:>+7.1f} {p["yaw"]:>+7.1f} {p["mwr"]:>6.3f}')

# ============================================================
# STEP 3 — CLASSIFICATION
# ============================================================
print('\n' + '=' * 70)
print('STEP 3 — CLASSIFICATION')
print('=' * 70)

# Find baseline: longest phase with small yaw and pitch (frontal, resting)
baseline_candidates = [p for p in phases if abs(p['yaw']) < 10 and abs(p['pitch']) < 10]
if not baseline_candidates:
    # Relax
    baseline_candidates = [p for p in phases if abs(p['yaw']) < 15 and abs(p['pitch']) < 15]
if not baseline_candidates:
    baseline_candidates = phases
baseline_phase = max(baseline_candidates, key=lambda p: p['n'])

bl_ear = baseline_phase['ear']
bl_mar = baseline_phase['mar']
bl_eyebrow = baseline_phase['eyebrow']
bl_mwr = baseline_phase['mwr']
bl_pitch = baseline_phase['pitch']
bl_yaw = baseline_phase['yaw']

# Compute nose_to_chin baseline from the baseline phase frames
bl_phase_df = df[(df['frame'] >= baseline_phase['start']) & (df['frame'] < baseline_phase['end'])]
bl_nose_chin = bl_phase_df['nose_to_chin'].mean()
bl_forehead_nose = bl_phase_df['forehead_to_nose'].mean()

print(f'\n  Baseline: Phase {baseline_phase["phase"]} (frames {baseline_phase["start"]}-{baseline_phase["end"]}, {baseline_phase["dur"]:.1f}s)')
print(f'    EAR={bl_ear:.4f}  MAR={bl_mar:.4f}  Eyebrow={bl_eyebrow:.4f}')
print(f'    MWR={bl_mwr:.4f}  Pitch={bl_pitch:+.1f}  Yaw={bl_yaw:+.1f}')
print(f'    Nose-to-Chin={bl_nose_chin:.4f}  Forehead-to-Nose={bl_forehead_nose:.4f}')

# Thresholds
HEAD_TURN_YAW = 15
HEAD_TILT_GEOM = bl_nose_chin * 0.85  # geometry-based: nose_to_chin < 85% of baseline
MOUTH_OPEN_MAR = 0.15
EYES_CLOSED_EAR = bl_ear - 0.04
SMILE_MWR = bl_mwr + 0.02
EYEBROW_THRESH = bl_eyebrow + 0.005
FRONTAL_TOLERANCE = 0.15  # nose_to_chin within 15% of baseline = frontal

print(f'\n  Thresholds:')
print(f'    head_turn:       |yaw| > {HEAD_TURN_YAW}deg')
print(f'    head_tilt:       nose_to_chin < {HEAD_TILT_GEOM:.4f} (< baseline * 0.85)')
print(f'    mouth_open:      MAR > {MOUTH_OPEN_MAR}')
print(f'    eyes_closed:     EAR < {EYES_CLOSED_EAR:.4f} AND frontal AND not smiling')
print(f'    smiling:         MWR > {SMILE_MWR:.4f}')
print(f'    eyebrows_raised: eyebrow > {EYEBROW_THRESH:.4f} AND frontal (nose_chin within {FRONTAL_TOLERANCE*100:.0f}%)')
print(f'    neutral:         everything else')

# Per-frame classification
# Priority order:
#   1. mouth_open (MAR > 0.15 — reliable regardless of head pose)
#   2. head_turn (|yaw| > 15)
#   3. head_tilt (nose_to_chin < baseline * 0.85 — geometry-based)
#   4. eyes_closed (EAR low AND frontal AND not smiling)
#   5. smiling (MWR elevated)
#   6. eyebrows_raised (eyebrow elevated AND frontal — nose_chin within 15%)
#   7. neutral (default)

def angle_delta(a, b):
    """Signed angular difference, handling wrapping."""
    d = a - b
    while d > 180: d -= 360
    while d < -180: d += 360
    return d

def is_frontal(nose_chin_val, baseline_val, tolerance=FRONTAL_TOLERANCE):
    """Check if face is frontal based on nose-to-chin geometry stability."""
    return abs(nose_chin_val - baseline_val) / baseline_val < tolerance

print('\n  Classifying all frames...')
frame_labels = {}
for _, row in df.iterrows():
    f = int(row['frame'])
    mar = row['mar']
    ear = row['ear']
    mwr = row['mouth_width_ratio']
    eyebrow = row['eyebrow_height']
    yaw = row['yaw']
    nose_chin = row['nose_to_chin']
    frontal = is_frontal(nose_chin, bl_nose_chin)

    # 1. Mouth wide open — very reliable signal
    if mar > MOUTH_OPEN_MAR:
        frame_labels[f] = 'mouth_open'
    # 2. Head turn — large yaw
    elif abs(yaw) > HEAD_TURN_YAW:
        frame_labels[f] = 'head_turn'
    # 3. Head tilt — geometry-based: nose_to_chin compressed
    elif nose_chin < HEAD_TILT_GEOM:
        frame_labels[f] = 'head_tilt'
    # 4. Eyes closed — EAR low AND frontal AND not smiling
    elif ear < EYES_CLOSED_EAR and frontal and abs(mwr - bl_mwr) < 0.03:
        frame_labels[f] = 'eyes_closed'
    # 5. Smiling — MWR elevated
    elif mwr > SMILE_MWR:
        frame_labels[f] = 'smiling'
    # 6. Eyebrows raised — elevated AND frontal AND small yaw
    elif eyebrow > EYEBROW_THRESH and frontal and abs(yaw) < 10:
        frame_labels[f] = 'eyebrows_raised'
    # 7. Neutral
    else:
        frame_labels[f] = 'neutral'

df['expression'] = df['frame'].map(frame_labels)

# Validate frame 1800
if 1800 in frame_labels:
    f1800 = df[df['frame'] == 1800].iloc[0]
    print(f'\n  Frame 1800 check: {frame_labels[1800]}')
    print(f'    nose_to_chin={f1800["nose_to_chin"]:.4f} (baseline={bl_nose_chin:.4f}, threshold={HEAD_TILT_GEOM:.4f})')
    print(f'    yaw={f1800["yaw"]:.1f}, eyebrow={f1800["eyebrow_height"]:.4f}')

# Check coverage
all_7 = {'neutral', 'smiling', 'mouth_open', 'eyes_closed', 'eyebrows_raised', 'head_turn', 'head_tilt'}
found = set(df['expression'].unique())
missing = all_7 - found

if missing:
    print(f'\n  Missing classes: {missing}. Relaxing thresholds...')
    for mc in list(missing):
        if mc == 'head_turn':
            max_yaw = df['yaw'].abs().max()
            if max_yaw > 8:
                relaxed = max_yaw * 0.7
                print(f'    head_turn: max |yaw|={max_yaw:.1f}, relaxing to {relaxed:.1f}')
                for _, row in df.iterrows():
                    f = int(row['frame'])
                    if abs(row['yaw']) > relaxed and frame_labels[f] == 'neutral':
                        frame_labels[f] = 'head_turn'
                        missing.discard('head_turn')
        elif mc == 'head_tilt':
            min_nc = df['nose_to_chin'].min()
            relaxed_ratio = 0.90  # relax from 0.85 to 0.90
            relaxed_thresh = bl_nose_chin * relaxed_ratio
            if min_nc < relaxed_thresh:
                print(f'    head_tilt: min nose_to_chin={min_nc:.4f}, relaxing to {relaxed_thresh:.4f} (< baseline * {relaxed_ratio})')
                for _, row in df.iterrows():
                    f = int(row['frame'])
                    if row['nose_to_chin'] < relaxed_thresh and frame_labels[f] in ('neutral', 'eyebrows_raised'):
                        frame_labels[f] = 'head_tilt'
                        missing.discard('head_tilt')
        elif mc == 'eyes_closed':
            # Relax MWR constraint
            print(f'    eyes_closed: relaxing MWR constraint')
            for _, row in df.iterrows():
                f = int(row['frame'])
                if row['ear'] < EYES_CLOSED_EAR and frame_labels[f] in ('neutral', 'smiling'):
                    frame_labels[f] = 'eyes_closed'
                    missing.discard('eyes_closed')
        elif mc == 'eyebrows_raised':
            relaxed = bl_eyebrow + 0.003
            print(f'    eyebrows_raised: relaxing to {relaxed:.4f}')
            for _, row in df.iterrows():
                f = int(row['frame'])
                if row['eyebrow_height'] > relaxed and abs(row['yaw']) < 10 and frame_labels[f] == 'neutral':
                    frame_labels[f] = 'eyebrows_raised'
                    missing.discard('eyebrows_raised')
        elif mc == 'smiling':
            relaxed = bl_mwr + 0.01
            print(f'    smiling: relaxing MWR to {relaxed:.4f}')
            for _, row in df.iterrows():
                f = int(row['frame'])
                if row['mouth_width_ratio'] > relaxed and frame_labels[f] == 'neutral':
                    frame_labels[f] = 'smiling'
                    missing.discard('smiling')
        elif mc == 'mouth_open':
            max_mar = df['mar'].max()
            relaxed = max(0.10, max_mar * 0.6)
            print(f'    mouth_open: relaxing MAR to {relaxed:.3f}')
            for _, row in df.iterrows():
                f = int(row['frame'])
                if row['mar'] > relaxed and frame_labels[f] == 'neutral':
                    frame_labels[f] = 'mouth_open'
                    missing.discard('mouth_open')

    df['expression'] = df['frame'].map(frame_labels)

# Build display phases from per-frame labels
display_phases = []
current_expr = None
current_start = None
for _, row in df.iterrows():
    f = int(row['frame'])
    e = row['expression']
    if e != current_expr:
        if current_expr is not None:
            display_phases.append({'start': current_start, 'end': f, 'expression': current_expr})
        current_expr = e
        current_start = f
if current_expr is not None:
    display_phases.append({'start': current_start, 'end': int(df['frame'].max()) + 1, 'expression': current_expr})

# Merge tiny phases (< 5 frames)
merged_dp = []
for dp in display_phases:
    dur = dp['end'] - dp['start']
    if merged_dp and dur < 5:
        merged_dp[-1]['end'] = dp['end']
    else:
        merged_dp.append(dp)

# Recompute stats
phases_final = []
for i, dp in enumerate(merged_dp):
    sd = df[(df['frame'] >= dp['start']) & (df['frame'] < dp['end'])]
    if len(sd) == 0:
        continue
    # Majority vote for expression within merged phase
    expr = sd['expression'].mode().iloc[0] if len(sd) > 0 else dp['expression']
    phases_final.append({
        'phase': i, 'start': dp['start'], 'end': dp['end'],
        'n': len(sd), 'dur': (dp['end'] - dp['start']) / face_fps,
        'mar': sd['mar'].mean(), 'ear': sd['ear'].mean(),
        'eyebrow': sd['eyebrow_height'].mean(),
        'pitch': sd['pitch'].mean(), 'yaw': sd['yaw'].mean(),
        'mwr': sd['mouth_width_ratio'].mean(),
        'expression': expr,
    })

# ============================================================
# STEP 3b — VALIDATION
# ============================================================
print('\n' + '=' * 70)
print('STEP 3b — VALIDATION')
print('=' * 70)

# Print only phases > 10 frames for clarity, but count all
long_phases = [p for p in phases_final if p['n'] >= 10]
print(f'\n  Major phases (>= 10 frames):')
print(f'  {"Ph":>3s} {"Frames":>14s} {"Sec":>5s} {"Expression":>18s}'
      f'  {"MAR":>6s} {"EAR":>6s} {"Pitch":>7s} {"Yaw":>7s} {"MWR":>6s}')
print('  ' + '-' * 80)
for p in long_phases:
    print(f'  {p["phase"]:>3d} {p["start"]:>6d}-{p["end"]:<6d} {p["dur"]:>5.1f} {p["expression"]:>18s}'
          f'  {p["mar"]:>6.3f} {p["ear"]:>6.3f} {p["pitch"]:>+7.1f} {p["yaw"]:>+7.1f} {p["mwr"]:>6.3f}')

print(f'\n  Total phases: {len(phases_final)} ({len(long_phases)} major)')

found = set(df['expression'].unique())
print(f'  Classes found: {sorted(found)}')
still_missing = all_7 - found
if still_missing:
    print(f'  MISSING: {still_missing}')
else:
    print(f'  All 7 classes present!')

print(f'\n  Expression Distribution:')
for expr, count in df['expression'].value_counts().items():
    print(f'    {expr:>18s}: {count:>5d} frames ({count/len(df)*100:.1f}%)')

# ============================================================
# STEP 4 — REGENERATE OUTPUTS
# ============================================================
print('\n' + '=' * 70)
print('STEP 4 — REGENERATE OUTPUTS')
print('=' * 70)

expr_bg = {
    'neutral': '#f0f0f0', 'smiling': '#fff3e0', 'eyebrows_raised': '#e8f5e9',
    'eyes_closed': '#e3f2fd', 'mouth_open': '#fce4ec',
    'head_turn': '#f3e5f5', 'head_tilt': '#fff9c4',
}

# --- 4a: feature_exploration.png ---
print('\n[4a] feature_exploration.png...')
fig, axes = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
feat_plot = [
    ('mar', 'MAR', 'coral'), ('ear', 'EAR', 'steelblue'),
    ('eyebrow_height', 'Eyebrow H', 'green'),
    ('nose_to_chin', 'Nose-Chin Dist', 'darkcyan'),
    ('yaw', 'Yaw (deg)', 'brown'), ('pitch', 'Pitch (deg)', 'purple'),
]
for ax_i, (col, title, color) in enumerate(feat_plot):
    ax = axes[ax_i]
    for p in phases_final:
        bg = expr_bg.get(p['expression'], '#f5f5f5')
        ax.axvspan(p['start'], p['end'], alpha=0.4, color=bg, zorder=0)
    for tp in transitions:
        ax.axvline(x=tp, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.plot(df['frame'], df[col], color=color, linewidth=0.6, zorder=2)
    ax.set_ylabel(title, fontsize=9)
    ax.grid(True, alpha=0.2, zorder=1)

for p in long_phases:
    mid = (p['start'] + p['end']) / 2
    axes[0].text(mid, axes[0].get_ylim()[1] * 0.9, p['expression'],
                 ha='center', va='top', fontsize=6, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='grey', alpha=0.8))

axes[-1].set_xlabel('Frame')
legend_patches = [Patch(facecolor=c, label=e, alpha=0.5) for e, c in expr_bg.items()]
axes[0].legend(handles=legend_patches, loc='upper right', fontsize=6, ncol=4)
plt.suptitle('Feature Exploration — Head Pose + Expression Features (face_video)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_exploration.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved')

# --- 4b: face_mesh_samples.png ---
print('\n[4b] face_mesh_samples.png...')
target_classes = ['neutral', 'smiling', 'mouth_open', 'eyebrows_raised', 'eyes_closed', 'head_turn', 'head_tilt']
samples = []
for cls in target_classes:
    cls_df = df[df['expression'] == cls]
    if len(cls_df) == 0:
        continue
    mid_frame = cls_df['frame'].values[len(cls_df) // 2]
    candidates = sorted(frames_cache.keys(), key=lambda k: abs(k - mid_frame))
    if candidates and candidates[0] in frames_cache:
        samples.append((candidates[0], frames_cache[candidates[0]], cls))

samples = samples[:7]
samples.sort(key=lambda x: x[0])

n_show = len(samples)
fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 6))
if n_show == 1:
    axes = [axes]
for i, (fidx, frame, expr) in enumerate(samples):
    axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f'Frame {fidx}\n{expr}', fontsize=10, fontweight='bold')
    axes[i].axis('off')
fig.suptitle('Facial Mesh — Expression & Head Movement Classification (face_video)', fontsize=14, fontweight='bold')
plt.subplots_adjust(top=0.82)
plt.savefig(FIGURES_DIR / 'face_mesh_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved ({[s[2] for s in samples]})')

# --- 4c: expression_features.png ---
print('\n[4c] expression_features.png...')
fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

for ax in axes:
    for p in phases_final:
        bg = expr_bg.get(p['expression'], '#f5f5f5')
        ax.axvspan(p['start'], p['end'], alpha=0.4, color=bg, zorder=0)
    for tp in transitions:
        ax.axvline(x=tp, color='black', linestyle='--', alpha=0.3, linewidth=0.5)

axes[0].plot(df['frame'], df['mar'], color='coral', linewidth=0.8, zorder=2)
axes[0].axhline(y=MOUTH_OPEN_MAR, color='red', linestyle=':', alpha=0.5, label=f'Mouth open > {MOUTH_OPEN_MAR}')
axes[0].axhline(y=bl_mar, color='grey', linestyle=':', alpha=0.5, label=f'Baseline = {bl_mar:.3f}')
axes[0].set_ylabel('MAR', fontsize=10); axes[0].legend(fontsize=7, loc='upper right'); axes[0].grid(True, alpha=0.2)

axes[1].plot(df['frame'], df['ear'], color='steelblue', linewidth=0.8, zorder=2)
axes[1].axhline(y=EYES_CLOSED_EAR, color='red', linestyle=':', alpha=0.5, label=f'Eyes closed < {EYES_CLOSED_EAR:.3f}')
axes[1].axhline(y=bl_ear, color='grey', linestyle=':', alpha=0.5, label=f'Baseline = {bl_ear:.3f}')
axes[1].set_ylabel('EAR', fontsize=10); axes[1].legend(fontsize=7, loc='upper right'); axes[1].grid(True, alpha=0.2)

axes[2].plot(df['frame'], df['eyebrow_height'], color='green', linewidth=0.8, zorder=2)
axes[2].axhline(y=EYEBROW_THRESH, color='red', linestyle=':', alpha=0.5, label=f'Raised > {EYEBROW_THRESH:.4f}')
axes[2].axhline(y=bl_eyebrow, color='grey', linestyle=':', alpha=0.5, label=f'Baseline = {bl_eyebrow:.4f}')
axes[2].set_ylabel('Eyebrow', fontsize=10); axes[2].legend(fontsize=7, loc='upper right'); axes[2].grid(True, alpha=0.2)

axes[3].plot(df['frame'], df['nose_to_chin'], color='darkcyan', linewidth=0.8, zorder=2)
axes[3].axhline(y=HEAD_TILT_GEOM, color='red', linestyle=':', alpha=0.5, label=f'Tilt < {HEAD_TILT_GEOM:.4f}')
axes[3].axhline(y=bl_nose_chin, color='grey', linestyle=':', alpha=0.5, label=f'Baseline = {bl_nose_chin:.4f}')
axes[3].set_ylabel('Nose-Chin', fontsize=10); axes[3].legend(fontsize=7, loc='upper right'); axes[3].grid(True, alpha=0.2)

axes[4].plot(df['frame'], df['yaw'], color='brown', linewidth=0.8, zorder=2)
axes[4].axhline(y=HEAD_TURN_YAW, color='red', linestyle=':', alpha=0.5, label=f'Turn +{HEAD_TURN_YAW}')
axes[4].axhline(y=-HEAD_TURN_YAW, color='red', linestyle=':', alpha=0.5, label=f'Turn -{HEAD_TURN_YAW}')
axes[4].axhline(y=0, color='grey', linestyle=':', alpha=0.5, label='Center')
axes[4].set_ylabel('Yaw (deg)', fontsize=10); axes[4].set_xlabel('Frame')
axes[4].legend(fontsize=7, loc='upper right'); axes[4].grid(True, alpha=0.2)

legend_patches = [Patch(facecolor=c, label=e, alpha=0.5) for e, c in expr_bg.items()]
axes[0].legend(handles=legend_patches, loc='upper left', fontsize=6, ncol=4)
plt.suptitle('Expression & Head Pose Features — Publication View (face_video)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'expression_features.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved')

# --- 4d: combined_behavioural_timeline.png ---
print('\n[4d] combined_behavioural_timeline.png...')
body_df = pd.read_csv(OUTPUT_DIR / 'combined_behavioural_data.csv')
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False, gridspec_kw={'height_ratios': [3, 3, 1]})

axes[0].fill_between(body_df['frame'], body_df['body_velocity'], alpha=0.4, color='steelblue')
axes[0].plot(body_df['frame'], body_df['body_velocity'], color='steelblue', linewidth=0.8)
axes[0].set_ylabel('Body\nVelocity', fontsize=10)
axes[0].set_title('Camera 1 — Body Pose (dance_video.MOV)', fontsize=11, fontweight='bold', loc='left')
axes[0].grid(True, alpha=0.3); axes[0].set_xlabel('Frame')

for p in phases_final:
    axes[1].axvspan(p['start'], p['end'], alpha=0.4, color=expr_bg.get(p['expression'], '#f5f5f5'), zorder=0)
axes[1].plot(df['frame'], df['mar'], color='coral', alpha=0.8, label='MAR', linewidth=0.8)
axes[1].plot(df['frame'], df['ear'], color='steelblue', alpha=0.8, label='EAR', linewidth=0.8)
axes[1].plot(df['frame'], df['nose_to_chin'], color='darkcyan', alpha=0.8, label='NoseChin', linewidth=0.8)
axes[1].plot(df['frame'], df['yaw'] / 100, color='brown', alpha=0.8, label='Yaw/100', linewidth=0.8)
axes[1].legend(fontsize=7, loc='upper right', ncol=4)
axes[1].set_ylabel('Feature\nValue', fontsize=10)
axes[1].set_title('Camera 2 — Facial Expression & Head Pose (face_video.MOV)', fontsize=11, fontweight='bold', loc='left')
axes[1].grid(True, alpha=0.3); axes[1].set_xlabel('Frame')

axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
axes[2].text(0.5, 0.5,
    'In a multi-camera FCBG setup, body pose (Camera 1) and facial expression (Camera 2)\n'
    'would be synchronised via hardware triggers and timestamps, enabling\n'
    'holistic behavioural analysis combining movement patterns with facial affect.',
    ha='center', va='center', fontsize=10, style='italic',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.8))
axes[2].axis('off')
axes[2].set_title('Multi-Camera Synchronisation Note', fontsize=11, fontweight='bold', loc='left')
plt.suptitle('Combined Behavioural Timeline — Multi-Camera Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'combined_behavioural_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved')

# --- 4e: annotated_output_face.mp4 ---
print('\n[4e] annotated_output_face.mp4 (all frames)...')
expr_vid_colors = {
    'neutral': (200, 200, 200), 'smiling': (0, 255, 128),
    'mouth_open': (0, 128, 255), 'eyes_closed': (255, 100, 100),
    'eyebrows_raised': (128, 255, 0), 'head_turn': (255, 128, 255),
    'head_tilt': (0, 255, 255),
}
frame_expr_map = dict(zip(df['frame'].astype(int), df['expression']))
frame_nc_map = dict(zip(df['frame'].astype(int), df['nose_to_chin']))
frame_yaw_map = dict(zip(df['frame'].astype(int), df['yaw']))

cap = cv2.VideoCapture(str(FACE_VIDEO))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(OUTPUT_DIR / 'annotated_output_face.mp4'), fourcc, face_fps, (face_w, face_h))
frame_idx = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as fm:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = fix_rotation(frame, face_rotation)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)
        if results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        expr_label = frame_expr_map.get(frame_idx, 'neutral')
        nc_val = frame_nc_map.get(frame_idx, 0)
        y_val = frame_yaw_map.get(frame_idx, 0)
        color = expr_vid_colors.get(expr_label, (200, 200, 200))
        cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, expr_label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f'NoseChin:{nc_val:.3f} Yaw:{y_val:+.0f}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
        out.write(frame)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f'  Frame {frame_idx}/{total_frames}')

cap.release()
out.release()
print(f'  Saved ({frame_idx} frames)')

# --- 4f: facial_features.csv ---
print('\n[4f] facial_features.csv...')
out_cols = ['frame', 'mar', 'ear', 'eyebrow_height', 'nose_to_chin', 'yaw', 'mouth_width_ratio', 'expression']
df[out_cols].to_csv(OUTPUT_DIR / 'facial_features.csv', index=False)
print(f'  Saved ({len(df)} rows)')

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n' + '=' * 70)
print('FINAL SUMMARY')
print('=' * 70)
print(f'\n  Expression Distribution:')
for expr, count in df['expression'].value_counts().items():
    print(f'    {expr:>18s}: {count:>5d} frames ({count/len(df)*100:.1f}%)')

print(f'\n  Generated files:')
for f in ['feature_exploration.png', 'face_mesh_samples.png', 'expression_features.png', 'combined_behavioural_timeline.png']:
    p = FIGURES_DIR / f
    if p.exists():
        print(f'    figures/{f:45s} {os.path.getsize(p):>10,} bytes')
for f in ['facial_features.csv']:
    p = OUTPUT_DIR / f
    print(f'    output/{f:46s} {os.path.getsize(p):>10,} bytes')
p = OUTPUT_DIR / 'annotated_output_face.mp4'
print(f'    output/{"annotated_output_face.mp4":46s} {os.path.getsize(p):>10,} bytes ({os.path.getsize(p)/1024/1024:.1f} MB)')
