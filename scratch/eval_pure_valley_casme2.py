import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
sys.path.append('.')

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import find_peaks

annotations_path = '/home/inadio/datasets/secondaries/cas(me)^2/CAS(ME)^2code_final.xlsx'
cache_dir = '/home/inadio/datasets/secondaries/cas(me)^2/cache'

print("Loading CAS(ME)^2 dataset...", flush=True)
df_rule1 = pd.read_excel(annotations_path, sheet_name='naming rule1', header=None)
sub_map = {int(r[2]): {'prefix': str(r[1])} for _, r in df_rule1.iterrows()}
df_rule2 = pd.read_excel(annotations_path, sheet_name='naming rule2', header=None)
stimulus_map = {str(r[1]): f'{int(r[0]):04d}' for _, r in df_rule2.iterrows()}
df = pd.read_excel(annotations_path, sheet_name='CASFEcode_final', header=None)
df.columns = ['Subject_ID', 'Clip_Name', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'AUs', 'Valence', 'Type', 'Emotion']
df = df[df['Type'] == 'micro-expression'].copy()
df = df[(df['OffsetFrame'] - df['OnsetFrame'] + 1) <= 100].copy()

TIME_MARGIN = 0.05
FPS = 30
margin_frames = int(TIME_MARGIN * FPS)
MAX_SEARCH_RADIUS = 100

data_cache = []
for idx, row in df.iterrows():
    sub_id = int(row['Subject_ID'])
    clip_name = str(row['Clip_Name'])
    emotion_raw = row['Emotion']
    sub_info = sub_map.get(sub_id)
    stimulus_code = stimulus_map.get(clip_name.split('_')[0])
    if not sub_info or not stimulus_code:
        continue
    sub_prefix = sub_info['prefix']
    npz_path = os.path.join(cache_dir, f'{sub_prefix}_{stimulus_code}.npz')
    if not os.path.exists(npz_path):
        continue
        
    data = np.load(npz_path, mmap_mode='r')
    magnitudes = data['magnitudes'].tolist()
    flow_tensor = torch.tensor(data['flow'], dtype=torch.float32)
    gt_onset = int(row['OnsetFrame'])
    gt_offset = int(row['OffsetFrame'])
    
    emo_clean = str(emotion_raw).lower().strip()
    if emo_clean == 'happiness':
        emotion = 'positive'
    elif emo_clean in ['disgust', 'fear', 'sadness', 'anger', 'pain', 'helpless']:
        emotion = 'negative'
    else:
        emotion = None
        
    data_cache.append({
        'sub_prefix': sub_prefix,
        'magnitudes': magnitudes,
        'flow_tensor': flow_tensor,
        'gt_onset': gt_onset,
        'gt_offset': gt_offset,
        'emotion': emotion
    })

print(f"Loaded {len(data_cache)} micro-expressions.", flush=True)

def find_pure_valley_boundaries(signal, apex_index, uptick_tolerance=0.0):
    N = len(signal)
    apex_val = signal[apex_index]
    left_bound = max(0, apex_index - MAX_SEARCH_RADIUS)
    right_bound = min(N - 1, apex_index + MAX_SEARCH_RADIUS)
    
    run_min_val = apex_val
    run_min_idx = apex_index
    for i in range(apex_index - 1, left_bound - 1, -1):
        val = signal[i]
        if val <= run_min_val:
            run_min_val = val
            run_min_idx = i
        else:
            if (val - run_min_val) > uptick_tolerance * (apex_val - run_min_val + 1e-6):
                break
    onset = run_min_idx
    
    run_min_val = apex_val
    run_min_idx = apex_index
    for i in range(apex_index + 1, right_bound + 1):
        val = signal[i]
        if val <= run_min_val:
            run_min_val = val
            run_min_idx = i
        else:
            if (val - run_min_val) > uptick_tolerance * (apex_val - run_min_val + 1e-6):
                break
    offset = run_min_idx
    
    return onset, offset

uptick_tolerances = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

print("\n--- Pure Valley Traversal (No Cutoff Ratio) ---", flush=True)
print(f"{'Uptick Tol':<12} | {'Spot F1':<10} | {'Avg IoU':<10} | {'TPs (IoU>=0.5)':<15} | {'Avg Window':<14}", flush=True)
print("-" * 75, flush=True)

for tol in uptick_tolerances:
    spotted_intervals, gt_intervals = [], []
    window_lengths = []
    
    for item in data_cache:
        mags = item['magnitudes']
        gt_onset = item['gt_onset']
        gt_offset = item['gt_offset']
        
        peaks, _ = find_peaks(mags, distance=5, prominence=0.1)
        if len(peaks) == 0:
            apex_idx = np.argmax(mags)
        else:
            apex_idx = peaks[np.argmax([mags[p] for p in peaks])]
            
        onset_for_feat, offset_for_feat = find_pure_valley_boundaries(mags, apex_idx, uptick_tolerance=tol)
        
        onset_spotted = max(0, onset_for_feat - margin_frames)
        offset_spotted = min(len(mags) - 1, offset_for_feat + margin_frames)
        
        spotted_intervals.append((onset_spotted, offset_spotted))
        gt_intervals.append((gt_onset, gt_offset))
        window_lengths.append(offset_for_feat - onset_for_feat + 1)
        
    tps = 0
    ious = []
    for (s_onset, s_offset), (g_onset, g_offset) in zip(spotted_intervals, gt_intervals):
        intersection = max(0, min(s_offset, g_offset) - max(s_onset, g_onset) + 1)
        union = (s_offset - s_onset + 1) + (g_offset - g_onset + 1) - intersection
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
        if iou >= 0.5:
            tps += 1
            
    n_samples = len(spotted_intervals)
    spot_prec = tps / n_samples
    spot_rec = tps / n_samples
    spot_f1 = 2 * spot_prec * spot_rec / (spot_prec + spot_rec) if (spot_prec + spot_rec) > 0 else 0
    avg_iou = np.mean(ious)
    avg_win = np.mean(window_lengths)
    
    print(f"{tol:<12.2f} | {spot_f1:<10.4f} | {avg_iou:<10.4f} | {tps}/{n_samples:<13} | {avg_win:<14.1f} frames", flush=True)
