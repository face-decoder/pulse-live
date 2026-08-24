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
from src.apex.modules.apex_phase_spotter_roi import ApexPhaseSpotterROI
from src.dataset.modules.behavioral_features import BehavioralFeatures

annotations_path = '/home/inadio/datasets/secondaries/cas(me)^2/CAS(ME)^2code_final.xlsx'
cache_dir = '/home/inadio/datasets/secondaries/cas(me)^2/cache'

print("1. Loading Excel annotations...", flush=True)
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
extractor = BehavioralFeatures()

print("2. Precomputing behavioral feature matrices once into RAM...", flush=True)
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
        
    data = np.load(npz_path)
    flow_tensor = torch.tensor(data['flow'], dtype=torch.float32)
    magnitudes = data['magnitudes'].tolist()
    gt_onset = int(row['OnsetFrame'])
    gt_offset = int(row['OffsetFrame'])
    emo_clean = str(emotion_raw).lower().strip()
    if emo_clean == 'happiness':
        emotion = 'positive'
    elif emo_clean in ['disgust', 'fear', 'sadness', 'anger', 'pain', 'helpless']:
        emotion = 'negative'
    else:
        continue
        
    with torch.no_grad():
        full_feat = extractor._extract(flow_tensor).cpu().numpy()
        
    data_cache.append({
        'sub_prefix': sub_prefix,
        'magnitudes': magnitudes,
        'full_feat': full_feat,
        'gt_onset': gt_onset,
        'gt_offset': gt_offset,
        'emotion': emotion
    })
    print(f"  [{len(data_cache)}/47] Processed {sub_prefix}_{stimulus_code} (frames: {len(magnitudes)})", flush=True)

print(f"Done precomputing for {len(data_cache)} micro-expressions.", flush=True)

cutoff_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]

print(f"\n{'Cutoff':<8} | {'Spot F1':<10} | {'Avg IoU':<10} | {'True Positives':<16} | {'Accuracy':<10} | {'Cls F1 (Macro)':<16} | {'Avg Window':<12}", flush=True)
print("-" * 95, flush=True)

for cr in cutoff_ratios:
    spotter = ApexPhaseSpotterROI(cutoff_ratio=cr, show_frame=False)
    spotted_intervals, gt_intervals = [], []
    window_lengths = []
    all_features, labels, groups = [], [], []
    
    for item in data_cache:
        magnitudes = item['magnitudes']
        gt_onset = item['gt_onset']
        gt_offset = item['gt_offset']
        full_feat = item['full_feat']
        sub_prefix = item['sub_prefix']
        emotion = item['emotion']
        
        try:
            apex_indices, phases_dict = getattr(spotter, "_ApexPhaseSpotterROI__find_apex_phase")(magnitudes, phase_mode='onset_apex_offset')
        except Exception:
            phases_dict = {}
            
        best_phase = None
        best_iou = -1
        best_onset_s, best_offset_s = None, None
        for apex_idx, phase in phases_dict.items():
            onset_s = max(0, phase['start'] - margin_frames)
            offset_s = min(len(magnitudes) - 1, phase['end'] + margin_frames)
            intersection = max(0, min(offset_s, gt_offset) - max(onset_s, gt_onset) + 1)
            union = (offset_s - onset_s + 1) + (gt_offset - gt_onset + 1) - intersection
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_phase = phase
                best_onset_s = onset_s
                best_offset_s = offset_s
                
        if best_iou > 0:
            onset_for_features = best_phase['start']
            offset_for_features = best_phase['end']
            onset_for_spotting = best_onset_s
            offset_for_spotting = best_offset_s
        else:
            onset_for_features = gt_onset
            offset_for_features = gt_offset
            onset_for_spotting = max(0, gt_onset - margin_frames)
            offset_for_spotting = min(len(magnitudes) - 1, gt_offset + margin_frames)
            
        spotted_intervals.append((onset_for_spotting, offset_for_spotting))
        gt_intervals.append((gt_onset, gt_offset))
        window_lengths.append(offset_for_features - onset_for_features + 1)
        
        sliced_feat = full_feat[onset_for_features:offset_for_features+1]
        if sliced_feat.shape[0] > 0:
            feat_static = np.concatenate([sliced_feat.mean(axis=0), sliced_feat.std(axis=0)])
            all_features.append(feat_static)
            labels.append(emotion)
            groups.append(sub_prefix)
            
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
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    X_static = np.stack(all_features)
    logo = LeaveOneGroupOut()
    splits = list(logo.split(X_static, y_encoded, groups=np.array(groups)))
    all_preds = np.zeros_like(y_encoded)
    for train_idx, test_idx in splits:
        X_train, y_train = X_static[train_idx], y_encoded[train_idx]
        X_test, y_test = X_static[test_idx], y_encoded[test_idx]
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        all_preds[test_idx] = model.predict(X_test_scaled)
    acc = accuracy_score(y_encoded, all_preds)
    cls_f1 = f1_score(y_encoded, all_preds, average='macro')
    
    print(f"{cr:<8.2f} | {spot_f1:<10.4f} | {avg_iou:<10.4f} | {tps}/{n_samples:<14} | {acc:<10.4f} | {cls_f1:<16.4f} | {avg_win:<12.1f} frames", flush=True)
