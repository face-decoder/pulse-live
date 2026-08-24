import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
sys.path.append('.')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from src.apex.modules.apex_phase_spotter_roi import ApexPhaseSpotterROI
from src.dataset.modules.behavioral_features import BehavioralFeatures
from src.models.modules.cnn_1d_extractor import CNN1DExtractor
from src.models.modules.cnn_transformer.cnn_transformer import CNN_Transformer

torch.manual_seed(42)
np.random.seed(42)

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

print("2. Precomputing behavioral feature matrices...", flush=True)
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

print(f"Precomputed features for {len(data_cache)} micro-expressions.", flush=True)

class CNN1DClassifier(nn.Module):
    def __init__(self, in_channels=47, out_channels=64, num_classes=2, dropout_p=0.4):
        super().__init__()
        self.extractor = CNN1DExtractor(in_channels=in_channels, out_channels=out_channels, dropout_p=dropout_p)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes)
        )
    def forward(self, x, mask=None):
        feat = self.extractor(x)
        pooled = self.pool(feat).squeeze(-1)
        return self.classifier(pooled)

cutoff_ratios = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]

print(f"\n{'Cutoff':<8} | {'SVM Acc':<10} | {'SVM F1':<10} | {'1D-CNN Acc':<12} | {'1D-CNN F1':<12} | {'Transformer Acc':<16} | {'Transformer F1':<16}", flush=True)
print("-" * 95, flush=True)

for cr in cutoff_ratios:
    spotter = ApexPhaseSpotterROI(cutoff_ratio=cr, show_frame=False)
    all_seqs, labels, groups = [], [], []
    
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
        else:
            onset_for_features = gt_onset
            offset_for_features = gt_offset
            
        sliced_feat = full_feat[onset_for_features:offset_for_features+1]
        if sliced_feat.shape[0] > 0:
            all_seqs.append(sliced_feat)
            labels.append(emotion)
            groups.append(sub_prefix)
            
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    
    X_static = np.stack([np.concatenate([seq.mean(axis=0), seq.std(axis=0)]) for seq in all_seqs])
    logo = LeaveOneGroupOut()
    splits = list(logo.split(X_static, y_encoded, groups=np.array(groups)))
    svm_preds = np.zeros_like(y_encoded)
    for train_idx, test_idx in splits:
        X_train, y_train = X_static[train_idx], y_encoded[train_idx]
        X_test, y_test = X_static[test_idx], y_encoded[test_idx]
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        svm_preds[test_idx] = model.predict(X_test_scaled)
    svm_acc = accuracy_score(y_encoded, svm_preds)
    svm_f1 = f1_score(y_encoded, svm_preds, average='macro')
    
    max_len = max(seq.shape[0] for seq in all_seqs)
    N = len(all_seqs)
    X_padded = np.zeros((N, 47, max_len), dtype=np.float32)
    mask_padded = np.ones((N, max_len), dtype=bool)
    for i, seq in enumerate(all_seqs):
        t_len = seq.shape[0]
        X_padded[i, :, :t_len] = seq.T
        mask_padded[i, :t_len] = False
        
    logo_dl = LeaveOneGroupOut()
    dl_splits = list(logo_dl.split(X_padded, y_encoded, groups=np.array(groups)))
    
    cnn_preds = np.zeros_like(y_encoded)
    for train_idx, test_idx in dl_splits:
        X_train, y_train = X_padded[train_idx], y_encoded[train_idx]
        X_test, y_test = X_padded[test_idx], y_encoded[test_idx]
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        torch.manual_seed(42)
        cnn_model = CNN1DClassifier(in_channels=47, out_channels=64, num_classes=num_classes)
        optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        in_t = torch.tensor(X_train_norm, dtype=torch.float32)
        tgt_t = torch.tensor(y_train, dtype=torch.long)
        
        cnn_model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            loss = criterion(cnn_model(in_t), tgt_t)
            loss.backward()
            optimizer.step()
            
        cnn_model.eval()
        with torch.no_grad():
            test_in = torch.tensor(X_test_norm, dtype=torch.float32)
            cnn_preds[test_idx] = torch.argmax(cnn_model(test_in), dim=1).numpy()
            
    cnn_acc = accuracy_score(y_encoded, cnn_preds)
    cnn_f1 = f1_score(y_encoded, cnn_preds, average='macro')
    
    trans_preds = np.zeros_like(y_encoded)
    for train_idx, test_idx in dl_splits:
        X_train, y_train, m_train = X_padded[train_idx], y_encoded[train_idx], mask_padded[train_idx]
        X_test, y_test, m_test = X_padded[test_idx], y_encoded[test_idx], mask_padded[test_idx]
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        torch.manual_seed(42)
        trans_model = CNN_Transformer(in_channels=47, d_model=64, nhead=4, num_layers=2, num_classes=num_classes, dropout_p=0.3)
        optimizer = optim.AdamW(trans_model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        in_t = torch.tensor(X_train_norm, dtype=torch.float32)
        mask_t = torch.tensor(m_train, dtype=torch.bool)
        tgt_t = torch.tensor(y_train, dtype=torch.long)
        
        trans_model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            loss = criterion(trans_model(in_t, mask=mask_t), tgt_t)
            loss.backward()
            optimizer.step()
            
        trans_model.eval()
        with torch.no_grad():
            test_in = torch.tensor(X_test_norm, dtype=torch.float32)
            test_m = torch.tensor(m_test, dtype=torch.bool)
            trans_preds[test_idx] = torch.argmax(trans_model(test_in, mask=test_m), dim=1).numpy()
            
    trans_acc = accuracy_score(y_encoded, trans_preds)
    trans_f1 = f1_score(y_encoded, trans_preds, average='macro')
    
    print(f"{cr:<8.2f} | {svm_acc:<10.4f} | {svm_f1:<10.4f} | {cnn_acc:<12.4f} | {cnn_f1:<12.4f} | {trans_acc:<16.4f} | {trans_f1:<16.4f}", flush=True)
