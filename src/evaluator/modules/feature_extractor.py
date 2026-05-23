import torch
import numpy as np

class FeatureExtractor:
    """Class tunggal untuk mengekstrak fitur rata-rata temporal (mengabaikan mask)."""
    def __init__(self, device):
        self.device = device

    def extract(self, model, loader):
        model.eval()
        all_features, all_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in loader:
                B = batch_x.shape[0]
                valid_mask = ~batch_mask.to(self.device)  # (B, max_len)

                if batch_x.ndim == 3:  # (B, C, T)
                    T = batch_x.shape[2]
                    valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(batch_x)
                    x_masked = batch_x.to(self.device) * valid_mask_expanded.float()
                    valid_lengths = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                    mean_features = x_masked.sum(dim=2) / valid_lengths
                    mean_features = mean_features.view(B, -1)
                elif batch_x.ndim == 6:  # (B, N_roi, C, T, H, W)
                    T = batch_x.shape[3]
                    valid_mask_expanded = valid_mask.view(B, 1, 1, T, 1, 1).expand_as(batch_x)
                    x_masked = batch_x.to(self.device) * valid_mask_expanded.float()
                    valid_lengths = valid_mask.sum(dim=1).clamp_min(1.0).view(B, 1, 1, 1, 1)
                    mean_features = x_masked.sum(dim=3) / valid_lengths
                    mean_features = mean_features.view(B, -1)
                else:
                    raise ValueError(f"FeatureExtractor unsupported batch_x ndim: {batch_x.ndim}")
                
                all_features.append(mean_features.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
                
        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)