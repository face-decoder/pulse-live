import torch
import torch.nn.functional as F
import numpy as np

class MetricsEvaluator:
    """Class tunggal untuk melakukan forward pass dan menghitung loss/akurasi."""
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, loader, criterion=None):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in loader:
                batch_x, batch_y, batch_mask = batch_x.to(self.device), batch_y.to(self.device), batch_mask.to(self.device)
                logits = model(batch_x, mask=batch_mask)
                
                if criterion:
                    loss = criterion(logits, batch_y)
                    running_loss += loss.item() * batch_x.size(0)
                
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        
        return val_loss, val_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)