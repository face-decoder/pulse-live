import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ROCCurvePlotter:
    """Class tunggal untuk menggambar ROC Curve dan AUC."""
    def show(self, y_true, y_prob, title='ROC Curve'):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='purple', lw=2.5, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()