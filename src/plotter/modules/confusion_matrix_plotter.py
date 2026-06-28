import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlotter:
    """Class tunggal untuk merender visualisasi Heatmap Confusion Matrix."""
    def __init__(self, target_names):
        self.target_names = target_names

    def show(self, y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Oranges', 
                    xticklabels=self.target_names, 
                    yticklabels=self.target_names)
        
        plt.title(title, fontdict={'fontsize': 14})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()