import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

class TSNEPlotter:
    """Class tunggal untuk menghitung dan menggambar sebaran t-SNE."""
    def __init__(self, target_names):
        self.target_names = target_names

    def show(self, train_feats, train_lbls, val_feats, val_lbls):
        X_combined = np.vstack((train_feats, val_feats))
        y_combined = np.concatenate((train_lbls, val_lbls))
        split_labels = ['Train'] * len(train_feats) + ['Validation'] * len(val_feats)
        
        safe_perplexity = min(30, len(X_combined) - 1)
        tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_combined)
        
        df_tsne = pd.DataFrame({
            'TSNE-1': X_tsne[:, 0], 'TSNE-2': X_tsne[:, 1],
            'Split': split_labels,
            'Class': [self.target_names[lbl] for lbl in y_combined]
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('t-SNE Dimensionality Reduction (Feature Distribution)', fontsize=16, fontweight='bold', y=1.02)
        
        sns.scatterplot(data=df_tsne, x='TSNE-1', y='TSNE-2', hue='Split', palette={'Train': 'royalblue', 'Validation': 'darkorange'}, alpha=0.7, s=60, ax=axes[0])
        axes[0].set_title('Dataset Split (Data Shift Check)')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        sns.scatterplot(data=df_tsne, x='TSNE-1', y='TSNE-2', hue='Class', style='Class', palette={self.target_names[0]: 'mediumseagreen', self.target_names[1]: 'crimson'}, alpha=0.8, s=70, ax=axes[1])
        axes[1].set_title('Target Class (Separability Check)')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()