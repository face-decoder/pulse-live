import matplotlib.pyplot as plt

class HistoryPlotter:
    """Class tunggal untuk menggambar tren Loss dan Akurasi."""
    def show(self, history: dict):
        epochs_range = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='royalblue')
        axes[0].plot(epochs_range, history['val_loss'], label='Validation Loss', color='crimson')
        axes[0].set_title('Loss Trend')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        axes[1].plot(epochs_range, history['train_acc'], label='Train Accuracy', color='royalblue')
        axes[1].plot(epochs_range, history['val_acc'], label='Validation Accuracy', color='crimson')
        axes[1].set_title('Accuracy Trend')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()