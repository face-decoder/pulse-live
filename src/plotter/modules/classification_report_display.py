import pandas as pd
from sklearn.metrics import classification_report
from IPython.display import display, HTML

class ClassificationReportDisplay:
    """Class tunggal untuk merender HTML Table metrik (F1, Precision, Recall)."""
    def __init__(self, target_names):
        self.target_names = target_names

    def show(self, y_true, y_pred, loss=None, acc=None):
        # Dictionary untuk classification report dengan output_dict=True
        report_dict = classification_report(y_true, y_pred, target_names=self.target_names, output_dict=True, zero_division=0)
        
        # Konversi dictionary ke DataFrame untuk tampilan yang lebih baik
        df_report = pd.DataFrame(report_dict).transpose()
        
        header = "<h4>Classification Report</h4>"
        if loss is not None and acc is not None:
            header = f"<h4>Metrics (Loss: {loss:.4f} | Accuracy: {acc:.4f})</h4>"
        
        # Tampilkan header dan DataFrame sebagai HTML
        display(HTML(header))

        # Tampilkan DataFrame report klasifikasi
        display(df_report)