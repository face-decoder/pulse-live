import pandas as pd
from IPython.display import HTML, display
from sklearn.metrics import classification_report


class ClassificationReportDisplay:
    def __init__(self, target_names):
        self.target_names = target_names

    def show(self, y_true, y_pred, loss=None, acc=None):
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.target_names,
            output_dict=True,
            zero_division=0,
        )

        df_report = pd.DataFrame(report_dict).transpose()

        header = "<h4>Classification Report</h4>"
        if loss is not None and acc is not None:
            header = f"<h4>Metrics (Loss: {loss:.4f} | Accuracy: {acc:.4f})</h4>"

        display(HTML(header))

        display(df_report)
