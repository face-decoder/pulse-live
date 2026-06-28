import pandas as pd
from sklearn.metrics import classification_report
from IPython.display import display, HTML

class TrainSummaryDisplay:
    def show(self, summary_df=None):
        if summary_df is None:
            display(HTML("<p style='color: red;'>No summary data to display.</p>"))
            return
        
        
        train_summary_df = pd.DataFrame(
            {
                "metric": [
                    "best_epoch",
                    "best_ema_val_f1",
                    "best_val_f1",
                    "best_val_bacc",
                    "best_val_loss",
                    "best_val_acc",
                    "best_threshold",
                    "best_threshold_metric",
                    "best_threshold_score",
                ],
                "value": [
                    summary_df["best_epoch"],
                    summary_df["best_ema_val_f1"],
                    summary_df["best_val_f1"],
                    summary_df["best_val_bacc"],
                    summary_df["best_val_loss"],
                    summary_df["best_val_acc"],
                    summary_df["best_threshold"],
                    summary_df["best_threshold_metric"],
                    summary_df["best_threshold_score"],
                ],
            }
        )

        display(train_summary_df)