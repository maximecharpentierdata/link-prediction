from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def show_roc_and_f1(test_labels: List[int], test_preds: List[float]):
    fpr, tpr, thresholds = roc_curve(test_labels, test_preds)
    roc_auc = roc_auc_score(test_labels, test_preds)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkred", label="ROC curve (area = %0.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.show()

    th = thresholds[np.argmax(tpr - fpr)]
    print("f1 score", f1_score(test_labels, [pred > th for pred in test_preds]))
    return th


def generate_submission_data(submit_preds: List[int], name: str):
    submission_df = pd.DataFrame(dict(category=submit_preds))
    submission_df = submission_df.reset_index()
    submission_df.columns = ["id", "category"]
    submission_df.to_csv(f"submissions/{name}.csv", index=None)
