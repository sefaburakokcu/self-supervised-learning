from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Comprehensive evaluation metrics"""

    def __init__(self, class_names):
        self.class_names = class_names

    def compute_all_metrics(self, y_true, y_pred):
        """Compute all classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
        }

        # Per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None)
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i]

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        ))