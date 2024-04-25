import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
import torch
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc


def compute_accuracy(predictions, labels):
    # Convert predictions to binary labels (0 or 1)
    predicted_labels = torch.round(predictions)
    # Compare predicted labels with ground truth labels
    correct_count = (predicted_labels == labels).sum().item()
    total_count = labels.size(0)
    # Compute accuracy
    accuracy = correct_count / total_count
    return accuracy


def compute_auroc(predictions, labels):
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    try:
        auroc = roc_auc_score(y_true=labels, y_score=predictions)
        return auroc
    except ValueError as e:
        print("Auroc computation failed, ", e)
        return 0.5


def multi_class_eval(scores, labels, K):
    with torch.no_grad():
        accuracy_macro = accuracy(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        accuracy_micro = accuracy(preds=scores, target=labels, average='micro', task='multiclass', num_classes=K).item()
        preds = np.argmax(scores.cpu().numpy(), axis=1)
        accuracy_balanced = balanced_accuracy_score(y_true=labels.cpu().numpy(), y_pred=preds)

        precision_macro = precision(preds=scores, target=labels, average='macro', task='multiclass',
                                    num_classes=K).item()
        precision_micro = precision(preds=scores, target=labels, average='micro', task='multiclass',
                                    num_classes=K).item()

        recall_macro = recall(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        recall_micro = recall(preds=scores, target=labels, average='micro', task='multiclass', num_classes=K).item()

        f1_macro = f1_score(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()
        f1_micro = f1_score(preds=scores, target=labels, average='micro', task='multiclass', num_classes=K).item()

        auroc_macro = auroc(preds=scores, target=labels, average='macro', task='multiclass', num_classes=K).item()

    return (accuracy_macro, accuracy_micro, accuracy_balanced, precision_macro, precision_micro, recall_macro,
            recall_micro, f1_macro, f1_micro, auroc_macro,)
