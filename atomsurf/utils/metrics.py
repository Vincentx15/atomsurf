import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
import torch
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc


def compute_accuracy(predictions, labels, add_sigmoid=False):
    # Convert predictions to binary labels (0 or 1)
    if add_sigmoid:
        predictions = torch.sigmoid(predictions)
    predicted_labels = torch.round(predictions)
    # Compare predicted labels with ground truth labels
    correct_count = (predicted_labels == labels).sum().item()
    total_count = labels.size(0)
    # Compute accuracy
    if total_count>0:
        accuracy = correct_count / total_count
    else:
        print('no number')
        accuracy=0.5
    return accuracy
def compute_f1metrics(predictions, labels, threshold=0.5, add_sigmoid=True, eps=1e-8):
    """
    Compute precision, recall, and F1 score for binary classification.

    Args:
        predictions (Tensor): Raw logits or probabilities, shape [N] or [N, 1]
        labels (Tensor): Binary ground truth labels (0 or 1), same shape
        threshold (float): Threshold for converting probabilities to binary (default: 0.5)
        add_sigmoid (bool): Whether to apply sigmoid to predictions
        eps (float): Small value to avoid division by zero

    Returns:
        dict with keys: 'precision', 'recall', 'f1'
    """
    if add_sigmoid:
        predictions = torch.sigmoid(predictions)

    predicted_labels = (predictions > threshold).int()
    labels = labels.int()

    tp = ((predicted_labels == 1) & (labels == 1)).sum().item()
    fp = ((predicted_labels == 1) & (labels == 0)).sum().item()
    fn = ((predicted_labels == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

from sklearn.metrics import average_precision_score, roc_auc_score

def compute_auc_metrics(pred_probs, labels):
    """
    pred_probs: Tensor or np.array of probabilities (after sigmoid)
    labels: Ground-truth binary labels (0 or 1)

    Returns: dict with AUROC and AUPRC
    """
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    try:
        auroc = roc_auc_score(labels, pred_probs)
        auprc = average_precision_score(labels, pred_probs)
        return auroc,auprc
    except ValueError as e:
        print("Auroc computation failed, ", e)
        return 0.5,0.5
    

def compute_auroc(predictions, labels):
    labels = labels.detach().cpu().numpy().astype(int)
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
