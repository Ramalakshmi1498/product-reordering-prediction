# src/evaluate.py

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the trained model using classification metrics.
    
    Args:
        model: Trained Keras model.
        X_val: Validation features.
        y_val: True labels for validation set.
    """
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    
    # Metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Reordered", "Reordered"], yticklabels=["Not Reordered", "Reordered"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
