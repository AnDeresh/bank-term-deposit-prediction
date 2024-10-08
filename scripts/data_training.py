import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Function to train and evaluate a machine learning model
def train_and_evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: Optional[float] = None,
    model_name: Optional[str] = None,
    hyperparameters: Optional[Dict] = None,
    results_table: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """
    Train the model, evaluate its performance, and visualize the results using an adjustable threshold.

    Parameters:
    model (BaseEstimator): The machine learning model to train.
    X_train (pd.DataFrame): Training feature set.
    X_test (pd.DataFrame): Test feature set.
    y_train (pd.Series): Training labels.
    y_test (pd.Series): Test labels.
    threshold (Optional[float]): Custom decision threshold for predictions. Defaults to None (uses 0.5).
    model_name (Optional[str]): Name of the model for results tracking. Defaults to None.
    hyperparameters (Optional[Dict]): Model hyperparameters for results tracking. Defaults to None.
    results_table (Optional[pd.DataFrame]): DataFrame to store evaluation results. Defaults to None.

    Returns:
    Optional[pd.DataFrame]: The updated results table with new evaluation metrics if provided.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set using the default or custom threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    if threshold is None:
        y_pred = model.predict(X_test)
    else:
        y_pred = (y_proba >= threshold).astype(int)

    # Calculate F1-scores for train and test sets
    f1_train: float = round(f1_score(y_train, model.predict(X_train)), 2)
    f1_test: float = round(f1_score(y_test, y_pred), 2)
    print(f"F1 Score (Test): {f1_test:.2f}")

    # Calculate ROC AUC score
    roc_auc: float = round(roc_auc_score(y_test, y_proba), 2)
    print(f"ROC AUC Score: {roc_auc:.2f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # If a results table is provided, store the evaluation results
    if results_table is not None and model_name is not None and hyperparameters is not None:
        new_results = pd.DataFrame({
            'Model': [model_name],
            'Hyperparameters': [hyperparameters],
            'F1 Score (Train)': [f1_train],
            'F1 Score (Test)': [f1_test],
            'ROC AUC Score': [roc_auc],
            'Threshold': [round(threshold, 2) if threshold is not None else 0.5],
            'Comments': [None]
        })
        results_table = pd.concat([results_table, new_results], ignore_index=True)

    return results_table

# Define a function to update comments for a specific model in the results table
def update_comments(results_table: pd.DataFrame, model_name: str, comment: Optional[str]) -> pd.DataFrame:
    """
    Update comments for a specific model in the results table.

    Parameters:
    results_table (pd.DataFrame): DataFrame containing model evaluation results.
    model_name (str): Name of the model for which the comment should be updated.
    comment (Optional[str]): The comment to add or update. If None, it will clear the comment.

    Returns:
    pd.DataFrame: Updated results table with the modified comment.
    """
    # Check if the specified model exists in the results table
    if model_name not in results_table['Model'].values:
        raise ValueError(f"Model '{model_name}' not found in the results table.")

    # Update the 'Comments' column for the specified model
    results_table.loc[results_table['Model'] == model_name, 'Comments'] = comment

    return results_table
