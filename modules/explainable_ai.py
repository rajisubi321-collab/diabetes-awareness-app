import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

def calculate_permutation_importance(model, X, y):
    # Subsample is prepared by caller to keep this interactive.
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=1
    )
    return pd.DataFrame({
        "feature": X.columns,
        "importance": result.importances_mean,
        "std": result.importances_std
    }).sort_values("importance", ascending=False).reset_index(drop=True)
