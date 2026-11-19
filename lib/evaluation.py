import numpy as np

def f1_score(X, X_hat, threshold_activated=1e-5):
    """
    Compute the F1 score for the estimated source variances.
    
    Parameters
    ----------
    X : array-like, shape (n_dipoles, n_timepoints)
        The true source signals over time.
    X_hat : array-like, shape (n_dipoles, n_timepoints)
        The estimated source signals over time.
    threshold_activated : float, optional
        The threshold to determine if a dipole is considered activated based on its variance. Default is
    Returns
    -------
    f1 : float
        The F1 score for the estimated source variances.
    """
    # Identify activated dipoles based on the estimated source signals
    activated_estimated = np.where(np.var(X_hat, axis=1) > threshold_activated)[0]
    
    # Identify activated dipoles based on the true source signals
    activated_true = np.where(np.var(X, axis=1) > threshold_activated)[0]
    
    # Compute true positives, false positives, and false negatives
    tp = len(set(activated_estimated) & set(activated_true))
    fp = len(set(activated_estimated) - set(activated_true))
    fn = len(set(activated_true) - set(activated_estimated))
    
    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1
