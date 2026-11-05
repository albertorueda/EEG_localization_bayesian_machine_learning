import mne
import numpy as np
import matplotlib.pyplot as plt

def get_leadfield():
    """
    Get the leadfield matrix for a standard EEG setup.
    
    Returns
    -------
    L : array-like, shape (n_sensors, n_dipoles)
        The leadfield matrix mapping dipoles to sensors.
    """
    # Create a standard EEG montage and source space
    montage = mne.channels.make_standard_montage("colin27_1020")
    chs = [c for c in montage.ch_names if c not in ["T3", "T4", "T5", "T6"]]
    info = mne.create_info(chs, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)
    
    # Create a spherical head model and source space
    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    src = mne.setup_volume_source_space(
        sphere=sphere, pos=15.0, exclude=30.0, add_interpolator=False
    )
    
    # Create the forward solution (leadfield matrix)
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, meg=False, eeg=True, mindist=5.0
    )
    L = fwd["sol"]["data"]
    
    return L

def activate_original_dipoles(n_dipoles, n_active, info_source_space, amplitude=0.5):
    """
    Randomly activate a specified number of dipoles.

    Parameters
    ----------
    n_dipoles : int
        Total number of dipoles.
    n_active : int
        Number of dipoles to activate.
    info_source_space : mne.SourceSpaces
        The source space information.

    Returns
    -------
    X : array-like, shape (n_dipoles, n_timepoints)
        The source signals with activated dipoles.
    """
    indices = np.random.choice(n_dipoles, size=n_active, replace=False)
    t = np.linspace(0, 1, int(info_source_space["sfreq"]), endpoint=False)

    # Signal of the indices dipoles
    x_1 = amplitude * np.sin(2 * np.pi * t)
    X = np.zeros((n_dipoles, int(info_source_space["sfreq"])))
    for idx in indices:
        X[idx] = x_1
        
    return X

def forward_model(L, X, noise_std=0.1):
    """
    Generate sensor measurements from source signals using the leadfield matrix.

    Parameters
    ----------
    L : array-like, shape (n_sensors, n_dipoles)
        The leadfield matrix mapping dipoles to sensors.
    X : array-like, shape (n_dipoles, n_timepoints)
        The source signals over time.
    noise_std : float
        Standard deviation of Gaussian noise to add to the measurements.

    Returns
    -------
    y : array-like, shape (n_sensors, n_timepoints)
        The sensor measurements with added noise.
    """
    y = L @ X + np.random.normal(0, noise_std, size=(L.shape[0], X.shape[1]))
    return y

def champagne(y, leadfield, beta, phi_init, max_iter=1000, tol=1e-6):
    """
    Champagne algorithm for source localization.

    Parameters
    ----------
    y : array-like, shape (n_sensors, n_timepoints)
        The sensor measurements over time.
    leadfield : array-like, shape (n_sensors, n_dipoles)
        The leadfield matrix mapping dipoles to sensors.
    beta : float
        Regularization parameter.
    phi_init : array-like, shape (n_dipoles,n_dipoles)
        Initial values for the source variances.
    max_iter : int, optional
        Maximum number of iterations for the algorithm (default is 1000).
    tol : float, optional
        Tolerance for convergence (default is 1e-6).

    Returns
    phi : array-like, shape (n_dipoles,n_dipoles)
        Estimated source variances after convergence.
    -------
    """

    n_dipoles = leadfield.shape[1]
    phi = phi_init.copy()
    
    leadfield_mult = leadfield.T @ leadfield
    leadfield_y = leadfield.T @ y

    # Iterative update of source variances
    for _ in range(max_iter):  # Maximum number of iterations
        # Save the current phi for convergence check
        phi_prev = phi.copy()
        
        # Compute Gamma
        Gamma = phi + beta * leadfield_mult
        
        # Compute x_hat
        x_hat = beta * np.linalg.solve(Gamma, leadfield_y)
        
        # Update phi
        full_matrix = beta * np.linalg.solve(Gamma, leadfield_mult)
        
        # 0's in every row and column except for the diagonal
        for j in range(n_dipoles):
            num = full_matrix[j][j]
            phi[j][j] = num/((np.sum(x_hat[j] ** 2) + 1e-10)/x_hat.shape[1])  # Add small constant to avoid division by zero
        
        # Check for convergence
        if np.linalg.norm(phi - phi_prev) < tol:
            break

    return phi

def f1_score(X, phi_hat, threshold_source_variance=1e5, threshold_true_variance=1e-5):
    """
    Compute the F1 score for the estimated source variances.
    
    Parameters
    ----------
    X : array-like, shape (n_dipoles, n_timepoints)
        The true source signals over time.
    phi_hat : array-like, shape (n_dipoles, n_dipoles)
        The estimated source variances after convergence.
    Returns
    -------
    f1 : float
        The F1 score for the estimated source variances.
    """
    # Identify activated dipoles based on the estimated source variances
    activated_estimated = np.where(phi_hat.diagonal() < threshold_source_variance)[0]
    
    # Identify activated dipoles based on the true source signals
    activated_true = np.where([np.var(X[i]) > threshold_true_variance for i in range(X.shape[0])])[0]
    
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

def main():
    # Get the leadfield matrix
    L = get_leadfield()
    
    # Define parameters
    n_dipoles = L.shape[1]
    beta = 4  # Regularization parameter
    phi_init = np.eye(n_dipoles) * 1e-3  # Initial source variances
    
    f1_scores = {}
    for n_active in range(1, n_dipoles + 1, 5):  # Vary the number of active dipoles from 1 to n_dipoles in steps of 5
        # Activate original dipoles
        X = activate_original_dipoles(n_dipoles, n_active, info_source_space=L)
        
        # Generate sensor measurements
        y = forward_model(L, X, noise_std=1/beta)
        
        # Run Champagne algorithm
        phi_hat = champagne(y, L, beta, phi_init)
        
        # Compute F1 score
        f1 = f1_score(X, phi_hat)
        f1_scores[n_active] = f1
        
        # Print the F1 score for the current number of active dipoles
        print(f"Number of Active Dipoles: {n_active}, F1 Score: {f1:.4f}")
        
    # Save F1 scores to a file
    np.save("f1_scores.npy", f1_scores)
    
    # Plot and save image of F1 scores
    plt.figure()
    plt.plot(list(f1_scores.keys()), list(f1_scores.values()), marker='o')
    plt.title("F1 Score vs Number of Active Dipoles")
    plt.xlabel("Number of Active Dipoles")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig("f1_scores_plot.png")