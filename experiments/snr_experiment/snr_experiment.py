import sys
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cho_factor, cho_solve

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
    
    return info, L

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
    n_timepoints = y.shape[1]
    phi = phi_init.copy()
    
    leadfield_mult = leadfield.T @ leadfield
    leadfield_y = leadfield.T @ y

    # Iterative update of source variances
    for _ in range(max_iter):  # Maximum number of iterations
        # Save the current phi for convergence check
        phi_prev = phi.copy()
        
        # Compute Gamma
        Gamma = beta * leadfield_mult
        Gamma[np.diag_indices_from(Gamma)] += phi

        # Compute x_hat with Cholesky decomposition for numerical stability
        c_and_lower = cho_factor(Gamma, check_finite=False)
        x_hat = beta * cho_solve(c_and_lower, leadfield_y, check_finite=False)
        
        # Update phi
        full_matrix = beta * cho_solve(c_and_lower, leadfield_mult, check_finite=False)
        phi = np.diag(full_matrix) / ((np.sum(x_hat ** 2, axis=1) + 1e-10) / n_timepoints)
        
        # Check for convergence
        if np.linalg.norm(phi - phi_prev) < tol * np.linalg.norm(phi_prev):
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
    activated_estimated = np.where(phi_hat < threshold_source_variance)[0]
    
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

# Element k in the (k = 1, 2, ..., N_VALUES * N_REPS) array
# translate to (value of n_active, repetición):
#   value_idx = (k - 1) // N_REPS      ->  n_active = START + STEP * value_idx
#   rep       = (k - 1) %  N_REPS
# With START=-20, STEP=5, N_VALUES=13 and N_REPS=10 -> array [-20, -15, ..., 45]
START = -20
STEP = 5
N_VALUES = 17
N_REPS = 10

def main():
    # Index of the job in the array [1, 2, ..., N_VALUES * N_REPS]
    job_index = int(sys.argv[1])
    value_idx = (job_index - 1) // N_REPS
    rep = (job_index - 1) % N_REPS
    assert value_idx < N_VALUES, "El índice del array excede N_VALUES * N_REPS"
    snr = START + STEP * value_idx
    np.random.seed(job_index)
    
    # Get the leadfield matrix
    info, L = get_leadfield()
    
    # Define parameters
    n_dipoles = L.shape[1]
    beta = 4  # Regularization parameter
    phi_init = np.ones(n_dipoles) * 1e-3  # Initial source variances
    
    start_time = time.time()
    print(f"Job {job_index}: snr={snr}, repetition {rep + 1}/{N_REPS}...", flush=True)
 
    # Activate original dipoles
    X = activate_original_dipoles(n_dipoles, 10, info_source_space=info)
 
    # Generate sensor measurements
    signal_power = np.mean((L @ X) ** 2)
    noise_std = np.sqrt(signal_power / 10 ** (snr / 10))
    beta = 1.0 / noise_std ** 2
    y = forward_model(L, X, noise_std=noise_std)
 
    # Run Champagne algorithm
    phi_hat = champagne(y, L, beta, phi_init)
 
    # Compute F1 score
    f1 = f1_score(X, phi_hat)
 
    elapsed = time.time() - start_time
    print(f"SNR: {snr}, Rep: {rep}, F1 Score: {f1:.4f}, Time: {elapsed:.2f} seconds", flush=True)
 
    # Save the result of THIS experiment in its own file
    os.makedirs("results", exist_ok=True)
    np.savez(f"results/f1_n{snr:04d}_r{rep:02d}.npz",
             snr=snr, rep=rep, f1=f1, time=elapsed)
    
if __name__ == "__main__":
    main()