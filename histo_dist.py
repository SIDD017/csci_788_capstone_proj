import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import loggamma


def read_flo_file(file_path: str):
    """Read an optical flow file in .flo format."""
    with open(file_path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError("Invalid .flo file format")
        
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        flow_data = np.fromfile(f, np.float32, count=2 * width * height)
        flow = flow_data.reshape((height, width, 2))
        flow[flow > 1e9] = np.nan
    
    return flow


def warp_by_vector_field(image: np.ndarray, vectors_uv: np.ndarray):
    """Warp an image using the given vector field."""
    h, w = image.shape[:2]
    y, x = np.indices((h, w)).astype(np.float32)
    x_warped = np.clip(x - vectors_uv[..., 0], 0, w - 1)
    y_warped = np.clip(y - vectors_uv[..., 1], 0, h - 1)
    return cv.remap(image, x_warped, y_warped, cv.INTER_LINEAR)


def sliding_window_ar1_helper(t: torch.Tensor, win_size: int):
    """Compute local neighborhood differences for smoothness analysis."""
    t = t.flatten(start_dim=2)
    h, w, d = t.shape
    
    sliding_windows = t.unfold(0, win_size, 1).unfold(1, win_size, 1)
    assert sliding_windows.shape == (
        h - win_size + 1, w - win_size + 1, d, win_size, win_size,
    )
    
    left, right = win_size // 2, w - win_size // 2
    top, bottom = win_size // 2, h - win_size // 2
    central_values = t[top:bottom, left:right]
    
    diff = sliding_windows - central_values.unsqueeze(-1).unsqueeze(-1)
    norm_diff = torch.linalg.norm(diff, dim=2)
    
    return norm_diff


def robust_loss(x, alpha, c):
    """
    General robust loss function from Barron 2019.
    
    Args:
        x: residual values
        alpha: shape parameter controlling robustness
        c: scale parameter
    
    Returns:
        loss values
    """
    eps = 1e-6
    
    if np.abs(alpha - 2) < eps:
        # L2 loss
        return 0.5 * (x / c) ** 2
    elif np.abs(alpha) < eps:
        # Cauchy loss
        return np.log(0.5 * (x / c) ** 2 + 1)
    elif alpha < -1e6:
        # Welsch loss
        return 1 - np.exp(-0.5 * (x / c) ** 2)
    else:
        # General case
        return (np.abs(alpha - 2) / alpha) * (
            ((x / c) ** 2 / np.abs(alpha - 2) + 1) ** (alpha / 2) - 1
        )


def log_partition_approx(alpha):
    """
    Approximation of log partition function for the robust distribution.
    Using cubic spline interpolation for efficiency.
    """
    # Known values at specific alpha
    known_alphas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    known_log_Z = np.array([
        np.log(np.pi * np.sqrt(2)),  # alpha=0 (Cauchy)
        1.15,  # approximate
        1.13,  # approximate for alpha=1
        0.98,  # approximate
        0.5 * np.log(2 * np.pi),  # alpha=2 (Gaussian)
        0.75,  # approximate
        0.85   # approximate
    ])
    
    # Linear interpolation for simplicity
    return np.interp(alpha, known_alphas, known_log_Z)


def nll_robust_distribution(x, alpha, c):
    """
    Negative log-likelihood of the robust distribution.
    """
    return robust_loss(x, alpha, c) + log_partition_approx(alpha) + np.log(c)


def fit_robust_distribution(data, alpha_bounds=(0, 2), init_alpha=1.0, init_c=None):
    """
    Fit the robust distribution to data by minimizing negative log-likelihood.
    
    Args:
        data: observed residuals/errors
        alpha_bounds: bounds for alpha parameter
        init_alpha: initial value for alpha
        init_c: initial value for c (if None, uses MAD estimate)
    
    Returns:
        optimal_alpha, optimal_c
    """
    data = data[~np.isnan(data)].flatten()
    
    if init_c is None:
        # Use median absolute deviation as initial scale estimate
        init_c = np.median(np.abs(data - np.median(data))) * 1.4826
        if init_c < 1e-6:
            init_c = np.std(data)
    
    def objective(params):
        alpha, log_c = params
        c = np.exp(log_c)
        nll = nll_robust_distribution(data, alpha, c)
        return np.sum(nll)
    
    # Optimize
    result = minimize(
        objective,
        x0=[init_alpha, np.log(init_c)],
        bounds=[alpha_bounds, (-10, 10)],
        method='L-BFGS-B'
    )
    
    optimal_alpha = result.x[0]
    optimal_c = np.exp(result.x[1])
    
    return optimal_alpha, optimal_c


def pdf_robust_distribution(x, alpha, c):
    """
    Probability density function of the robust distribution.
    """
    nll = nll_robust_distribution(x, alpha, c)
    return np.exp(-nll)


def plot_with_fitted_distribution(data, title, alpha=None, c=None, fit=True):
    """
    Plot histogram with fitted robust distribution overlay.
    """
    data = data[~np.isnan(data)].flatten()
    
    if fit:
        # Fit distribution to data
        alpha_opt, c_opt = fit_robust_distribution(data)
        print(f"{title}:")
        print(f"  Optimal α = {alpha_opt:.4f}")
        print(f"  Optimal c = {c_opt:.6f}")
    else:
        alpha_opt, c_opt = alpha, c
        print(f"{title}: using α = {alpha_opt:.4f}, c = {c_opt:.6f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        data, bins=100, density=True, alpha=0.6, 
        color='steelblue', edgecolor='black', linewidth=0.5
    )
    
    # Overlay fitted PDF
    x_range = np.linspace(bins[0], bins[-1], 1000)
    pdf_values = pdf_robust_distribution(x_range, alpha_opt, c_opt)
    plt.plot(x_range, pdf_values, 'r-', linewidth=2.5, 
             label=f'Fitted PDF (α={alpha_opt:.3f}, c={c_opt:.4f})')
    
    # Also show Gaussian for comparison
    mu, sigma = np.mean(data), np.std(data)
    gaussian_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    plt.plot(x_range, gaussian_pdf, 'g--', linewidth=2, 
             label=f'Gaussian (μ={mu:.4f}, σ={sigma:.4f})', alpha=0.7)
    
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return alpha_opt, c_opt


# Main analysis
if __name__ == "__main__":
    # Configuration
    IMAGE = "Venus"
    
    # Load data
    print(f"Loading data for {IMAGE}...")
    flo = read_flo_file(f"./ground_truth_images/{IMAGE}/flow10.flo")
    img1 = cv.imread(f"./images/{IMAGE}/frame10.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(f"./images/{IMAGE}/frame11.png", cv.IMREAD_GRAYSCALE)
    pred_img2 = warp_by_vector_field(img1, flo)
    
    # ============ 1. Data Term Error Distribution ============
    err = img2 / 255.0 - pred_img2 / 255.0
    alpha_data, c_data = plot_with_fitted_distribution(
        err, 
        f"Data Term Errors [{IMAGE}] - Fitted Robust Distribution",
        fit=True
    )
    plt.savefig(f'data_errors_{IMAGE}_fitted.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============ 2. Flow Magnitude Distribution (AR0) ============
    mag_uv = np.linalg.norm(flo, axis=-1)
    alpha_mag, c_mag = plot_with_fitted_distribution(
        mag_uv,
        f"Flow Magnitudes [{IMAGE}] - Fitted Robust Distribution",
        fit=True
    )
    plt.savefig(f'flow_magnitudes_{IMAGE}_fitted.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============ 3. Smoothness Distribution (AR1) ============
    mag_uv_3 = sliding_window_ar1_helper(torch.from_numpy(flo), 3)
    mag_uv_7 = sliding_window_ar1_helper(torch.from_numpy(flo), 7)
    
    # Fit to 3x3 window
    alpha_smooth3, c_smooth3 = plot_with_fitted_distribution(
        mag_uv_3.numpy(),
        f"Flow Local Differences 3x3 [{IMAGE}] - Fitted Robust Distribution",
        fit=True
    )
    plt.savefig(f'smoothness_3x3_{IMAGE}_fitted.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Fit to 7x7 window
    alpha_smooth7, c_smooth7 = plot_with_fitted_distribution(
        mag_uv_7.numpy(),
        f"Flow Local Differences 7x7 [{IMAGE}] - Fitted Robust Distribution",
        fit=True
    )
    plt.savefig(f'smoothness_7x7_{IMAGE}_fitted.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============ 4. Combined Comparison Plot ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot all distributions together
    datasets = [
        (err, "Data Errors", alpha_data, c_data),
        (mag_uv, "Flow Magnitudes", alpha_mag, c_mag),
        (mag_uv_3.numpy(), "Smoothness 3x3", alpha_smooth3, c_smooth3),
        (mag_uv_7.numpy(), "Smoothness 7x7", alpha_smooth7, c_smooth7)
    ]
    
    for ax, (data, name, alpha, c) in zip(axes.flat, datasets):
        data_clean = data[~np.isnan(data)].flatten()
        ax.hist(data_clean, bins=80, density=True, alpha=0.5, color='steelblue')
        
        x_range = np.linspace(np.percentile(data_clean, 1), 
                              np.percentile(data_clean, 99), 500)
        pdf_vals = pdf_robust_distribution(x_range, alpha, c)
        ax.plot(x_range, pdf_vals, 'r-', linewidth=2, 
                label=f'α={alpha:.3f}, c={c:.4f}')
        
        ax.set_title(name, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fitted Robust Distributions - {IMAGE}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'all_distributions_{IMAGE}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============ Summary ============
    print("\n" + "="*60)
    print("SUMMARY OF OPTIMAL PARAMETERS")
    print("="*60)
    print(f"Data Term:       α = {alpha_data:.4f}, c = {c_data:.6f}")
    print(f"Flow Magnitude:  α = {alpha_mag:.4f}, c = {c_mag:.6f}")
    print(f"Smoothness 3x3:  α = {alpha_smooth3:.4f}, c = {c_smooth3:.6f}")
    print(f"Smoothness 7x7:  α = {alpha_smooth7:.4f}, c = {c_smooth7:.6f}")
    print("="*60)