import numpy as np
from flow import Flow6p
import torch

def create_diffeomorphism_mask(flow: Flow6p, threshold: float = -0.01) -> np.ndarray:
    """
    Create a mask showing where diffeomorphism breaks down based on the Jacobian determinant.
    
    Diffeomorphism requires det(J) > 0, where J = I + B (B is the affine_b matrix).
    Values close to or below 0 indicate folding/tearing.
    
    Args:
        flow: Flow6p object with affine parameters
        threshold: Values below this are marked as breakdown (default: -0.01)
    
    Returns:
        mask: uint8 image (0=valid, 255=breakdown)
    """
    # Get affine B matrix: shape (H, W, 2, 2)
    B = flow.affine_b.detach().cpu()
    
    # Compute Jacobian: J = I + B
    I = torch.eye(2).unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 2)
    J = I + B
    
    # Compute determinant at each pixel
    # det([[a,b],[c,d]]) = ad - bc
    det_J = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    
    # Create mask: breakdown where det(J) <= threshold
    breakdown_mask = (det_J <= threshold).numpy().astype(np.uint8) * 255
    
    # Create colored visualization
    # Green = valid (det > 0), Red = breakdown (det <= threshold)
    h, w = breakdown_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color invalid regions red
    colored_mask[breakdown_mask > 0] = [0, 0, 255]
    
    # Color valid regions green (scaled by how positive det is)
    valid_region = det_J > threshold
    det_normalized = torch.clamp(det_J[valid_region] * 100, 0, 255).numpy().astype(np.uint8)
    colored_mask[valid_region.numpy(), 1] = det_normalized
    
    return colored_mask