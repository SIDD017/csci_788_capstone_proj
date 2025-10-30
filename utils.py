import cv2 as cv
import numpy as np
import struct
import torch.nn.functional as F

import torch


def _flatten_dict(d: dict, key_sep="_") -> dict:
    """Flattens a nested dictionary."""
    out = {}

    def flatten(x: dict, name: str = ""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + key_sep)
        else:
            if name[:-1] in out:
                raise ValueError(
                    f"Duplicate key created during flattening: {name[:-1]}"
                )
            out[name[:-1]] = x

    flatten(d)
    return out


def np_im_to_torch(image_np):
    # TODO - convert to float if needed
    return torch.from_numpy(np.atleast_3d(image_np)).permute(2,0,1).unsqueeze(0)


def uint8_to_float32(image):
    return image.astype(np.float32) / 255.0


def convert_torch_to_cv(image_tensor):
    # image_tensor shape is [1, C, H, W]
    img_np = image_tensor.cpu().numpy()[0]           # shape: [C, H, W]
    img_np = np.transpose(img_np, (1, 2, 0))           # shape: [H, W, C]
    # If single channel, convert to 2D
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]
    # Assume image data is in [0,1] float range; scale to [0,255] and convert to uint8
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
    return img_np


def debug_edge_weighting(w_edge):
    v_edge = w_edge.clone().cpu().numpy()[0,0]
    v_edge = (np.clip(v_edge, 0, 1) * 255).astype(np.uint8)
    cv.imshow("downweight edge", v_edge)
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit(0)


def warp_image_with_flow(flow, image=None):
    if image is None:
        image = flow.image2
    B, C, H, W = image.shape
    dev, dt = image.device, image.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    x_new, y_new = flow.warped_coords(xs, ys)
    # Convert to [-1,1] range for grid_sample
    gx = (x_new / (W - 1)) * 2 - 1
    gy = (y_new / (H - 1)) * 2 - 1
    # Make size (1,H,W,2), expected by grid_sample
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(image, grid, align_corners=True, mode="bilinear", padding_mode="border")

def read_flo_file(filepath):
    with open(filepath, 'rb') as f:
        # Read header
        tag = struct.unpack('f', f.read(4))[0]
        if tag != 202021.25:
            raise ValueError(f"Invalid .flo file tag: {tag}, expected 202021.25")
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        # Read flow data
        flow_data = struct.unpack('f' * (width * height * 2), f.read(width * height * 2 * 4))
        # Reshape to (height, width, 2)
        flow = np.array(flow_data, dtype=np.float32).reshape(height, width, 2)
    return flow


def visualize_flow_hsv(flow_uv, max_magnitude = None):
    nan_mask = np.any(np.isnan(flow_uv), axis=2)
    flow_uv[nan_mask] = 0
    magnitude = np.linalg.norm(flow_uv, axis=2)
    if max_magnitude is None:
        max_magnitude = np.max(magnitude)
    angle = np.arctan2(-flow_uv[..., 1], -flow_uv[..., 0])
    hsv = np.zeros(flow_uv.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) * 180 / np.pi / 2
    hsv[..., 1] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    hsv[nan_mask, :] = 0
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def visualize_gt_flow_hsv(flow_uv, max_magnitude= None):
    # Handle invalid/unknown flow values (commonly marked as very large values in Middlebury)
    invalid_mask = np.logical_or(
        np.abs(flow_uv[..., 0]) > 1e9,
        np.abs(flow_uv[..., 1]) > 1e9
    )
    # Also handle NaN values
    nan_mask = np.any(np.isnan(flow_uv), axis=2)
    invalid_mask = np.logical_or(invalid_mask, nan_mask)
    # Create a copy and set invalid values to 0
    flow_clean = flow_uv.copy()
    flow_clean[invalid_mask] = 0
    # Calculate magnitude and angle
    magnitude = np.linalg.norm(flow_clean, axis=2)
    if max_magnitude is None:
        max_magnitude = np.max(magnitude[~invalid_mask]) if np.any(~invalid_mask) else 1.0
    
    angle = np.arctan2(-flow_clean[..., 1], -flow_clean[..., 0])
    
    # Create HSV image
    hsv = np.zeros(flow_uv.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) * 180 / np.pi / 2).astype(np.uint8)  # Hue
    hsv[..., 1] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)  # Saturation
    hsv[..., 2] = 255  # Value
    
    # Set invalid pixels to black
    hsv[invalid_mask, :] = 0
    
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)