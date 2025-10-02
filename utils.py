import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
import struct


def process_args():
    parser = argparse.ArgumentParser(description="Two-frame optical flow")
    parser.add_argument("image1", help="First image", type=Path)
    parser.add_argument("image2", help="Second image", type=Path)
    parser.add_argument("gtimage", help="Ground truth flow file", type=Path)
    parser.add_argument("--levels", help="Number of pyramid levels", type=int, default=5)
    parser.add_argument("--window_size", help="Window size", type=int, default=7)
    parser.add_argument("--alpha", help="Regularization parameter", type=float, default=1e-3)
    parser.add_argument("--use_affine", help="Use affine refinement or translation", type=bool, default=False)
    parser.add_argument("--opencv_init", help="Calculate initial flow using opencv method", type=bool, default=False)
    parser.add_argument(
        "--goodness-threshold",
        help="Mismatch threshold for forward/reverse flow to be 'good'",
        type=float,
        default=2.0,
    )
    args = parser.parse_args()

    # Read the images
    args.image1 = uint8_to_float32(cv.imread(str(args.image1), cv.IMREAD_GRAYSCALE))
    args.image2 = uint8_to_float32(cv.imread(str(args.image2), cv.IMREAD_GRAYSCALE))

    #TODO: Assert checks = shape, dims, types, etc
    return args


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


def uint8_to_float32(image):
    return image.astype(np.float32) / 255.0



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