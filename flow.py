import cv2 as cv
import numpy as np
import argparse
from pathlib import Path
from ground_truth_flo import read_flo_file, visualize_gt_flow_hsv

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


def resize_for_pyramid(image, levels):
    height, width = image.shape[:2]
    divisible_by = 2 ** (levels - 1)
    new_height = int(np.ceil(height / divisible_by) * divisible_by)
    new_width = int(np.ceil(width / divisible_by) * divisible_by)
    return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)


def gaussian_pyramid(image, levels, resize = True):
    if resize:
        image = resize_for_pyramid(image, levels)
    pyr = [image]
    for k in range(levels - 1):
        pyr.append(cv.pyrDown(pyr[k]))
    return pyr


def solve_optical_flow_constraint_equation(Ix, Iy, It, alpha = 1e-3, axis=-1):
    IxIx = np.sum(Ix * Ix, axis=axis)
    IxIy = np.sum(Ix * Iy, axis=axis)
    IyIy = np.sum(Iy * Iy, axis=axis)
    IxIt = np.sum(Ix * It, axis=axis)
    IyIt = np.sum(Iy * It, axis=axis)

    m00 = IxIx + alpha
    m01 = IxIy
    m10 = IxIy
    m11 = IyIy + alpha
    det_m = m00 * m11 - m01 * m10

    #Shoudln't we check for det_m == 0?

    # [u, v] = -inv(M) * [IxIt, IyIt]
    u = -(m11 * IxIt - m01 * IyIt) / det_m
    v = -(-m10 * IxIt + m00 * IyIt) / det_m

    return np.stack([u, v], axis=axis)


def warp_with_flow(image, flow_uv):
    h, w = image.shape[:2]
    x, y = np.meshgrid(range(w), range(h))
    x_warped = x + flow_uv[..., 0]
    y_warped = y + flow_uv[..., 1]

    # Logic of cv.remap is that it will calculate g[y,x] = f[y_warped[y,x], x_warped[y,x]]; since
    # we're calculating x_warped and y_warped from the forward flow, this means that g[y,x] behaves
    # like a reverse warp.
    return cv.remap(
        image, x_warped.astype(np.float32), y_warped.astype(np.float32), cv.INTER_LINEAR
    )


def calculate_gradients(image1, image2):
    # Unlike with edge detection, we care here about the 'units' of the gradients being correct
    # (∆brightness per pixel). This requires scaling the sobel kernel appropriately.
    # For gradient_x and gradient_y, we can get a better estimate by averaging the two images.
    gradient_x = (
        cv.Sobel(image1, cv.CV_32F, 1, 0, ksize=3, scale=1 / 8)
        + cv.Sobel(image2, cv.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    ) / 2
    gradient_y = (
        cv.Sobel(image1, cv.CV_32F, 0, 1, ksize=3, scale=1 / 8)
        + cv.Sobel(image2, cv.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    ) / 2
    # For gradient_t, we just take the difference; this will have units of change in brightness
    # per frame.
    gradient_t = image2 - image1
    return gradient_x, gradient_y, gradient_t


def sliding_window_view(image, window_size, *pad_args, **pad_kwargs):
    h, w = image.shape[:2]
    half_window = window_size // 2
    padded_image = np.pad(image, *pad_args, **pad_kwargs, pad_width=half_window)
    return np.lib.stride_tricks.sliding_window_view(
        padded_image, (window_size, window_size), axis=(0, 1)
    ).reshape(h, w, window_size**2)


def estimate_optical_flow(image1, image2, window, initial_flow_uv = None, alpha = 1e-3):
    h, w = image1.shape[:2]
    if initial_flow_uv is None:
        flow_uv = np.zeros((h, w, 2), dtype=np.float32)
    else:
        flow_uv = initial_flow_uv.copy()

    # Warp the second image using the initial flow
    image2_warped = warp_with_flow(image2, flow_uv)

    # Calculate the spatial gradients
    gradient_x, gradient_y, gradient_t = calculate_gradients(image1, image2_warped)

    # Slice a window x window region around each pixel. Result is shape (h, w, window**2)
    grad_x_windows = sliding_window_view(gradient_x, window, mode="reflect")
    grad_y_windows = sliding_window_view(gradient_y, window, mode="reflect")
    grad_t_windows = sliding_window_view(gradient_t, window, mode="reflect")

    # Calculate the update to the optical flow
    flow_uv_temp = solve_optical_flow_constraint_equation(
        grad_x_windows, grad_y_windows, grad_t_windows, alpha=alpha, axis=-1
    )

    return flow_uv + flow_uv_temp


def coarse_to_fine_optical_flow(
    image1,
    image2,
    levels,
    window_size,
    alpha = 1e-3,
):
    original_size = image1.shape[:2]
    pyramid1 = gaussian_pyramid(image1, levels, resize=True)
    pyramid2 = gaussian_pyramid(image2, levels, resize=True)

    # Start with the coarsest level
    flow_uv = estimate_optical_flow(pyramid1[-1], pyramid2[-1], window_size, alpha=alpha)

    # Work backwards, refining the flow at each level
    for k in range(levels - 2, -1, -1):
        flow_uv = cv.pyrUp(flow_uv)
        flow_uv *= 2

        flow_uv = estimate_optical_flow(
            pyramid1[k], pyramid2[k], window_size, initial_flow_uv=flow_uv, alpha=alpha
        )

    return cv.resize(flow_uv, original_size[::-1], interpolation=cv.INTER_LINEAR)


def calculate_forward_reverse_flow(
    image1,
    image2,
    levels,
    window_size,
    alpha,
    goodness_threshold,
):
    # Calculate the forward optical flow
    forward_flow = coarse_to_fine_optical_flow(image1, image2, levels, window_size, alpha)
    # Calculate the reverse optical flow
    reverse_flow = coarse_to_fine_optical_flow(image2, image1, levels, window_size, alpha)
    # Catch occlusion issues and other bad estimates by checking for places where forward and
    # reverse flow disagree
    is_bad_flow_estimate = np.linalg.norm(forward_flow + reverse_flow, axis=2) > goodness_threshold

    # Final estimate of the optical flow: average forward and reverse, then set poorly-estimated
    # regions to nan
    flow = (forward_flow - reverse_flow) / 2
    flow[is_bad_flow_estimate] = np.nan

    return flow


def process_args():
    parser = argparse.ArgumentParser(description="Two-frame optical flow")
    parser.add_argument("image1", help="First image", type=Path)
    parser.add_argument("image2", help="Second image", type=Path)
    parser.add_argument("gtimage", help="Ground truth flow file", type=Path)
    parser.add_argument("--levels", help="Number of pyramid levels", type=int, default=5)
    parser.add_argument("--window_size", help="Window size", type=int, default=7)
    parser.add_argument("--alpha", help="Regularization parameter", type=float, default=1e-3)
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

def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    args = process_args()

    flow = calculate_forward_reverse_flow(
        args.image1,
        args.image2,
        args.levels,
        args.window_size,
        args.alpha,
        args.goodness_threshold,
    )

    try:
        gt_flow = read_flo_file(args.gtimage)
    except Exception as e:
        exit(1)


    from refine import refine_dense_flow

    flow_refined, I2_warp = refine_dense_flow(
        args.image1, args.image2, flow,
        # steps=400, lr=0.1, edge_beta=20.0, eps=1e-3, lambda_smooth=0.1, device="cpu"
        steps=10000, lr=0.9, edge_beta=20.0, eps=1e-3, lambda_smooth=0.1, device="cuda"
    )

    disp = visualize_flow_hsv(flow_refined)
    cv.imshow("Dense refined flow", disp)
    cv.imshow("I2 warped", I2_warp)

    # from refine import refine_patch_affine

    # thetas, flow_patch, I2_warp_patch, boxes = refine_patch_affine(
    #     args.image1,     # HxW float32 [0,1]
    #     args.image2,     # HxW float32 [0,1]
    #     flow,            # LK init HxWx2
    #     patch=64,        # try 16, 24, 32 depending on texture
    #     stride=16,       # stride < patch for overlap and smoother blend
    #     steps=1,
    #     lr=0.05,
    #     edge_beta=20.0,
    #     eps=1e-3,
    #     lambda_smooth=1e-3,  # 0 for no smoothing, >0 to reduce seams
    #     device="cpu",
    # )

    # # visualize with your helper
    # disp = visualize_flow_hsv(flow_patch)
    # cv.imshow("Patch-affine refined flow", disp)
    # cv.imshow("I2 warped (patch-affine)", I2_warp_patch)

    # Save the optical flow as an HSV image
    display = visualize_flow_hsv(flow)
    gt_display = visualize_gt_flow_hsv(gt_flow)
    cv.imshow("Optical flow (custom)", display)
    cv.imshow("Optical flow (ground truth)", gt_display)
    cv.imshow("Image1", args.image1)
    cv.imshow("Image2", args.image2)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()