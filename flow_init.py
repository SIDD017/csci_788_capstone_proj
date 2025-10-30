from utils import *

# Helper functions
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

    #Shouldn't we check for det_m == 0?
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


def calculate_initial_flow(image1, image2, init_params):
    # TODO: Add other initialization methods from opencv optflow package
    if init_params.get("init_method", "farneback") == "farneback":
        forward_flow = cv.calcOpticalFlowFarneback(image1, 
                                                   image2, 
                                                   None, 
                                                   pyr_scale=0.5, 
                                                   levels=init_params["levels"], 
                                                   winsize=init_params["window_size"], 
                                                   iterations=3, 
                                                   poly_n=5, 
                                                   poly_sigma=1.2, 
                                                   flags=0)
        reverse_flow = cv.calcOpticalFlowFarneback(image2, 
                                                   image1, 
                                                   None, 
                                                   pyr_scale=0.5, 
                                                   levels=init_params["levels"], 
                                                   winsize=init_params["window_size"], 
                                                   iterations=3, 
                                                   poly_n=5, 
                                                   poly_sigma=1.2, 
                                                   flags=0)
    else:  
        image1 = uint8_to_float32(image1)
        image2 = uint8_to_float32(image2)
        forward_flow = coarse_to_fine_optical_flow(image1, 
                                                   image2, 
                                                   init_params["levels"], 
                                                   init_params["window_size"], 
                                                   init_params["alpha"])
        reverse_flow = coarse_to_fine_optical_flow(image2, 
                                                   image1, 
                                                   init_params["levels"], 
                                                   init_params["window_size"], 
                                                   init_params["alpha"])
    # Catch occlusion issues and other bad estimates by checking for places where forward and
    # reverse flow disagre  
    is_bad_flow_estimate = np.linalg.norm(forward_flow + reverse_flow, axis=2) > init_params["goodness_threshold"]

    flow = (forward_flow - reverse_flow) / 2
    flow[is_bad_flow_estimate] = np.nan
    return flow