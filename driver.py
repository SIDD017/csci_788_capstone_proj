from utils import *
from flow_final import *
from ground_truth_flo import visualize_gt_flow_hsv, read_flo_file


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


def main():
    # Process args
    args = process_args()

    # Create dictionary of initial flow parameters
    init_params = {
        "levels": args.levels,
        "window_size": args.window_size,
        "alpha": args.alpha,
        "goodness_threshold": args.goodness_threshold,
    }

    # Read Ground truth flow
    try:
        gt_flow = read_flo_file(args.gtimage)
    except Exception as e:
        exit(1)

    # Calculate initial flow using specified methods in args
    if args.use_affine:
        # Affine flow
        flow = AffineFlow(args.image1, args.image2, init_params, use_opencv=args.opencv_init)
    else:
        # Custom Lucas Kanade (from assignment)
        flow = CustomLucasKanadeFlow(args.image1, args.image2, init_params, use_opencv=args.opencv_init)

    # Use specified refinement method in args (2 params or affine 8 params)
    refine_params = {
        "steps": 1000 if args.use_affine else 10000,
        "lr": 0.9,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1,
        "device": "cuda"
    }

    flow_refined, I2_warp = flow.refine_flow(args.image1, args.image2, gt_flow, refine_params)

    # If affine refinement, optionally visualize all 8 params
    # - translation using the hsv trick
    # - origins using the hsv trick (or maybe quiver/vector plot to show how 
    # origins converge to the axis of rotation/scaling/shearing of the object)
    # - remaining 4 params as either grayscale planes or by warping a small reference image at each pixel location8

    # Display results using visualize_flow_hsv

    # Save files if specified in args


if __name__ == "__main__":
    main()
