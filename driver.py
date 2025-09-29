import argparse
from pathlib import Path
from utils import uint8_to_float32, visualize_flow_hsv, read_flo_file, visualize_gt_flow_hsv
from flow_interface import *
from refine_utils import refine_flow


def process_args():
    parser = argparse.ArgumentParser(description="Optical flow")
    parser.add_argument("image1", help="First image", type=Path)
    parser.add_argument("image2", help="Second image", type=Path)
    parser.add_argument("gtimage", help="Ground truth flow file", type=Path)
    parser.add_argument("--use_affine", help="Expand to affine parameters during refinement with gradient descent", type=bool, default=False)
    parser.add_argument("--opencv_init", help="Initialize optical flow using OpenCV Farneback dense flow", type=bool, default=False)
    parser.add_argument("--visualize_flow_params", help="Visualize flow parameters", type=bool, default=False)
    parser.add_argument("--log_results", help="Log and add results to TensorBoard/MLFlow repository", type=bool, default=False)
    args = parser.parse_args()

    #TODO: Assert checks = shape, dims, types, etc
    return args


def preprocess_input_images(image1_path, image2_path, opencv_init=False):
    # Read the images and preprocess them according to whether we are using OpenCV initialization or custom
    # coarse-to-fine Lucas-Kanade initialization1
    if opencv_init:
        #TODO: Any additional preprocessing for OpenCV method?
        raise NotImplementedError("OpenCV initialization preprocessing not implemented yet")    
    else:
        image1 = uint8_to_float32(cv.imread(str(image1_path), cv.IMREAD_GRAYSCALE))
        image2 = uint8_to_float32(cv.imread(str(image2_path), cv.IMREAD_GRAYSCALE))
    return image1, image2   


def main():
    # Process args
    args = process_args()

    # Read and preprocess images
    image1, image2 = preprocess_input_images(args.image1, args.image2, opencv_init=args.opencv_init)

    # Read Ground truth flow
    try:
        gt_flow = read_flo_file(args.gtimage)
    except Exception as e:
        print(f"Error reading ground truth flow file: {e}")
        exit(1)

    # Create dictionary of parameters for initializing flow
    # TODO: Make these command line args or read as JSON config file
    init_params = {
        "levels": 5,
        "window_size": 7,
        "alpha": 1e-3,
        "goodness_threshold": 2.0,
        "device": "cuda"
    }

    # Create specified flow object (affine or 2 param lucas kanade)
    # if args.use_affine:
    #     # Affine flow
    #     flow = AffineFlow(args.image1, args.image2, init_params, use_opencv=args.opencv_init)
    # else:
    #     # Custom Lucas Kanade (from assignment)
    flow = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)

    # Use specified refinement method in args (2 params or affine 8 params)
    # TODO: Make these command line args or read a JSON config file
    refine_params = {
        "steps": 1000 if args.use_affine else 10000,
        "lr": 0.9,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1,
    }

    # Refine the inital flow using gradient descent
    flow_refined, I2_warp = refine_flow(flow, refine_params)

    disp = visualize_flow_hsv(flow_refined)
    cv.imshow("Dense refined flow", disp)
    cv.imshow("I2 warped", I2_warp)

    display = visualize_flow_hsv(flow.init_flow.cpu().numpy())
    gt_display = visualize_gt_flow_hsv(gt_flow)
    cv.imshow("Optical flow (custom)", display)
    cv.imshow("Optical flow (ground truth)", gt_display)
    cv.imshow("Image1", image1)
    cv.imshow("Image2", image2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # # Visualize the results and optionally the flow parameters as well if specified
    # if args.visualize_flow_params:
    #     flow.visualize_flow()

    # # Log results if specified
    # if args.log_results:
    #     flow.log_results()

if __name__ == "__main__":
    main()
