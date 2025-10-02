import argparse
from pathlib import Path
from utils import uint8_to_float32, read_flo_file
from flow_interface import *
from refine_utils import refine_flow


def process_args():
    parser = argparse.ArgumentParser(description="Optical flow")
    parser.add_argument("image1", help="First image", type=Path)
    parser.add_argument("image2", help="Second image", type=Path)
    parser.add_argument("gtimage", help="Ground truth flow file", type=Path)
    parser.add_argument("--use_affine", help="Expand to affine parameters during refinement with gradient descent", type=bool, default=False)
    parser.add_argument("--opencv_init", help="Initialize optical flow using OpenCV Farneback dense flow", type=bool, default=False)
    parser.add_argument("--visualize_flow_params", help="Visualize flow parameters along with results (for debugging purposes)", type=bool, default=False)
    parser.add_argument("--log_results", help="Log experiment details to MLflow", type=bool, default=False)
    args = parser.parse_args()

    #TODO: Assert checks = shape, dims, types, etc
    return args


def preprocess_input_images(image1_path, image2_path, opencv_init=False):
    # Read the images and preprocess them according to whether we are using OpenCV initialization or custom
    # coarse-to-fine Lucas-Kanade initialization
    if opencv_init:
        #TODO: Any additional preprocessing for OpenCV method?
        raise NotImplementedError("OpenCV initialization preprocessing not implemented yet")    
    else:
        image1 = uint8_to_float32(cv.imread(str(image1_path), cv.IMREAD_GRAYSCALE))
        image2 = uint8_to_float32(cv.imread(str(image2_path), cv.IMREAD_GRAYSCALE))
    return image1, image2   


def main():
    args = process_args()

    image1, image2 = preprocess_input_images(args.image1, args.image2, opencv_init=args.opencv_init)
    try:
        gt_flow = read_flo_file(args.gtimage)
    except Exception as e:
        print(f"Error reading ground truth flow file: {e}")
        exit(1)
    
    # TODO: Make these command line args or read as JSON config file
    init_params = {
        "levels": 5,
        "window_size": 7,
        "alpha": 1e-3,
        "goodness_threshold": 2.0,
        "device": "cuda"
    }
    # TODO: Make these command line args or read a JSON config file
    refine_params = {
        "steps": 10000,
        "lr": 0.9,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1,
    }

    # Create specified flow object (affine or 2 param lucas kanade)
    if args.use_affine:
        # Affine flow
        # flow = AffineFlowWithLocalOrigins(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)
        flow = AffineFlow(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)
    else:
        # Custom Lucas Kanade (from assignment)
        flow = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)

    # Start MLflow run if logging enabled; set tracking URI to local server
    import mlflow
    import tempfile
    if args.log_results:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Optical Flow Refinement Experiment")
        with mlflow.start_run():
            # Log command-line flags and parameters
            mlflow.log_param("use_affine", args.use_affine)
            mlflow.log_param("opencv_init", args.opencv_init)
            mlflow.log_param("visualize_flow_params", args.visualize_flow_params)
            mlflow.log_params(init_params)
            mlflow.log_params(refine_params)
        
            flow_refined, I2_warp = refine_flow(flow, refine_params)
            I2_warp_uint8 = (np.clip(I2_warp, 0, 1) * 255).astype(np.uint8)
            # Save I2_warp and log as artifact
            temp_warp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            cv.imwrite(temp_warp, I2_warp_uint8)
            mlflow.log_artifact(temp_warp, artifact_path="outputs")
            
            # Visualize and save regular flow visualization
            # flow.visualize_flow()
            # if args.visualize_flow_params:
            #     flow.visualize_params()
            
            # (Optional) If available, use flow.log_results() to log extra metrics/artifacts.
            # For example:
            # loss = 0.0  # Replace with actual loss value
            # epe = flow.epe_error()
            # ang_err = flow.angular_error() if hasattr(flow, "angular_error") else 0.0
            # flow.log_results(loss, epe, ang_err, I2_warp)
    else:
        # Refine the inital flow using gradient descent
        flow_refined, I2_warp = refine_flow(flow, refine_params)

        cv.imshow("I2 warped", I2_warp)
        flow.visualize_flow()

        if args.visualize_flow_params:
            flow.visualize_params()


if __name__ == "__main__":
    main()
