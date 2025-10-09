import argparse
from pathlib import Path
from utils import uint8_to_float32, read_flo_file
from flow_interface import *
from refine_utils import refine_flow, convert_torch_to_cv


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
        "steps": 50000,
        "lr": 1e-2,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1,
    }

    # Create specified flow object (affine or 2 param lucas kanade)
    if args.use_affine:
        # Affine flow
        flow = AffineFlowWithLocalOrigins(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)
        # flow = AffineFlow(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)
    else:
        # Custom Lucas Kanade (from assignment)
        flow = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)

    # Start MLflow run if logging enabled; set tracking URI to local server
    import mlflow
    import tempfile
    if args.log_results:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Optical Flow Refinement Experiment Updated (Separated TV terms)")
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("use_affine", args.use_affine)
            mlflow.log_param("opencv_init", args.opencv_init)
            mlflow.log_param("visualize_flow_params", args.visualize_flow_params)
            mlflow.log_params(init_params)
            mlflow.log_params(refine_params)
        
            # Refine the flow and compute the warped image
            flow_refined, I2_warp = refine_flow(flow, refine_params)

            # Convert I2_warp to uint8 (assuming it's in [0, 1] range)
            I2_warp_uint8 = (np.clip(I2_warp, 0, 1) * 255).astype(np.uint8)

            # Save the warped image and log as an artifact
            temp_warp = "I2_warp.png"
            cv.imwrite(temp_warp, I2_warp_uint8)
            mlflow.log_artifact(temp_warp, artifact_path="outputs")

            # Save image1 converted from torch format and log as an artifact
            temp_warp = "image1.png"
            cv.imwrite(temp_warp, convert_torch_to_cv(flow.image1))
            mlflow.log_artifact(temp_warp, artifact_path="outputs")

            # Save image2 converted from torch format and log as an artifact
            temp_warp = "image2.png"
            cv.imwrite(temp_warp, convert_torch_to_cv(flow.image2))
            mlflow.log_artifact(temp_warp, artifact_path="outputs")

            # Save the flow parameter visualization (using HSV) as an artifact
            temp_warp = "flow_params.png"
            disp = visualize_flow_hsv(flow.params.detach().cpu().numpy())
            cv.imwrite(temp_warp, disp)
            mlflow.log_artifact(temp_warp, artifact_path="outputs")

            # Visualize the initial flow and ground truth flow
            temp_warp = "init_flow.png"
            display = visualize_flow_hsv(flow.init_flow)
            gt_display = visualize_gt_flow_hsv(flow.gt_flow.cpu().numpy())
            cv.imwrite(temp_warp, display)
            mlflow.log_artifact(temp_warp, artifact_path="outputs")
            temp_warp = "gt_flow.png"
            cv.imwrite(temp_warp, gt_display)
            mlflow.log_artifact(temp_warp, artifact_path="outputs")

            # Save plots to mlflow as well
            temp_warp = "loss_plots.png"
            plt.figure(figsize=(18, 4))
    
            plt.subplot(1, 5, 1)
            plt.plot(flow.log_metrics["loss_log"])
            plt.title("Total Loss")
            
            plt.subplot(1, 5, 2)
            plt.plot(flow.log_metrics["data_loss_log"])
            plt.title("Data Loss")
            
            plt.subplot(1, 5, 3)
            plt.plot(flow.log_metrics["smoothness_loss_log"])
            plt.title("Smoothness Loss")
            
            plt.subplot(1, 5, 4)
            plt.plot(flow.log_metrics["epe_log"])
            plt.title("EPE to GT")
            
            plt.subplot(1, 5, 5)
            plt.plot(flow.log_metrics["angular_log"])
            plt.title("Angular Err (rad)")

            plt.tight_layout()
            plt.savefig(temp_warp)
            plt.show()
            mlflow.log_artifact(temp_warp, artifact_path="outputs")
            plt.close()

            # Individual tv plots
            plt.figure(figsize=(18, 4))
            plt.subplot(1, 3, 1)
            plt.plot(flow.log_metrics["uv_tv_log"])
            plt.title("UV TV")
            if args.use_affine:
                plt.subplot(1, 3, 2)
                plt.plot(flow.log_metrics["A_tv_log"])
                plt.title("A TV")
                plt.subplot(1, 3, 3)
                plt.plot(flow.log_metrics["origin_tv_log"])
                plt.title("Origin TV")
            plt.tight_layout()
            temp_warp = "tv_plots.png"
            plt.savefig(temp_warp)
            plt.show()
            mlflow.log_artifact(temp_warp, artifact_path="outputs")
            plt.close()

            # Visualize and save regular flow visualization
            flow.visualize_flow()
            if args.visualize_flow_params:
                flow.visualize_params()

    else:
        # Refine the inital flow using gradient descent
        flow_refined, I2_warp = refine_flow(flow, refine_params)

        cv.imshow("I2 warped", I2_warp)
        flow.visualize_flow()

        if args.visualize_flow_params:
            flow.visualize_params()


if __name__ == "__main__":
    main()
