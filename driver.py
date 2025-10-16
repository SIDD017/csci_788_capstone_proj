import argparse
from pathlib import Path
from utils import uint8_to_float32, read_flo_file
from flow_interface import *
from refine_utils import refine_flow, convert_torch_to_cv
import mlflow


def process_args():
    parser = argparse.ArgumentParser(description="Optical flow")
    parser.add_argument("image1", help="First image", type=Path)
    parser.add_argument("image2", help="Second image", type=Path)
    parser.add_argument("gtimage", help="Ground truth flow file", type=Path)
    parser.add_argument("--use_affine", help="Expand to affine parameters during refinement with gradient descent", type=bool, default=False)
    parser.add_argument("--opencv_init", help="Initialize optical flow using OpenCV Farneback dense flow", type=bool, default=False)
    parser.add_argument("--visualize_flow_params", help="Visualize flow parameters along with results (for debugging purposes)", type=bool, default=False)
    parser.add_argument("--log_results", help="Log experiment details to MLflow", type=bool, default=False)
    parser.add_argument("--exp_name", help="Experiment name for MLflow logging", type=str, default="Optical flow refinement")
    parser.add_argument("--local_mlflow_store", help="Local directory for MLflow logging", type=str, default="mlruns")
    args = parser.parse_args()

    #TODO: Assert checks = shape, dims, types, etc
    return args


def read_input_images(image1_path, image2_path):
    ''' Read input images as grayscale uint8 '''
    image1 = cv.imread(str(image1_path), cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(str(image2_path), cv.IMREAD_GRAYSCALE)
    return image1, image2

def read_input_ground_truth(gtimage_path):
    ''' Read input ground truth flow from .flo file '''
    try:
        gt_flow = read_flo_file(gtimage_path)
    except Exception as e:
        print(f"Error reading ground truth flow file: {e}")
        exit(1)
    return gt_flow


def log_mlflow_params(args, init_params, refine_params):
    ''' Log parameters to MLflow '''
    mlflow.log_param("image1", str(args.image1))
    mlflow.log_param("image2", str(args.image2))
    mlflow.log_param("use_affine", args.use_affine)
    mlflow.log_param("opencv_init", args.opencv_init)
    mlflow.log_param("visualize_flow_params", args.visualize_flow_params)
    mlflow.log_params(init_params)
    mlflow.log_params(refine_params)


def log_mlflow_images(flow, I2_warp, flow_refined):
    ''' Log images to MLflow '''
    # Convert I2_warp to uint8 (assuming it's in [0, 1] range)
    I2_warp_uint8 = (np.clip(I2_warp, 0, 1) * 255).astype(np.uint8)
    result_img = {"I2_warp": I2_warp_uint8, 
                  "image1": convert_torch_to_cv(flow.image1), 
                  "image2": convert_torch_to_cv(flow.image2),
                  "flow": visualize_flow_hsv(flow_refined), 
                  "init_flow": visualize_flow_hsv(flow.init_flow),
                  "gt_flow": visualize_gt_flow_hsv(flow.gt_flow.cpu().numpy())}
    for name, img in result_img.items():
        plot_title = f"{name}.png"
        cv.imwrite(plot_title, img)
        mlflow.log_artifact(plot_title, artifact_path="outputs")


def log_mlflow_loss_plots(flow, log_plots=False):
    ''' Log loss and error metric plots to MLflow '''
    plot_title = "loss_plots.png"
    fig = plt.figure(figsize=(18, 4))
    
    log_subplots = {"Total Loss": "loss_log",
                "Data Loss": "data_loss_log",
                "Smoothness Loss": "smoothness_loss_log",
                "EPE to GT": "epe_log",
                "Angular Err (rad)": "angular_log"}

    for i, (title, key) in enumerate(log_subplots.items(), 1):
        plt.subplot(1, 5, i)
        plt.plot(flow.log_metrics[key])
        plt.title(title)
    plt.tight_layout()
    if not log_plots:
        plt.savefig(plot_title)
        plt.show()
    elif log_plots:
        mlflow.log_artifact(plot_title, artifact_path="outputs")
    plt.close(fig)


def log_mlflow_tv_plots(flow, args):
    ''' Log TV metric plots to MLflow '''
    # Individual tv plots
    plot_title = "tv_plots.png"
    fig = plt.figure(figsize=(18, 4))
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
    if not args.log_results:
        plt.savefig(plot_title)
        plt.show()
    elif args.log_results:
        mlflow.log_artifact(plot_title, artifact_path="outputs")
    plt.close(fig)


def display_initial_flow(flow):
    ''' Display initial flow for debugging purposes '''
    disp = visualize_flow_hsv(flow.init_flow)
    cv.imshow("Initial optical flow", disp)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    args = process_args()
    image1, image2 = read_input_images(args.image1, args.image2)
    gt_flow = read_input_ground_truth(args.gtimage)
    
    # Parameters for initial flow calculation and refinement
    # TODO: Convert to CLI args or JSON config file
    init_params = {
        "levels": 5,
        "window_size": 7,
        "alpha": 1e-3,
        "goodness_threshold": 2.0,
        "device": "cuda"
    }
    # TODO: Convert to CLI args or JSON config file
    refine_params = {
        "steps": 10000,
        "lr": 1e-2,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1, 
    }

    # Create specified flow object (affine or 2 param lucas kanade)
    if args.use_affine:
        flow = AffineFlowWithLocalOrigins(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)
        # flow = AffineFlow(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)
    else:
        flow = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=args.opencv_init)

    # Start MLflow run if logging enabled; set tracking URI to local server
    if args.log_results:
        # get current working directory
        cwd = Path.cwd()
        mlflow_dir = cwd / args.local_mlflow_store
        mlflow.set_tracking_uri("file:" + str(mlflow_dir))
        mlflow.set_experiment(args.exp_name)
        with mlflow.start_run():
            log_mlflow_params(args, init_params, refine_params)
            flow_refined, I2_warp = refine_flow(flow, refine_params)
            # Log final plots and results as artifacts
            log_mlflow_images(flow, I2_warp, flow_refined)
            log_mlflow_loss_plots(flow, log_plots=args.log_results)
            log_mlflow_tv_plots(flow, args)
    else:
        display_initial_flow(flow)
        flow_refined, I2_warp = refine_flow(flow, refine_params)
        cv.imshow("I2 warped", I2_warp)
        flow.visualize_flow()
        if args.visualize_flow_params:
            flow.visualize_params()


if __name__ == "__main__":
    main()
