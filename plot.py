import cv2 as cv
import jsonargparse
import torch
from flow import BaseFlow
import mlflow

from utils import convert_torch_to_cv, visualize_flow_hsv, visualize_gt_flow_hsv

REFINED_FLOW_FILE_NAME = "refined_flow_params.pt"
LOG_IMAGES_FILE_NAME = "log_images.pt"

def disp_images(flow_uv, input_images):
    """Display input images and flow visualizations using OpenCV."""
    cv.imshow("Image1", convert_torch_to_cv(input_images["image1"]))
    cv.imshow("Image2", convert_torch_to_cv(input_images["image2"]))
    disp = flow_uv
    disp_np = disp.detach().cpu().numpy()
    disp_hsv = visualize_flow_hsv(disp_np)
    cv.imshow("Dense refined flow", disp_hsv)
    gt_disp = visualize_gt_flow_hsv(input_images["gtimage"].cpu().numpy())
    cv.imshow("Optical flow (ground truth)", gt_disp)
    cv.waitKey(0)
    cv.destroyAllWindows()


def log_mlflow_artifacts(input_images: dict, flow: BaseFlow):
    """Log input images and refined flow parameters as mlflow artifacts."""
    serialized_flow = flow.serialize()
    torch.save(serialized_flow, REFINED_FLOW_FILE_NAME)
    mlflow.log_artifact(REFINED_FLOW_FILE_NAME, "outputs")
    input_images_cpu = {k: v.to("cpu").detach() for k, v in input_images.items()}
    torch.save(input_images_cpu, LOG_IMAGES_FILE_NAME)
    mlflow.log_artifact(LOG_IMAGES_FILE_NAME, "outputs")
    import os
    os.remove(REFINED_FLOW_FILE_NAME)
    os.remove(LOG_IMAGES_FILE_NAME)


def get_mlfow_run_artifacts(exp_name: str, run_name: str):
    """Get mlflow run object given the exp name and run name."""
    from pathlib import Path
    mlflow.set_tracking_uri("file:" + str(Path.cwd() / "mlruns"))
    mlflow.set_experiment(exp_name)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(exp_name)
    runs = client.search_runs(experiment.experiment_id, 
                              filter_string=f"attributes.run_name = '{run_name}'")
    if not runs:
        raise ValueError(f"Not found experiment '{exp_name}' with \
                         run name '{run_name}'")
    artifacts = client.list_artifacts(runs[0].info.run_id, 
                                      path="outputs")
    logged_images = {}
    refined_flow_params = {}
    for artifact in artifacts:
        if artifact.path.endswith(LOG_IMAGES_FILE_NAME):
            local_path = client.download_artifacts(runs[0].info.run_id, 
                                                   artifact.path)
            logged_images = torch.load(local_path)
        elif artifact.path.endswith(REFINED_FLOW_FILE_NAME):
            local_path = client.download_artifacts(runs[0].info.run_id, 
                                                   artifact.path)
            refined_flow_params = torch.load(local_path)
    return logged_images, refined_flow_params


if __name__ == "__main__":
    """Given an exp name,get latest run and load in the logged images and display them"""
    parser = jsonargparse.ArgumentParser(description="Display logged mlflow images")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    exp_name = args.exp_name
    run_name = args.run_name
    logged_images, refined_flow_params = get_mlfow_run_artifacts(exp_name, 
                                                                 run_name)
    disp_images(refined_flow_params["uv"], logged_images)
