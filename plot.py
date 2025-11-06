import ast
import cv2 as cv
import numpy as np
import torch

from utils import convert_torch_to_cv, visualize_flow_hsv, visualize_gt_flow_hsv

def disp_images(flow_uv, input_images):
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


if __name__ == "__main__":
    """given an exp name,get latest run and load in the logged images and display them"""
    import mlflow
    from pathlib import Path
    exp_name = "2p_10k_new_test"
    mlflow.set_tracking_uri("file:" + str(Path.cwd() / "mlruns"))
    mlflow.set_experiment(exp_name)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(exp_name)
    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    run = runs[0]
    artifacts = client.list_artifacts(run.info.run_id, path="outputs")
    logged_images = {}
    refined_flow_params = {}
    for artifact in artifacts:
        if artifact.path.endswith("log_images.json"):
            local_path = client.download_artifacts(run.info.run_id, artifact.path)
            import json
            with open(local_path, 'r') as f:
                logged_images = json.load(f)
        elif artifact.path.endswith("refined_flow_params.json"):
            local_path = client.download_artifacts(run.info.run_id, artifact.path)
            import json
            with open(local_path, 'r') as f:
                refined_flow_params = json.load(f)
    # reconstruct tensors from strings
    for k, v in logged_images.items():
        if isinstance(v, str):
            v = ast.literal_eval(v)
        logged_images[k] = torch.tensor(np.array(v)).unsqueeze(0)
    for k, v in refined_flow_params.items():
        if isinstance(v, str):
            v = ast.literal_eval(v)
        refined_flow_params[k] = torch.tensor(np.array(v)).unsqueeze(0)
    disp_images(refined_flow_params["params"], logged_images)
