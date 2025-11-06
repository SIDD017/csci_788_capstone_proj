import numpy as np
import torch
from flow_init import calculate_initial_flow, initialize_flow
from flow_new import BaseFlow, Flow2p, Flow6p
from pathlib import Path
import mlflow
import cv2 as cv

from plot_new import disp_images
from refine_new import refine
from utils import convert_torch_to_cv, np_im_to_torch, read_flo_file, uint8_to_float32, visualize_flow_hsv, visualize_gt_flow_hsv

def _flatten_dict(d: dict, key_sep="_") -> dict:
    """Flattens a nested dictionary."""
    out = {}

    def flatten(x: dict, name: str = ""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + key_sep)
        else:
            if name[:-1] in out:
                raise ValueError(
                    f"Duplicate key created during flattening: {name[:-1]}"
                )
            out[name[:-1]] = x

    flatten(d)
    return out

def read_input_images(image1_path: Path, 
                      image2_path: Path, 
                      gtimage_path: Path,
                      device: torch.device) -> dict[str, torch.Tensor]:
    image1 = cv.imread(str(image1_path), cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(str(image2_path), cv.IMREAD_GRAYSCALE)
    gtimage = read_flo_file(gtimage_path)
    return {"image1": np_im_to_torch(uint8_to_float32(image1)).to(device), 
            "image2": np_im_to_torch(uint8_to_float32(image2)).to(device), 
            "gtimage": torch.from_numpy(gtimage.astype(np.float32)).to(device)}


def run_pipeline(init_params: dict,
                 refine_params: dict,
                 image1_path: Path,
                 image2_path: Path,
                 gtimage_path: Path):
    # read in images as torch tensors on correct device
    input_images = read_input_images(image1_path, 
                                     image2_path, 
                                     gtimage_path, 
                                     init_params["device"])
    # initial flow from lk or farneback
    init_flow = initialize_flow(image1_path, image2_path, init_params)
    # build flow object with inital flow
    flow = Flow6p(input_images["image1"].shape[-2:], 
                  device=init_params["device"], 
                  init_flow=init_flow)
    # refinement loop
    refine(flow, input_images, refine_params)
    # save mlflow artifacts: serialized params and final image results as np arrays
    serialized_flow = {k: v.to("cpu").detach().numpy() for k, v in flow.serialize().items()}
    mlflow.log_dict(serialized_flow, "outputs/refined_flow_params.json")
    log_input_images = {k: v.to("cpu").detach().numpy() for k, v in input_images.items()}
    mlflow.log_dict(log_input_images, "outputs/log_images.json")
    # display final results
    disp_images(flow.uv, input_images)
    


def main():
    """Driver code"""
    import jsonargparse
    parser = jsonargparse.ArgumentParser(description="Dense optical flow refinement",
                                         default_config_files=["config.yaml"])
    parser.add_function_arguments(run_pipeline)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:" + str(Path.cwd() / "mlruns"))
    exp_name = "2p_10k_new_test"
    mlflow.set_experiment(exp_name)

    args_instantiated = parser.instantiate_classes(args)
    with mlflow.start_run():
        mlflow.log_params(_flatten_dict(args.as_dict()))
        run_pipeline(**args_instantiated.as_dict())

if __name__ == "__main__":
    main()