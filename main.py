from flow_init import initialize_flow
from flow import BaseFlow, Flow2p, Flow6p
from pathlib import Path
import mlflow

from plot import disp_images, log_mlflow_artifacts
from refine import refine
from utils import read_input_images, _flatten_dict


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
    log_mlflow_artifacts(input_images, flow)
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