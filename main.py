from flow_init import initialize_flow
from flow import BaseFlow, Flow2p, Flow6p, Flow8p
from pathlib import Path
import mlflow

from plot import disp_images, log_mlflow_artifacts
from refine import refine, grid_search
from utils import read_input_images, _flatten_dict

def check_exisitng_runs(exp_name: str, run_name: str) -> bool:
    """Check if an mlflow experiment with given name and run name exists."""
    if mlflow.get_experiment_by_name(exp_name) is not None:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(exp_name)
        runs = client.search_runs(experiment.experiment_id, 
                                  filter_string=f"attributes.run_name = \
                                    '{run_name}'")
        return runs is not None and len(runs) > 0
    return False

def run_pipeline(init_params: dict,
                 refine_params: dict,
                 image1_path: Path,
                 image2_path: Path,
                 gtimage_path: Path):
    """Initialize and refine optical flow between two images."""
    input_images = read_input_images(image1_path, 
                                     image2_path, 
                                     gtimage_path, 
                                     init_params["device"])
    init_flow = initialize_flow(image1_path, image2_path, init_params)
    flow = Flow6p(input_images["image1"].shape[-2:], 
                  device=init_params["device"], 
                  init_flow=init_flow)
    grid_search(flow, input_images, refine_params)
    log_mlflow_artifacts(input_images, flow)
    disp_images(flow.uv, input_images)


def main():
    """Driver code"""
    import jsonargparse
    parser = jsonargparse.ArgumentParser(description="Optical flow refinement",
                                         default_config_files=["config.yaml"])
    parser.add_argument("--exp_name", type=str, default="default_experiment")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_function_arguments(run_pipeline)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:" + str(Path.cwd() / "mlruns"))
    exp_name = args.exp_name
    run_name = args.run_name
    if check_exisitng_runs(exp_name, run_name):
        print(f"Experiment '{exp_name}' with run name '{run_name}' \
              already exists. Exiting to avoid overwrite.")
        return
    mlflow.set_experiment(exp_name)

    args_instantiated = parser.instantiate_classes(args)
    pipeline_args = args_instantiated.as_dict()
    pipeline_args.pop('exp_name', None)
    pipeline_args.pop('run_name', None)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_flatten_dict(args.as_dict()))
        run_pipeline(**pipeline_args)

if __name__ == "__main__":
    main()