from pathlib import Path
import mlflow
from plot import log_mlflow_images
from refine import refine_flow
from utils import _flatten_dict
from flow import CustomLucasKanadeFlow, AffineFlow, AffineFlowWithLocalOrigins
import cv2 as cv


def build_flow_object(image1, image2, gtimage_path, init_params):
    if init_params.get("model") == "translation":
        flow_class = CustomLucasKanadeFlow
    elif init_params.get("model") == "affine":
        flow_class = AffineFlow
    elif init_params.get("model") == "affine_with_local_origins":
        flow_class = AffineFlowWithLocalOrigins
    else:
        raise ValueError(f"Unknown flow model: {init_params.get('model')}")
    return flow_class(image1, image2, gtimage_path, init_params)


def run_pipeline(init_params: dict, 
                 refine_params: dict,
                 image1_path: Path, 
                 image2_path: Path, 
                 gtimage_path: Path):
    """
    Instantiaites flow object and runs refinement pipeline with MLflow logging
    """
    flow = build_flow_object(image1_path, image2_path, gtimage_path, init_params)
    refine_flow(flow, refine_params)
    log_mlflow_images(flow)
    print(flow.params[..., 0:2], "\n")
    print(flow.params[..., 3:5], "\n")

    # TODO - store the refined/predicted flow values (not visualization) as a file (np.savez, .flo, whatever)
    # TODO - also store the final state of the model itself (uv, A, etc) as artifacts

    flow.visualize_params()


def main():
    import jsonargparse
    parser = jsonargparse.ArgumentParser(description="Dense Optical flow refinement and MLflow logging", 
                                         default_config_files=["config.yaml"])
    parser.add_function_arguments(run_pipeline)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:" + str(Path.cwd() / "mlruns"))
    exp_name = "6p_10k_test"
    mlflow.set_experiment(exp_name)

    # TODO: Check for previous runs (probably best to disable this during active development)
    args_instantiated = parser.instantiate_classes(args)
    with mlflow.start_run():
        mlflow.log_params(_flatten_dict(args.as_dict()))
        run_pipeline(**args_instantiated.as_dict())


if __name__ == "__main__":
    main()
