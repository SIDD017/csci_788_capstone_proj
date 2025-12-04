import torch
from flow_init import initialize_flow
from flow import BaseFlow, Flow2p, Flow6p, Flow8p
from pathlib import Path
import mlflow

from plot import disp_images, log_mlflow_artifacts
from refine import refine
from utils import read_input_images, _flatten_dict


def grid_search(flow: BaseFlow, input_images: dict[str, torch.Tensor], refine_params: dict):
    
    from pathlib import Path
    mlflow.set_tracking_uri("file:" + str(Path.cwd() / "mlruns"))
    mlflow.set_experiment("final_alpha_c_grid_search")   

    steps = 1000
    lr = 1e-1

    # Grid search for c values
    # lambda_ar0_uv_values = [1e-6, 1e-7, 1e-8, 1e-9]          # best so far 1e-6
    # lambda_ar0_affine_values = [1e-6, 1e-7, 1e-8, 1e-9]      # best so far 1e-6
    # lambda_ar1_uv_values = [0.4, 0.1, 0.05, 0.025, 0.01, 0.005]                       # best so far 0.1
    # lambda_ar1_affine_values = [8, 4, 1, 0.5, 0.1]                   # best so far 1
    # lambda_ar1_affine_uv_values = [0.4, 0.1, 0.05, 0.01, 0.005]                # best so far 0.1

    lambda_ar0_uv_values = [1e-9]          # best so far 1e-6
    lambda_ar0_affine_values = [1e-6]      # best so far 1e-6
    lambda_ar1_uv_values = [0.05]                       # best so far 0.1
    lambda_ar1_affine_values = [1]                   # best so far 1
    lambda_ar1_affine_uv_values = [0.05]                # best so far 0.1

    data_alpha_values = [-0.5, 0, 0.5, 1]                                     # best so far 1
    data_c_values = [0.01, 0.05, 0.075]                                      # best so far 1e-3 

    ar0_alpha_values = [-0.5, 0, 0.5, 1]                                      # best so far 1
    ar0_c_values = [0.5, 1, 1.5]                                       # best so far 1e-3

    ar1_alpha_values = [-1, 0, 1]                                      # best so far 1
    ar1_c_values = [0.1, 0.01]                                       # best so far 1e-3

    # if data_c_values is not None or ar0_c_values is not None or ar1_c_values is not None:
    best_epe = float('inf')
    best_params = {}

    total_combinations = len(data_alpha_values) * len(ar0_alpha_values) * len(ar1_alpha_values) * len(data_c_values) * len(ar0_c_values) * len(ar1_c_values) * len(lambda_ar0_uv_values) * len(lambda_ar0_affine_values) * len(lambda_ar1_uv_values) * len(lambda_ar1_affine_values) * len(lambda_ar1_affine_uv_values)
    current_combo = 0

    for lambda_ar0_uv in lambda_ar0_uv_values:
        for lambda_ar0_affine in lambda_ar0_affine_values:
            for lambda_ar1_uv in lambda_ar1_uv_values:
                for lambda_ar1_affine in lambda_ar1_affine_values:
                    for lambda_ar1_affine_uv in lambda_ar1_affine_uv_values:
                        for data_alpha in data_alpha_values:
                            for ar0_alpha in ar0_alpha_values:
                                for ar1_alpha in ar1_alpha_values:
                                    for data_c in data_c_values:
                                        for ar0_c in ar0_c_values:
                                            for ar1_c in ar1_c_values:
                                                with mlflow.start_run(run_name=str(current_combo + 1)):
                                                    current_combo += 1
                                                    print(f"\n{'='*60}")
                                                    print(f"Grid Search {current_combo}/{total_combinations}")
                                                    print(f"Testing: data_c={data_c}, ar0_c={ar0_c}, ar1_c={ar1_c}")
                                                    print(f"{'='*60}")
                                                    
                                                    # Create a copy of refine_params with current c values
                                                    current_params = refine_params.copy()
                                                    current_params["steps"] = steps
                                                    current_params["lr"] = lr
                                                    current_params["lambda_ar0_uv"] = lambda_ar0_uv
                                                    current_params["lambda_ar0_affine"] = lambda_ar0_affine
                                                    current_params["lambda_ar1_uv"] = lambda_ar1_uv
                                                    current_params["lambda_ar1_affine"] = lambda_ar1_affine
                                                    current_params["lambda_ar1_affine_uv"] = lambda_ar1_affine_uv
                                                    current_params["data_alpha"] = data_alpha
                                                    current_params["ar0_alpha"] = ar0_alpha
                                                    current_params["ar1_alpha"] = ar1_alpha
                                                    current_params["data_c"] = data_c
                                                    current_params["ar0_c"] = ar0_c
                                                    current_params["ar1_c"] = ar1_c

                                                    mlflow.log_params(current_params)

                                                    
                                                    # Reinitialize flow for this run
                                                    from copy import deepcopy
                                                    flow_copy = deepcopy(flow)
                                                    
                                                    # Run refinement with current parameters
                                                    final_epe = refine(flow_copy, input_images, current_params)
                                                    
                                                    # Log grid search results
                                                    mlflow.log_metrics({
                                                        f"grid_search/epe": final_epe,
                                                        # f"grid_search/lambda_ar0_uv": lambda_ar0_uv,
                                                        # f"grid_search/lambda_ar0_affine": lambda_ar0_affine,
                                                        # f"grid_search/lambda_ar1_uv": lambda_ar1_uv,
                                                        # f"grid_search/lambda_ar1_affine": lambda_ar1_affine,
                                                        # f"grid_search/lambda_ar1_affine_uv": lambda_ar1_affine_uv,
                                                        # f"grid_search/data_alpha": data_alpha,
                                                        # f"grid_search/ar0_alpha": ar0_alpha,
                                                        # f"grid_search/ar1_alpha": ar1_alpha,
                                                        # f"grid_search/data_c": data_c,
                                                        # f"grid_search/ar0_c": ar0_c,
                                                        # f"grid_search/ar1_c": ar1_c,
                                                    })
                                                    
                                                    # Track best parameters
                                                    if final_epe < best_epe:
                                                        best_epe = final_epe
                                                        best_params = {
                                                            "lambda_ar0_uv": lambda_ar0_uv,
                                                            "lambda_ar0_affine": lambda_ar0_affine,
                                                            "lambda_ar1_uv": lambda_ar1_uv,
                                                            "lambda_ar1_affine": lambda_ar1_affine,
                                                            "lambda_ar1_affine_uv": lambda_ar1_affine_uv,
                                                            "data_alpha": data_alpha,
                                                            "ar0_alpha": ar0_alpha,
                                                            "ar1_alpha": ar1_alpha,
                                                            "data_c": data_c,
                                                            "ar0_c": ar0_c,
                                                            "ar1_c": ar1_c
                                                        }
                                                        print(f"\n*** NEW BEST EPE: {best_epe:.4f} ***")
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE")
    print(f"Best EPE: {best_epe:.4f}")
    print(f"Best params: {best_params}")
    print(f"{'='*60}")
    
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_grid_search_epe", best_epe)

    refine_params.update(best_params)
    print(refine_params)


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


def main():
    """Driver code"""
    import jsonargparse
    parser = jsonargparse.ArgumentParser(description="Optical flow refinement",
                                         default_config_files=["config.yaml"])
    parser.add_argument("--exp_name", type=str, default="default_experiment")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_function_arguments(run_pipeline)
    args = parser.parse_args()

    args_instantiated = parser.instantiate_classes(args)
    pipeline_args = args_instantiated.as_dict()
    pipeline_args.pop('exp_name', None)
    pipeline_args.pop('run_name', None)
    run_pipeline(**pipeline_args)

if __name__ == "__main__":
    main()




# Best so far
'''
For steps : 400
learning rate : 0.1

Best EPE: 1.11563

Best params: {
'lambda_ar0_uv': 1e-07, 
'lambda_ar0_affine': 1e-08, 
'lambda_ar1_uv': 0.05, 
'lambda_ar1_affine': 1, 
'lambda_ar1_affine_uv': 0.05, 
'data_alpha': 1, 
'ar0_alpha': 1, 
'ar1_alpha': 1, 
'data_c': 0.001, 
'ar0_c': 0.001, 
'ar1_c': 0.001}
'''


'''
For steps : 400
learning rate : 0.1

Best EPE: 1.1480

Best params: {
'lambda_ar0_uv': 1e-06, 
'lambda_ar0_affine': 1e-06, 
'lambda_ar1_uv': 0.1, 
'lambda_ar1_affine': 1, 
'lambda_ar1_affine_uv': 0.1, 
'data_alpha': 1, 
'ar0_alpha': 1, 
'ar1_alpha': 1, 
'data_c': 0.001, 
'ar0_c': 0.001, 
'ar1_c': 0.001}
'''