import torch
from flow import BaseFlow, Metrics
import torch.nn.functional as F
import mlflow

def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)

def barron_loss(err: torch.Tensor, alpha: torch.Tensor = 1.0, c: float = 1e-3, dim=None) -> torch.Tensor:
    """Barron (2019) loss function."""
    alpha = torch.tensor(alpha, dtype=err.dtype, device=err.device)
    cost = 0
    diff2 = (err / c) ** 2
    if torch.abs(alpha - 2.0) < 1e-7:
        cost = 0.5 * diff2
    elif torch.abs(alpha - 0.0) < 1e-7:
        cost = torch.log1p(0.5 * diff2)
    elif torch.isinf(alpha) and alpha < 0:
        cost = 1 - torch.exp(-0.5 * diff2)
    else:
        cost = (
            torch.abs(alpha - 2)
            / alpha
            * (torch.pow(1 + diff2 / torch.abs(alpha - 2), alpha / 2) - 1)
        )
    return cost


def init_log_metrics():
    return {"data_log": [],
            "ar0_log": [],
            "ar1_log": [],
            "loss_log": [],
            "epe_log": [],
            "angular_log": []}


def sobel_magnitude(t):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy)


def refine(flow: BaseFlow, input_images: dict[str, torch.Tensor], refine_params: dict):
    log_metrics = init_log_metrics()

    # AR0 weights (magnitude penalties)
    lambda_ar0_uv = refine_params.get("lambda_ar0_uv", 0.0001)
    lambda_ar0_affine = refine_params.get("lambda_ar0_affine", 0.0001)
    
    # AR1 weights (smoothness penalties)
    lambda_ar1_uv = refine_params.get("lambda_ar1_uv", 0.1)
    lambda_ar1_affine = refine_params.get("lambda_ar1_affine", 0.1)
    lambda_ar1_affine_uv = refine_params.get("lambda_ar1_affine_uv", 0.1)

    # Loss function parameters
    data_alpha = refine_params.get("data_alpha", 1.0)
    data_c = refine_params.get("data_c", 1e-3)
    ar0_alpha = refine_params.get("ar0_alpha", 1.0)
    ar0_c = refine_params.get("ar0_c", 1e-3)
    ar1_alpha = refine_params.get("ar1_alpha", 1.0)
    ar1_c = refine_params.get("ar1_c", 1e-3)

    opt = torch.optim.Adam([flow.params], lr=refine_params.get("lr", 1e-1))

    # Create loss functions with specific parameters
    data_loss_fn = lambda x: barron_loss(x, alpha=data_alpha, c=data_c)
    ar0_loss_fn = lambda x: barron_loss(x, alpha=ar0_alpha, c=ar0_c)
    ar1_loss_fn = lambda x: barron_loss(x, alpha=ar1_alpha, c=ar1_c)

    for t in range(refine_params.get("steps", 1000)):
        opt.zero_grad()
        I2w = flow.warp_image(input_images["image2"])
        resid = I2w - input_images["image1"]

        data = (data_loss_fn(resid)).mean()

        ar0 = flow.ar0_terms(ar0_loss_fn)
        ar0_weighted = lambda_ar0_uv * ar0["uv"] + lambda_ar0_affine * ar0["b"]
        ar0_weighted = ar0_weighted.mean()

        ar1 = flow.ar1_terms(3, ar1_loss_fn)
        ar1_weighted = lambda_ar1_uv * ar1["uv"] + lambda_ar1_affine * ar1["b"] + lambda_ar1_affine_uv * ar1["affine_uv"]
        ar1_weighted = ar1_weighted.mean()

        loss = data + ar1_weighted + ar0_weighted
        loss.backward()
        opt.step()

        if (t % 50 == 0):
            print("\nIteration : ", t)
            print(loss.item())
            with torch.no_grad():
                log_metrics["data_log"].append(data.item())
                log_metrics["ar0_log"].append(ar0_weighted.item())
                log_metrics["ar1_log"].append(ar1_weighted.item())
                log_metrics["loss_log"].append(loss.item())
                log_metrics["epe_log"].append(Metrics.epe(input_images["gtimage"], flow.uv))
                log_metrics["angular_log"].append(Metrics.angular_error(input_images["gtimage"], flow.uv))
                print("epe : ", log_metrics["epe_log"][-1])
                print("ang error : ", log_metrics["angular_log"][-1])
            mlflow.log_metrics({k: lst[-1] for k, lst in log_metrics.items()}, step=t)
    return log_metrics["epe_log"][-1]
    # # Create loss functions with specific parameters
    # data_loss_fn = lambda x: barron_loss(x, alpha=-100, c=data_c)
    # ar0_loss_fn = lambda x: barron_loss(x, alpha=-100, c=ar0_c)
    # ar1_loss_fn = lambda x: barron_loss(x, alpha=-100, c=ar1_c)

    # for t in range(refine_params.get("steps", 1000), refine_params.get("steps", 1000) * 2):
    #     opt.zero_grad()
    #     I2w = flow.warp_image(input_images["image2"])
    #     resid = I2w - input_images["image1"]

    #     data = (data_loss_fn(resid)).mean()

    #     ar0 = flow.ar0_terms(ar0_loss_fn)
    #     ar0_weighted = lambda_ar0_uv * ar0["uv"] + lambda_ar0_affine * ar0["b"]
    #     ar0_weighted = ar0_weighted.mean()

    #     ar1 = flow.ar1_terms(3, ar1_loss_fn)
    #     ar1_weighted = lambda_ar1_uv * ar1["uv"] + lambda_ar1_affine * ar1["b"] + lambda_ar1_affine_uv * ar1["affine_uv"]
    #     ar1_weighted = ar1_weighted.mean()

    #     loss = data + ar1_weighted + ar0_weighted
    #     loss.backward()
    #     opt.step()

    #     if (t % 50 == 0):
    #         print("\nIteration : ", t)
    #         print(loss.item())
    #         with torch.no_grad():
    #             log_metrics["data_log"].append(data.item())
    #             log_metrics["ar0_log"].append(ar0_weighted.item())
    #             log_metrics["ar1_log"].append(ar1_weighted.item())
    #             log_metrics["loss_log"].append(loss.item())
    #             log_metrics["epe_log"].append(Metrics.epe(input_images["gtimage"], flow.uv))
    #             log_metrics["angular_log"].append(Metrics.angular_error(input_images["gtimage"], flow.uv))
    #             print("epe : ", log_metrics["epe_log"][-1])
    #             print("ang error : ", log_metrics["angular_log"][-1])
    #         mlflow.log_metrics({k: lst[-1] for k, lst in log_metrics.items()}, step=t)



def grid_search(flow: BaseFlow, input_images: dict[str, torch.Tensor], refine_params: dict):
    # Grid search for c values
    data_c_values = [ 1e-3, 5e-3, 1e-2]
    ar0_c_values = [1e-4, 5e-4, 1e-3]
    ar1_c_values = [1e-4, 5e-4, 1e-3]

    if data_c_values is not None or ar0_c_values is not None or ar1_c_values is not None:
        best_epe = float('inf')
        best_params = {}

        total_combinations = len(data_c_values) * len(ar0_c_values) * len(ar1_c_values)
        current_combo = 0

        for data_c in data_c_values:
            for ar0_c in ar0_c_values:
                for ar1_c in ar1_c_values:
                    current_combo += 1
                    print(f"\n{'='*60}")
                    print(f"Grid Search {current_combo}/{total_combinations}")
                    print(f"Testing: data_c={data_c}, ar0_c={ar0_c}, ar1_c={ar1_c}")
                    print(f"{'='*60}")
                    
                    # Create a copy of refine_params with current c values
                    current_params = refine_params.copy()
                    current_params["data_c"] = data_c
                    current_params["ar0_c"] = ar0_c
                    current_params["ar1_c"] = ar1_c
                    
                    # Reinitialize flow for this run
                    from copy import deepcopy
                    flow_copy = deepcopy(flow)
                    
                    # Run refinement with current parameters
                    final_epe = refine(flow_copy, input_images, current_params)
                    
                    # Log grid search results
                    # mlflow.log_metrics({
                    #     f"grid_search/epe": final_epe,
                    #     f"grid_search/data_c": data_c,
                    #     f"grid_search/ar0_c": ar0_c,
                    #     f"grid_search/ar1_c": ar1_c,
                    # })
                    
                    # Track best parameters
                    if final_epe < best_epe:
                        best_epe = final_epe
                        best_params = {
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
        
        # mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        # mlflow.log_metric("best_grid_search_epe", best_epe)
        
        # Use best parameters for final run
        refine_params.update(best_params)
        print(refine_params)
    
    # # Run final refinement (or single run if no grid search)
    # return _refine_single_run(flow, input_images, refine_params)