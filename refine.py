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

    # Edge downweighting
    edge_beta_data = refine_params.get("edge_beta_data", 20.0)
    edge_beta_ar1_uv = refine_params.get("edge_beta_ar1_uv", 20.0)
    edge_beta_ar1_affine = refine_params.get("edge_beta_ar1_affine", 20.0)
    edge_beta_ar1_affine_uv = refine_params.get("edge_beta_ar1_affine_uv", 20.0)
    # Compute edge weights
    w_edge_data = 1.0 / (1.0 + edge_beta_data * sobel_magnitude(input_images["image1"]))
    w_edge_ar1_uv = 1.0 / (1.0 + edge_beta_ar1_uv * sobel_magnitude(input_images["image1"]))
    w_edge_ar1_affine = 1.0 / (1.0 + edge_beta_ar1_affine * sobel_magnitude(input_images["image1"]))
    w_edge_ar1_affine_uv = 1.0 / (1.0 + edge_beta_ar1_affine_uv * sobel_magnitude(input_images["image1"]))
    w_edge_data = w_edge_data.detach()
    w_edge_ar1_uv = w_edge_ar1_uv.detach()
    w_edge_ar1_affine = w_edge_ar1_affine.detach()
    w_edge_ar1_affine_uv = w_edge_ar1_affine_uv.detach()

    eps_data = refine_params.get("eps_data", 1e-3)
    eps_ar1 = refine_params.get("eps_ar1", 1e-3)
    eps_ar0 = refine_params.get("eps_ar0", 1e-3)

    opt = torch.optim.Adam([flow.params], lr=refine_params.get("lr", 1e-1))

    # Create loss functions with specific parameters
    data_loss_fn = lambda x: charbonnier_loss(x, eps=eps_data)
    ar0_loss_fn = lambda x: charbonnier_loss(x, eps=eps_ar0)
    ar1_loss_fn = lambda x: charbonnier_loss(x, eps=eps_ar1)

    for t in range(refine_params.get("steps", 1000)):
        opt.zero_grad()
        I2w = flow.warp_image(input_images["image2"])
        resid = I2w - input_images["image1"]

        data = (data_loss_fn(resid) * w_edge_data).mean()

        ar0 = flow.ar0_terms(ar0_loss_fn)
        ar0_weighted = lambda_ar0_uv * ar0["uv"] + lambda_ar0_affine * ar0["b"]
        ar0_weighted = ar0_weighted.mean()

        ar1 = flow.ar1_terms(3, ar1_loss_fn)
        print(ar1["uv"].shape, w_edge_ar1_uv.shape)
        ar1_weighted = w_edge_ar1_uv * lambda_ar1_uv * ar1["uv"] + w_edge_ar1_affine * lambda_ar1_affine * ar1["b"] + w_edge_ar1_affine_uv * lambda_ar1_affine_uv * ar1["affine_uv"]
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