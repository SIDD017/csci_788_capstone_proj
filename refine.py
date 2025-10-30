import mlflow
import torch
import torch.nn.functional as F

from utils import warp_image_with_flow


def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)


def sobel_magnitude(t):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy)


def refine_flow(
    flow,
    refine_params=None,
):
    if flow.is_flow_refined == True:
        raise ValueError("Flow has already been refined")
    flow.is_flow_refined = True

    lambda_smooth = refine_params.get("lambda_smooth", 0.1)
    eps = refine_params.get("eps", 1e-3)
    w_edge = 1.0 / (1.0 + refine_params.get("edge_beta", 20.0) * sobel_magnitude(flow.image1))
    w_edge = w_edge.detach()
    opt = torch.optim.Adam([flow.params], lr=refine_params.get("lr", 1e-1))

    for t in range(refine_params.get("steps", 1000)):
        opt.zero_grad()
        I2w = warp_image_with_flow(flow)
        resid = I2w - flow.image1
        data = (charbonnier_loss(resid, eps=eps) * w_edge).mean()
        tv = flow.smoothness_tv()

        loss = data + lambda_smooth * sum(tv.values())
        loss.backward()
        opt.step()

        if (t % 50 == 0):
            flow.log_metrics["loss_log"].append(loss.item())
            with torch.no_grad():
                flow.log_metrics["data_loss_log"].append(data.item())
                flow.log_metrics["smoothness_loss_log"].append(sum(tv.values()).item())
                epe = flow.epe_error()
                flow.log_metrics["epe_log"].append(epe)
                ang = flow.angular_error()
                flow.log_metrics["angular_log"].append(ang)
                for k, v in tv.items():
                    flow.log_metrics[f"{k}_tv_log"].append(v.item())

            mlflow.log_metrics({k: lst[-1] for k, lst in flow.log_metrics.items()}, step=t)

        # TODO: Early break if loss is very low (convergence criteria)
    


# TODO: Coarse to fine refinement function that wraps refine_flow