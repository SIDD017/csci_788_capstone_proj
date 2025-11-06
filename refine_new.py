import torch
from flow_new import BaseFlow, Metrics
import torch.nn.functional as F
import cv2 as cv

from utils import convert_torch_to_cv


def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)


def sobel_magnitude(t):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy)


def refine(flow: BaseFlow, input_images: dict[str, torch.Tensor], refine_params: dict):
    lambda_smooth = refine_params.get("lambda_smooth", 0.1)
    eps = refine_params.get("eps", 1e-3)
    w_edge = 1.0 / (1.0 + refine_params.get("edge_beta", 20.0) * sobel_magnitude(input_images["image1"]))
    w_edge = w_edge.detach()
    opt = torch.optim.Adam([flow.params], lr=refine_params.get("lr", 1e-1))

    for t in range(refine_params.get("steps", 1000)):
        opt.zero_grad()
        I2w = flow.warp_image(input_images["image2"])
        resid = I2w - input_images["image1"]

        data = (charbonnier_loss(resid, eps=eps) * w_edge).mean()
        ar0 = flow.ar0_terms(charbonnier_loss)
        ar1 = flow.ar1_terms(3, charbonnier_loss)

        loss = data + lambda_smooth * sum(ar1.values()) + 0.0001 * sum(ar0.values())
        loss.backward()
        opt.step()

        if (t % 50 == 0):
            print(loss.item())
            with torch.no_grad():
                print("epe : ", Metrics.epe(input_images["gtimage"], flow.uv))
                print("ang error : ", Metrics.angular_error(input_images["gtimage"], flow.uv))

            # mlflow.log_metrics({k: lst[-1] for k, lst in flow.log_metrics.items()}, step=t)