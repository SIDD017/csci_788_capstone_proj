import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def np_im_to_torch(image_np):
    # TODO - convert to float if needed
    return torch.from_numpy(np.atleast_3d(image_np)).permute(2,0,1).unsqueeze(0)


# Utility functions for flow refinement
def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)


def sobel_magnitude(t):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    # TODO: Maybe add small epsilon here to avoid dividing by zero in edge weighting
    return torch.sqrt(gx*gx + gy*gy)


def plot_losses(loss_log, epe_log, angular_log):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(loss_log)
    plt.title("Total Loss")
    plt.subplot(1,3,2)
    plt.plot(epe_log)
    plt.title("EPE to GT")
    plt.subplot(1,3,3)
    plt.plot(angular_log)
    plt.title("Angular err (rad)")
    plt.show()


def warp_image_with_flow(flow, image=None):
    if image is None:
        image = flow.image2
    B, C, H, W = image.shape
    dev, dt = image.device, image.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    x_new, y_new = flow.warp_with_flow(xs, ys)
    # Convert to [-1,1] range for grid_sample
    gx = (x_new / (W - 1)) * 2 - 1
    gy = (y_new / (H - 1)) * 2 - 1
    # Make size (1,H,W,2), expected by grid_sample
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    # TODO: Padding mode?
    return F.grid_sample(image, grid, align_corners=True, mode="bilinear", padding_mode="border")


def refine_flow(
    flow,
    refine_params=None,
):
    if flow.is_flow_refined == True:
            raise ValueError("Flow has already been refined")
    flow.is_flow_refined = True

    # Refinement parameters
    steps = refine_params.get("steps", 300)
    lr = refine_params.get("lr", 1e-1)
    edge_beta = refine_params.get("edge_beta", 20.0)
    eps = refine_params.get("eps", 1e-3)
    lambda_smooth = refine_params.get("lambda_smooth", 0.1)

    # edge downweighting on I1
    w_edge = 1.0 / (1.0 + edge_beta * sobel_magnitude(flow.image1))
    w_edge = w_edge.detach()

    # Optimizer
    opt = torch.optim.Adam([flow.params], lr=lr)

    # Log metrics
    loss_log = []
    epe_log = []
    angular_log = []

    # Gradient descent to refine the flow
    for t in range(steps):
        # Initialize the gradients to zero
        opt.zero_grad()
        # Warp I2 using current flow estimate
        I2w = warp_image_with_flow(flow)
        # Data term: robust photometric error with Charbonnier loss
        resid = I2w - flow.image1
        data = (charbonnier_loss(resid, eps=eps) * w_edge).mean()

        # smoothness on flow with Charbonnier TV (forward differences)
        # flow: H x W x 2  (axis 0 = y/rows, axis 1 = x/cols)
        dy = torch.linalg.norm(flow.params[1:, :, :] - flow.params[:-1, :, :], dim=-1)   # (H-1) x W x 2
        dx = torch.linalg.norm(flow.params[:, 1:, :] - flow.params[:, :-1, :], dim=-1)   # H x (W-1) x 2
        tv = charbonnier_loss(dy, 1e-3).mean() + charbonnier_loss(dx, 1e-3).mean()

        loss = data + lambda_smooth * tv
        loss.backward()
        opt.step()

        if (t % 50 == 0):
            print(f"Iteration {t}: \nLoss={loss.item():.6f}")
            loss_log.append(loss.item())
            with torch.no_grad():
                # End-Point Error
                epe = flow.epe_error()
                print(f"\nEPE to GT: {epe:.4f}")
                epe_log.append(epe)
                # Angular error (in radians)
                ang = flow.angular_error()
                print(f"\nAngular err (rad): {ang:.4f}\n\n")
                angular_log.append(ang)

        # Early break if loss is very low (convergence criteria)
        if loss.item() < 1e-3:
            print(f"Converged at iteration {t} with loss {loss.item():.6f}")
            break

    # Plot loss and metrics
    plot_losses(loss_log, epe_log, angular_log)
    
    # Log results for MLFlow/TensorBoard if needed
    flow.log_metrics = {
        "loss_log": loss_log,
        "epe_log": epe_log,
        "angular_log": angular_log
    }

    with torch.no_grad():
        I2w = warp_image_with_flow(flow)
        return flow.params.detach().cpu().numpy(), I2w.squeeze().cpu().numpy()