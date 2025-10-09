import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def np_im_to_torch(image_np):
    # TODO - convert to float if needed
    return torch.from_numpy(np.atleast_3d(image_np)).permute(2,0,1).unsqueeze(0)


def convert_torch_to_cv(image_tensor):
    # image_tensor shape is [1, C, H, W]
    img_np = image_tensor.cpu().numpy()[0]           # shape: [C, H, W]
    img_np = np.transpose(img_np, (1, 2, 0))           # shape: [H, W, C]
    # If single channel, convert to 2D
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]
    # Assume image data is in [0,1] float range; scale to [0,255] and convert to uint8
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
    return img_np


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


def plot_losses(flow):
    # plot everything in flow.log_metrics
    plt.figure(figsize=(18, 4))
    for i, (key, values) in enumerate(flow.log_metrics.items()):
        plt.subplot(1, len(flow.log_metrics), i + 1)
        plt.plot(values)
        plt.title(key)
    plt.show()


def warp_image_with_flow(flow, image=None):
    if image is None:
        image = flow.image2
    B, C, H, W = image.shape
    dev, dt = image.device, image.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    x_new, y_new = flow.pred_flow(xs, ys)
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
    data_loss_log = []
    smoothness_loss_log = []
    loss_log = []
    epe_log = []
    angular_log = []

    # Gradient descent to refine the flow
    for t in range(steps):
        # flow.visualize_params()  # For debugging
        # Initialize the gradients to zero
        opt.zero_grad()
        # Warp I2 using current flow estimate
        I2w = warp_image_with_flow(flow)
        # Data term: robust photometric error with Charbonnier loss
        resid = I2w - flow.image1
        data = (charbonnier_loss(resid, eps=eps) * w_edge).mean()

        # smoothness on flow with Charbonnier TV (forward differences)
        # flow: H x W x 2  (axis 0 = y/rows, axis 1 = x/cols)
        tv = flow.smoothness_tv()

        origin_reg = flow.get_origin_reg()

        # loss = data + lambda_smooth * tv + 0.1 * origin_reg
        loss = data + lambda_smooth * sum(tv.values())
        loss.backward()
        opt.step()

        if (t % 50 == 0):
            print(f"Iteration {t}: \nLoss={loss.item():.6f}")
            flow.log_metrics["loss_log"].append(loss.item())
            with torch.no_grad():
                flow.log_metrics["data_loss_log"].append(data.item())
                flow.log_metrics["smoothness_loss_log"].append(sum(tv.values()).item())
                # End-Point Error
                epe = flow.epe_error()
                print(f"\nEPE to GT: {epe:.4f}")
                flow.log_metrics["epe_log"].append(epe)
                # Angular error (in radians)
                ang = flow.angular_error()
                print(f"\nAngular err (rad): {ang:.4f}\n\n")
                flow.log_metrics["angular_log"].append(ang)
                # TV terms
                for k, v in tv.items():
                    flow.log_metrics[f"{k}_tv_log"].append(v.item())

        # Early break if loss is very low (convergence criteria)
        # prev_loss = flow.log_metrics["loss_log"][-2] if len(flow.log_metrics["loss_log"]) > 1 else float('inf')
        # if prev_loss - loss.item() < 1e-8:
        #     print(f"Converged at iteration {t} with loss {loss.item():.6f}")
        #     break

    # Plot loss and metrics
    plot_losses(flow)
    
    # # Log results for MLFlow/TensorBoard if needed
    # flow.log_metrics = {
    #     "loss_log": loss_log,
    #     "epe_log": epe_log,
    #     "angular_log": angular_log
    # }

    with torch.no_grad():
        I2w = warp_image_with_flow(flow)
        return flow.params.detach().cpu().numpy(), I2w.squeeze().cpu().numpy()