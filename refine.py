import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)

def sobel_magnitude(t):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    # TODO: Maybe add small epsilon here to avoid dividing by zero in edge weighting
    return torch.sqrt(gx*gx + gy*gy)

def _warp_with_flow(I, flow):
    B, C, H, W = I.shape
    dev, dt = I.device, I.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    xw = xs + flow[...,0]
    yw = ys + flow[...,1]
    # Convert to [-1,1] range for grid_sample
    gx = (xw / (W - 1)) * 2 - 1
    gy = (yw / (H - 1)) * 2 - 1
    # Make size (1,H,W,2), expected by grid_sample
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    # TODO: Padding mode?
    return F.grid_sample(I, grid, align_corners=True, mode="bilinear", padding_mode="border")


def _warp_with_affine_flow(I, aff_flow):
    B, C, H, W = I.shape
    dev, dt = I.device, I.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    x_new = aff_flow[...,0]*xs + aff_flow[...,1]*ys + aff_flow[...,2]
    y_new = aff_flow[...,3]*xs + aff_flow[...,4]*ys + aff_flow[...,5]
    # Convert to [-1,1] range for grid_sample
    gx = (x_new / (W - 1)) * 2 - 1
    gy = (y_new / (H - 1)) * 2 - 1
    # Make size (1,H,W,2), expected by grid_sample
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    # TODO: Padding mode?
    return F.grid_sample(I, grid, align_corners=True, mode="bilinear", padding_mode="border")

def plot_losses(loss_log, epe_log=None, angular_log=None, save_plot=False, plot_path="loss/loss_plot.png"):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(loss_log)
    plt.title("Total Loss")
    if epe_log is not None:
        plt.subplot(1,3,2)
        plt.plot(epe_log)
        plt.title("EPE to GT")
    if angular_log is not None:
        plt.subplot(1,3,3)
        plt.plot(angular_log)
        plt.title("Angular err (rad)")
    if save_plot:
        plt.savefig(plot_path)
    else:
        plt.show()

def refine_dense_flow(
    image1_np,
    image2_np,
    init_flow_uv,
    gt_flow_uv=None,
    steps=300,
    lr=1e-1,
    edge_beta=20.0,
    eps=1e-3,
    lambda_smooth=0.1,
    device="cpu",
):
    I1 = torch.from_numpy(image1_np).unsqueeze(0).unsqueeze(0).to(device)
    I2 = torch.from_numpy(image2_np).unsqueeze(0).unsqueeze(0).to(device)

    # edge downweighting on I1
    w_edge = 1.0 / (1.0 + edge_beta * sobel_magnitude(I1))
    w_edge = w_edge.detach()

    # convert nans to zeros in the initial flow form LK
    init = np.nan_to_num(init_flow_uv.astype(np.float32), nan=0.0)
    flow = torch.from_numpy(init).to(device)
    flow = flow.clone().requires_grad_(True)

    opt = torch.optim.Adam([flow], lr=lr)

    # Log metrics
    loss_log = []
    epe_log = []
    angular_log = []

    if gt_flow_uv is not None:
        gp = torch.from_numpy(gt_flow_uv.astype(np.float32)).to(device)

    for t in range(steps):
        opt.zero_grad()
        I2w = _warp_with_flow(I2, flow)
        resid = I2w - I1 
        data = (charbonnier_loss(resid, eps=eps) * w_edge).mean()

        # smoothness on flow with Charbonnier TV (forward differences)
        # flow: H x W x 2  (axis 0 = y/rows, axis 1 = x/cols)
        dy = torch.linalg.norm(flow[1:, :, :] - flow[:-1, :, :], dim=-1)   # (H-1) x W x 2
        dx = torch.linalg.norm(flow[:, 1:, :] - flow[:, :-1, :], dim=-1)   # H x (W-1) x 2
        tv = charbonnier_loss(dy, 1e-3).mean() + charbonnier_loss(dx, 1e-3).mean()

        loss = data + lambda_smooth * tv
        loss.backward()
        opt.step()

        if (t % 50 == 0):
            print(f"Iteration {t}: \nLoss={loss.item():.6f}")
            loss_log.append(loss.item())
            if gt_flow_uv is not None:
                with torch.no_grad():
                    # End-Point Error
                    epe = torch.linalg.norm(flow - gp, dim=-1).mean().item()
                    print(f"\nEPE to GT: {epe:.4f}")
                    epe_log.append(epe)
                    # Angular error (in radians)
                    u, v   = flow[...,0], flow[...,1]
                    ug, vg = gp[...,0],   gp[...,1]
                    num = u*ug + v*vg
                    epsilon = 1e-8
                    den = torch.sqrt(u*u + v*v + epsilon) * torch.sqrt(ug*ug + vg*vg + epsilon)
                    ang = torch.acos(torch.clamp(num/den, -1.0, 1.0)).mean().item()
                    print(f"\nAngular err (rad): {ang:.4f}\n\n")
                    angular_log.append(ang)

        # Early break if loss is very low (convergence criteria)
        if loss.item() < 1e-5:
            print(f"Converged at iteration {t} with loss {loss.item():.6f}")
            break

    # Plot loss and metrics
    plot_losses(loss_log, epe_log if gt_flow_uv is not None else None, 
                angular_log if gt_flow_uv is not None else None,
                save_plot=False)

    with torch.no_grad():
        I2w = _warp_with_flow(I2, flow)
        return flow.detach().cpu().numpy(), I2w.squeeze().cpu().numpy()
    


def refine_dense_affine_flow(
    image1_np,
    image2_np,
    init_flow_uv,
    gt_flow_uv=None,
    steps=300,
    lr=1e-1,
    edge_beta=20.0,
    eps=1e-3,
    lambda_smooth=0.1,
    device="cpu",
):
    I1 = torch.from_numpy(image1_np).unsqueeze(0).unsqueeze(0).to(device)
    I2 = torch.from_numpy(image2_np).unsqueeze(0).unsqueeze(0).to(device)

    # edge downweighting on I1
    w_edge = 1.0 / (1.0 + edge_beta * sobel_magnitude(I1))
    w_edge = w_edge.detach()

    # convert NaNs to zeros in the initial flow from LK
    init = np.nan_to_num(init_flow_uv.astype(np.float32), nan=0.0)
    H, W, _ = init.shape

    # create a meshgrid for pixel coordinates (note: x = column index, y = row index)
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32),
                          indexing="ij")

    # Initialize the affine parameters to identity + translation from init flow
    aff_init = np.zeros((H, W, 6), dtype=np.float32)
    aff_init[...,0] = 1.0
    aff_init[...,1] = 0.0
    aff_init[...,2] = xs + init[...,0]
    aff_init[...,3] = 0.0
    aff_init[...,4] = 1.0
    aff_init[...,5] = ys + init[...,1]

    aff = torch.from_numpy(aff_init).to(device)
    aff = aff.clone().requires_grad_(True)

    opt = torch.optim.Adam([aff], lr=lr)

    # Log metrics
    loss_log = []
    epe_log = []
    angular_log = []

    if gt_flow_uv is not None:
        gp = torch.from_numpy(gt_flow_uv.astype(np.float32)).to(device)
    
    # Precompute the meshgrid for loss computations
    ys_t = torch.from_numpy(ys).to(device)
    xs_t = torch.from_numpy(xs).to(device)

    for t in range(steps):
        opt.zero_grad()
        I2w = _warp_with_affine_flow(I2, aff)
        resid = I2w - I1 
        data = (charbonnier_loss(resid, eps=eps) * w_edge).mean()
        
        # smoothness on the affine field (TV loss on all parameters)
        dy = torch.linalg.norm(aff[1:, :, :] - aff[:-1, :, :], dim=-1)
        dx = torch.linalg.norm(aff[:, 1:, :] - aff[:, :-1, :], dim=-1)
        tv = charbonnier_loss(dy, 1e-3).mean() + charbonnier_loss(dx, 1e-3).mean()

        loss = data + lambda_smooth * tv
        loss.backward()
        opt.step()

        if (t % 50 == 0):
            print(f"Iteration {t}: \nLoss={loss.item():.6f}")
            loss_log.append(loss.item())
            if gt_flow_uv is not None:
                with torch.no_grad():
                    pred_flow = torch.stack([
                        aff[...,0]*xs_t + aff[...,1]*ys_t + aff[...,2] - xs_t, 
                        aff[...,3]*xs_t + aff[...,4]*ys_t + aff[...,5] - ys_t], 
                        dim=-1)
                    # End-Point Error
                    epe = torch.linalg.norm(pred_flow - gp, dim=-1).mean().item()
                    print(f"\nEPE to GT: {epe:.4f}")
                    epe_log.append(epe)
                    # Angular error (in radians)
                    u, v   = pred_flow[...,0], pred_flow[...,1]
                    ug, vg = gp[...,0],   gp[...,1]
                    num = u*ug + v*vg
                    epsilon = 1e-8
                    den = torch.sqrt(u*u + v*v + epsilon) * torch.sqrt(ug*ug + vg*vg + epsilon)
                    ang = torch.acos(torch.clamp(num/den, -1.0, 1.0)).mean().item()
                    print(f"\nAngular err (rad): {ang:.4f}\n\n")
                    angular_log.append(ang)

        # Early break if loss is very low (convergence criteria)
        if loss.item() < 1e-5:
            print(f"Converged at iteration {t} with loss {loss.item():.6f}")
            break

    # Plot loss and metrics
    plot_losses(loss_log, epe_log if gt_flow_uv is not None else None, 
                angular_log if gt_flow_uv is not None else None,
                save_plot=False)

    # After refinement, compute the final warped image.
    with torch.no_grad():
        I2w = _warp_with_affine_flow(I2, aff)
        # Return the predicted flow: convert the affine field to a displacement field.
        pred_flow = torch.stack([
                        aff[...,0]*xs_t + aff[...,1]*ys_t + aff[...,2] - xs_t, 
                        aff[...,3]*xs_t + aff[...,4]*ys_t + aff[...,5] - ys_t], 
                        dim=-1)
        return pred_flow.detach().cpu().numpy(), I2w.squeeze().cpu().numpy()