import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2 as cv


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


def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)


def sobel_magnitude(t):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    # TODO: Maybe add small epsilon here to avoid dividing by zero in edge weighting
    return torch.sqrt(gx*gx + gy*gy)


def warp_image_with_flow(flow, image=None):
    if image is None:
        image = flow.image2
    B, C, H, W = image.shape
    dev, dt = image.device, image.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    x_new, y_new = flow.warped_coords(xs, ys)
    # Convert to [-1,1] range for grid_sample
    gx = (x_new / (W - 1)) * 2 - 1
    gy = (y_new / (H - 1)) * 2 - 1
    # Make size (1,H,W,2), expected by grid_sample
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    # TODO: Padding mode?
    return F.grid_sample(image, grid, align_corners=True, mode="bilinear", padding_mode="border")


def debug_edge_weighting(w_edge):
    v_edge = w_edge.clone().cpu().numpy()[0,0]
    v_edge = (np.clip(v_edge, 0, 1) * 255).astype(np.uint8)
    cv.imshow("downweight edge", v_edge)
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit(0)


def refine_flow(
    flow,
    refine_params=None,
):
    if flow.is_flow_refined == True:
        raise ValueError("Flow has already been refined")
    flow.is_flow_refined = True

    lambda_smooth = refine_params.get("lambda_smooth", 0.1)

    # edge downweighting on I1
    w_edge = 1.0 / (1.0 + refine_params.get("edge_beta", 20.0) * sobel_magnitude(flow.image1))
    w_edge = w_edge.detach()

    # debug_edge_weighting(w_edge)

    opt = torch.optim.Adam([flow.params], lr=refine_params.get("lr", 1e-1))

    # Gradient descent to refine the flow
    for t in range(refine_params.get("steps", 1000)):
        opt.zero_grad()
        # Warp I2 using current flow estimate
        I2w = warp_image_with_flow(flow)
        resid = I2w - flow.image1
        data = (charbonnier_loss(resid, eps=refine_params.get("eps", 1e-3)) * w_edge).mean()
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

            mlflow.log_metrics({k: lst[-1] for k, lst in flow.log_metrics.items()}, step=t)

        # Early break if loss is very low (convergence criteria)
        # prev_loss = flow.log_metrics["loss_log"][-2] if len(flow.log_metrics["loss_log"]) > 1 else float('inf')
        # if prev_loss - loss.item() < 1e-8:
        #     print(f"Converged at iteration {t} with loss {loss.item():.6f}")
        #     break
    with torch.no_grad():
        I2w = warp_image_with_flow(flow)
        disp = flow.pred_flow()
        return disp.detach().cpu().numpy(), I2w.squeeze().cpu().numpy()
    





def refine_flow_coarse_to_fine(flow, refine_params=None):
    """
    Refine the flow estimates using a coarse-to-fine (pyramidal) approach.
    This function creates an image pyramid by downsampling the input images
    and an initial flow pyramid from flow.params. At the coarsest level the
    flow is refined iteratively, and then the refined flow is upsampled and
    accumulated at each higher resolution level.
    
    refine_params should contain (in addition to those used in refine_flow):
      - pyramid_levels: number of levels in the pyramid (default: 3)
      - scale_factor: downscale factor per level (default: 0.5)
      - steps: number of iterations per level (default: 1000)
      - lr: learning rate for optimization (default: 0.1)
      - lambda_smooth: smoothness weight (default: 0.1)
      - eps: epsilon for charbonnier loss (default: 1e-3)
    """
    if flow.is_flow_refined:
        raise ValueError("Flow has already been refined")
    
    # Pyramid parameters
    pyramid_levels = refine_params.get("pyramid_levels", 3)
    scale_factor = refine_params.get("scale_factor", 0.5)
    steps = refine_params.get("steps", 1000)
    lr = refine_params.get("lr", 1e-1)
    lambda_smooth = refine_params.get("lambda_smooth", 0.1)
    
    # Build image pyramids for image1 and image2
    image1_pyr = [flow.image1]
    image2_pyr = [flow.image2]
    for i in range(1, pyramid_levels):
        new_H = int(image1_pyr[-1].shape[2] * scale_factor)
        new_W = int(image1_pyr[-1].shape[3] * scale_factor)
        image1_pyr.append(
            torch.nn.functional.interpolate(flow.image1, size=(new_H, new_W), mode="bilinear", align_corners=True)
        )
        image2_pyr.append(
            torch.nn.functional.interpolate(flow.image2, size=(new_H, new_W), mode="bilinear", align_corners=True)
        )
    # Process pyramid from coarsest to finest
    image1_pyr = image1_pyr[::-1]
    image2_pyr = image2_pyr[::-1]
    
    # Build a pyramid for the initial flow parameters.
    # Downsample the initial flow and scale the translation components accordingly.
    flow_pyr = []
    for i, img in enumerate(image1_pyr):
        # Downsample flow.params to the current resolution.
        ref_size = (img.shape[2], img.shape[3])
        flow_level = torch.nn.functional.interpolate(flow.params.detach(), size=ref_size, mode="bilinear", align_corners=True)
        # Scale flow values (assuming these are displacement parameters)
        scale = scale_factor ** (pyramid_levels - 1 - i)
        flow_level = flow_level * scale
        flow_pyr.append(flow_level)
    
    refined_flow = None
    # Iterate from coarse to fine
    for lvl in range(pyramid_levels):
        # Set the current pyramid level images
        flow.image1 = image1_pyr[lvl]
        flow.image2 = image2_pyr[lvl]
        # Initialize flow.params at current level (accumulated from previous level if not coarsest)
        flow.params = flow_pyr[lvl].clone().requires_grad_(True)
        
        opt = torch.optim.Adam([flow.params], lr=lr)
        for t in range(steps):
            opt.zero_grad()
            # Warp image2 using current flow estimation
            I2w = warp_image_with_flow(flow)
            resid = I2w - flow.image1
            data = (charbonnier_loss(resid, eps=refine_params.get("eps", 1e-3))).mean()
            tv = flow.smoothness_tv()
            # Here we ignore the origin regularization if not desired
            loss = data + lambda_smooth * sum(tv.values())
            loss.backward()
            opt.step()

            if t % 50 == 0:
                print(f"Level {lvl}, Iteration {t}: Loss {loss.item():.6f}")
        
        # The refined flow at current level
        refined_flow = flow.params.detach()
        # If not at finest level, upsample and accumulate the refined flow into the next level's flow
        if lvl < pyramid_levels - 1:
            next_size = (image1_pyr[lvl + 1].shape[2], image1_pyr[lvl + 1].shape[3])
            up_refined = torch.nn.functional.interpolate(refined_flow, size=next_size, mode="bilinear", align_corners=True)
            # Since we downscaled by 'scale_factor', we need to rescale displacement values accordingly
            up_refined = up_refined / scale_factor
            flow_pyr[lvl + 1] = flow_pyr[lvl + 1] + up_refined

    # Restore images to original resolution.
    orig_H = flow.image1.shape[2]  # current resolution might be different; use flow.image1 of finest level
    orig_W = flow.image1.shape[3]
    flow.image1 = torch.nn.functional.interpolate(flow.image1, size=(orig_H, orig_W), mode="bilinear", align_corners=True)
    flow.image2 = torch.nn.functional.interpolate(flow.image2, size=(orig_H, orig_W), mode="bilinear", align_corners=True)
    flow.params = refined_flow
    flow.is_flow_refined = True

    # Compute final warped image using refined flow.
    I2w_final = warp_image_with_flow(flow)
    return flow.params.detach().cpu().numpy(), I2w_final.squeeze().cpu().numpy()