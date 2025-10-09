from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from init_flow import calculate_initial_flow
from refine_utils import np_im_to_torch, charbonnier_loss, convert_torch_to_cv
from utils import visualize_flow_hsv, visualize_gt_flow_hsv

# Abstract base class for optical flow (2 Param lucas kanade and 6/8 param affine)
class Flow(ABC):
    def __init__(self, image1, image2, gt_flow, init_params, use_opencv=False):
        # Calculate the initial flow and setup everything to be torch tensors
        # on device, ready for refinement
        self.init_flow = calculate_initial_flow(image1, 
                                             image2, 
                                             init_params["levels"], 
                                             init_params["window_size"], 
                                             init_params["alpha"], 
                                             init_params["goodness_threshold"], 
                                             use_opencv)
        self.image1 = np_im_to_torch(image1).to(init_params.get("device", "cpu"))
        self.image2 = np_im_to_torch(image2).to(init_params.get("device", "cpu"))
        self.gt_flow = torch.from_numpy(gt_flow.astype(np.float32)).to(init_params.get("device", "cpu"))
        self.is_flow_refined = False
        self.log_metrics = {"data_loss_log": [],
                            "smoothness_loss_log": [],
                            "loss_log": [],
                            "epe_log": [],
                            "angular_log": [],
                            "A_tv_log": [],
                            "uv_tv_log": [],
                            "origin_tv_log": []}

    # Warp an image using the flow field
    @abstractmethod
    def pred_flow(self, xs, ys):
        pass

    @abstractmethod
    def smoothness_tv(self):
        pass

    @abstractmethod
    def get_origin_reg(self):
        pass

    @abstractmethod
    def epe_error(self):
        pass


    @abstractmethod
    def visualize_flow(self):
        pass

    # Visualize the flow field
    @abstractmethod
    def visualize_params(self):
        pass


class CustomLucasKanadeFlow(Flow):
    def __init__(self, image1, image2, gt_flow, init_params, use_opencv=False):
        super().__init__(image1, image2, gt_flow, init_params, use_opencv=use_opencv)
        temp = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        temp = torch.from_numpy(temp).to(init_params.get("device", "cpu"))
        self.params = temp.clone().requires_grad_(True)

    
    def pred_flow(self, xs, ys):
        xw = xs + self.params[...,0]
        yw = ys + self.params[...,1]
        return xw, yw
    
    def smoothness_tv(self):
        dy = torch.linalg.norm(self.params[1:, :, :] - self.params[:-1, :, :], dim=-1)   # (H-1) x W x 2
        dx = torch.linalg.norm(self.params[:, 1:, :] - self.params[:, :-1, :], dim=-1)   # H x (W-1) x 2
        return {"uv": charbonnier_loss(dy, 1e-3).mean() + charbonnier_loss(dx, 1e-3).mean()}

    def get_origin_reg(self):
        return 0.0

    def epe_error(self):
        epe = torch.linalg.norm(self.params - self.gt_flow, dim=-1)  # H x W
        return epe.mean().item()
    

    def angular_error(self):
        u, v   = self.params[...,0], self.params[...,1]
        ug, vg = self.gt_flow[...,0], self.gt_flow[...,1]
        num = u*ug + v*vg
        epsilon = 1e-8
        gtmag = torch.sqrt(ug*ug + vg*vg)
        mask = gtmag > 0.1
        den = torch.sqrt(u*u + v*v + epsilon) * torch.sqrt(ug*ug + vg*vg + epsilon)
        ang = torch.acos(torch.clamp(num/den, -1.0, 1.0)).mean().item()
        return ang
    
    def visualize_flow(self):
        cv.imshow("Image1", convert_torch_to_cv(self.image1))
        cv.imshow("Image2", convert_torch_to_cv(self.image2))
        disp = visualize_flow_hsv(self.params.detach().cpu().numpy())
        cv.imshow("Dense refined flow", disp)
        display = visualize_flow_hsv(self.init_flow)
        gt_display = visualize_gt_flow_hsv(self.gt_flow.cpu().numpy())
        cv.imshow("Optical flow (custom)", display)
        cv.imshow("Optical flow (ground truth)", gt_display)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def visualize_params(self):
        pass


class AffineFlow(Flow):
    def __init__(self, image1, image2, gt_flow, init_params, use_opencv=False):
        super().__init__(image1, image2, gt_flow, init_params, use_opencv=False)
        # Initialize the affine parameters to identity + translation from init flow
        temp = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        H, W, _ = temp.shape
        # create a meshgrid for pixel coordinates (note: x = column index, y = row index)
        ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                            np.arange(W, dtype=np.float32),
                            indexing="ij")
        self.params = np.zeros((H, W, 6), dtype=np.float32)
        self.params[...,0] = 1.0
        self.params[...,1] = 0.0
        self.params[...,2] = temp[...,0]
        self.params[...,3] = 0.0
        self.params[...,4] = 1.0
        self.params[...,5] = temp[...,1]
        self.params = torch.from_numpy(self.params).to(init_params.get("device", "cpu"))
        self.params = self.params.clone().requires_grad_(True)
        self.xs_t = torch.from_numpy(xs).to(init_params.get("device", "cpu"))
        self.ys_t = torch.from_numpy(ys).to(init_params.get("device", "cpu"))

    
    def pred_flow(self, xs, ys):
        x_new = self.params[...,0]*xs + self.params[...,1]*ys + self.params[...,2]
        y_new = self.params[...,3]*xs + self.params[...,4]*ys + self.params[...,5]
        return x_new, y_new
    
    def smoothness_tv(self):
        dy = self.params[1:, :, :] - self.params[:-1, :, :]   # (H-1) x W x 6
        dx = self.params[:, 1:, :] - self.params[:, :-1, :]   # H x (W-1) x 6
        return {"uv":charbonnier_loss(torch.linalg.norm(dy, dim=-1), 1e-3).mean() + charbonnier_loss(torch.linalg.norm(dx, dim=-1), 1e-3).mean()}
    
    def get_origin_reg(self):
        return 0.0
    
    def epe_error(self):
        x_new, y_new = self.pred_flow(self.xs_t, self.ys_t)
        pred_flow = torch.stack([x_new, y_new], dim=-1)
        epe = torch.linalg.norm(pred_flow - self.gt_flow, dim=-1)  # H x W
        return epe.mean().item()

    def angular_error(self):
        u, v = self.pred_flow(self.xs_t, self.ys_t)
        # u, v   = pred_flow[...,0], pred_flow[...,1]
        ug, vg = self.gt_flow[...,0], self.gt_flow[...,1]
        num = u*ug + v*vg
        epsilon = 1e-8
        gtmag = torch.sqrt(ug*ug + vg*vg)
        mask = gtmag > 0.1
        den = torch.sqrt(u*u + v*v + epsilon) * torch.sqrt(ug*ug + vg*vg + epsilon)
        ang = torch.acos(torch.clamp(num/den, -1.0, 1.0)).mean().item()
        return ang

    def visualize_params(self):
         # Make small reference patch (letter R)
        ref_np = np.zeros((16, 16), dtype=np.uint8)
        cv.putText(ref_np, "R", (1, 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv.LINE_AA)
        ref = torch.tensor(ref_np, dtype=torch.float32, device="cuda") / 255.0  # [H,W] float
        ref = ref.unsqueeze(0).unsqueeze(0)  # [N=1, C=1, H, W]
        ref_h, ref_w = ref.shape[2:]

        param_grid = self.params.detach().clone()  # [H, W, 6]
        H, W = param_grid.shape[:2]
        # param_grid[...,0] = 0.5
        # param_grid[...,1] = 0.0
        param_grid[...,2] = 0.0
        # param_grid[...,3] = 0.0
        # param_grid[...,4] = 0.5
        param_grid[...,5] = 0.0

        # Reshape into batch of 2x3 matrices
        M = param_grid.view(-1, 2, 3)  # [H*W, 2, 3]

        # Build normalized sampling grids for each affine transform
        # F.affine_grid generates coordinates for grid_sample
        grids = F.affine_grid(M, size=(H*W, 1, ref_h, ref_w), align_corners=False)  # [H*W, H, W, 2]

        # Apply all warps in parallel
        warped = F.grid_sample(ref.expand(H*W, -1, -1, -1), grids, align_corners=False)  # [H*W, 1, h, w]

        # Tile into big canvas
        warped = warped.squeeze(1)  # [H*W, h, w]
        canvas = warped.view(H, W, ref_h, ref_w).permute(0,2,1,3).reshape(H*ref_h, W*ref_w)

        plt.imshow(canvas.detach().cpu(), cmap="gray")
        plt.axis("off")                                                                                                                                                                                                                                                                                                                                                                                        
        plt.show()

        # Visualize translation parameters (indices 2 and 5)
        translation = self.params[..., [2, 5]].detach().cpu().numpy()  # shape: [H, W, 2]
        translation_hsv = visualize_flow_hsv(translation)
        cv.imshow("Translation (HSV)", translation_hsv)

        cv.waitKey(0)
        cv.destroyAllWindows()





class AffineFlowWithLocalOrigins(Flow):
    def __init__(self, image1, image2, gt_flow, init_params, use_opencv=False):
        super().__init__(image1, image2, gt_flow, init_params, use_opencv=False)
        # Initialize any additional parameters for affine flow and local origins
        temp = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        H, W, _ = temp.shape
        # create a meshgrid for pixel coordinates (note: x = column index, y = row index)
        ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                            np.arange(W, dtype=np.float32),
                            indexing="ij")
        self.params = np.zeros((H, W, 8), dtype=np.float32)
        self.params[...,0] = 1.0
        self.params[...,1] = 0.0
        self.params[...,2] = temp[...,0]
        self.params[...,3] = 0.0
        self.params[...,4] = 1.0
        self.params[...,5] = temp[...,1]
        self.params[...,6] = xs
        self.params[...,7] = ys
        self.params = torch.from_numpy(self.params).to(init_params.get("device", "cpu"))
        self.params = self.params.clone().requires_grad_(True)
        self.xs_t = torch.from_numpy(xs).to(init_params.get("device", "cpu"))
        self.ys_t = torch.from_numpy(ys).to(init_params.get("device", "cpu"))


    def pred_flow(self, xs, ys):
        ox = self.params[...,6]
        oy = self.params[...,7]
        localx = xs - ox
        localy = ys - oy
        x_new = ox + self.params[...,0]*localx + self.params[...,1]*localy + self.params[...,2]
        y_new = oy + self.params[...,3]*localx + self.params[...,4]*localy + self.params[...,5]
        return x_new, y_new
    
    def smoothness_tv(self):
        groups = {
            "A": [0, 1, 3, 4],
            "uv": [2, 5],
            "origin": [6, 7]
        }
        tv_terms = {}
        for key, indices in groups.items():
            group = self.params[..., indices]  # use fancy indexing
            dy = group[1:, :, :] - group[:-1, :, :]
            dx = group[:, 1:, :] - group[:, :-1, :]
            tv_terms[key] = charbonnier_loss(torch.linalg.norm(dy, dim=-1), 1e-3).mean() + \
                            charbonnier_loss(torch.linalg.norm(dx, dim=-1), 1e-3).mean()
        return tv_terms
        # tv_terms = {}
        # for key, (a, b) in groups.items():
        #     dy = self.params[1:, :, a:b] - self.params[:-1, :, a:b]
        #     dx = self.params[:, 1:, a:b] - self.params[:, :-1, a:b]
        #     tv_terms[key] = charbonnier_loss(torch.linalg.norm(dy, dim=-1), 1e-3).mean() + \
        #                     charbonnier_loss(torch.linalg.norm(dx, dim=-1), 1e-3).mean()
        # # tv = sum(tv_terms.values())
        # # return tv
        # return tv_terms
    
    def get_origin_reg(self):
        aff = self.params
        origin_reg = torch.mean((aff[...,6] - self.xs_t)**2 + (aff[...,7] - self.ys_t)**2)
        return origin_reg

    def epe_error(self):
        x_new, y_new = self.pred_flow(self.xs_t, self.ys_t)
        pred_flow = torch.stack([x_new, y_new], dim=-1)
        grid = torch.stack([self.xs_t, self.ys_t], dim=-1)
        disp = pred_flow - grid
        epe = torch.linalg.norm(disp - self.gt_flow, dim=-1)
        return epe.mean().item()
    

    def angular_error(self):
        x_new, y_new = self.pred_flow(self.xs_t, self.ys_t)
        pred_flow = torch.stack([x_new, y_new], dim=-1)
        grid = torch.stack([self.xs_t, self.ys_t], dim=-1)
        disp = pred_flow - grid  # Convert absolute coordinates to displacement
        u, v   = disp[..., 0], disp[..., 1]
        ug, vg = self.gt_flow[..., 0], self.gt_flow[..., 1]

        num = u * ug + v * vg
        epsilon = 1e-8
        pred_mag = torch.sqrt(u * u + v * v + epsilon)
        gt_mag   = torch.sqrt(ug * ug + vg * vg + epsilon)
        den = pred_mag * gt_mag
        ang = torch.acos(torch.clamp(num / den, -1.0, 1.0))
        return ang.mean().item()

    def visualize_flow(self):
        # Display the original images
        cv.imshow("Image1", convert_torch_to_cv(self.image1))
        cv.imshow("Image2", convert_torch_to_cv(self.image2))
        
        # Compute predicted absolute flow using the learned affine transform with local origins
        x_new, y_new = self.pred_flow(self.xs_t, self.ys_t)
        pred_flow_abs = torch.stack([x_new, y_new], dim=-1)
        
        # Convert to displacement by subtracting the original pixel grid
        grid = torch.stack([self.xs_t, self.ys_t], dim=-1)
        disp = pred_flow_abs - grid
        disp_np = disp.detach().cpu().numpy()
        
        # Visualize the predicted displacement field (refined flow)
        disp_hsv = visualize_flow_hsv(disp_np)
        cv.imshow("Dense refined flow", disp_hsv)
        
        # Visualize the initial flow and ground truth for comparison
        disp_init = visualize_flow_hsv(self.init_flow)
        gt_disp = visualize_gt_flow_hsv(self.gt_flow.cpu().numpy())
        cv.imshow("Optical flow (initial)", disp_init)
        cv.imshow("Optical flow (ground truth)", gt_disp)
        
        cv.waitKey(0)
        cv.destroyAllWindows()

    def visualize_params(self):
        # Make small reference patch (letter R)
        ref_np = np.zeros((16, 16), dtype=np.uint8)
        cv.putText(ref_np, "R", (1, 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv.LINE_AA)
        ref = torch.tensor(ref_np, dtype=torch.float32, device="cuda") / 255.0  # [H,W] float
        ref = ref.unsqueeze(0).unsqueeze(0)  # [N=1, C=1, H, W]
        ref_h, ref_w = ref.shape[2:]

        param_grid = self.params[:,:,:6].detach().clone()  # [H, W, 6]
        H, W = param_grid.shape[:2]
        # param_grid[...,0] = 1
        # param_grid[...,1] = 0.0
        param_grid[...,2] = 0.0
        # param_grid[...,3] = 0.0
        # param_grid[...,4] = 1
        param_grid[...,5] = 0.0

        # Reshape into batch of 2x3 matrices
        M = param_grid.view(-1, 2, 3)  # [H*W, 2, 3]

        # Build normalized sampling grids for each affine transform
        # F.affine_grid generates coordinates for grid_sample
        grids = F.affine_grid(M, size=(H*W, 1, ref_h, ref_w), align_corners=False)  # [H*W, H, W, 2]

        # Apply all warps in parallel
        warped = F.grid_sample(ref.expand(H*W, -1, -1, -1), grids, align_corners=False)  # [H*W, 1, h, w]

        # Tile into big canvas
        warped = warped.squeeze(1)  # [H*W, h, w]
        canvas = warped.view(H, W, ref_h, ref_w).permute(0,2,1,3).reshape(H*ref_h, W*ref_w)

        plt.imshow(canvas.detach().cpu(), cmap="gray")
        plt.axis("off")
        plt.show()

        # Visualize translation parameters (indices 2 and 5)
        translation = self.params[..., [2, 5]].detach().cpu().numpy()  # shape: [H, W, 2]
        translation_hsv = visualize_flow_hsv(translation)
        cv.imshow("Translation (HSV)", translation_hsv)

        # Visualize local origins (indices 6 and 7)
        local_origins = self.params[..., [6, 7]].detach().cpu().numpy()  # shape: [H, W, 2]
        local_origins_hsv = visualize_flow_hsv(local_origins)
        cv.imshow("Local Origins (HSV)", local_origins_hsv)

        cv.waitKey(0)
        cv.destroyAllWindows()