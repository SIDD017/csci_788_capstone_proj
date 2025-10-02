from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
import torch

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
        return charbonnier_loss(dy, 1e-3).mean() + charbonnier_loss(dx, 1e-3).mean()

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
        return charbonnier_loss(torch.linalg.norm(dy, dim=-1), 1e-3).mean() + charbonnier_loss(torch.linalg.norm(dx, dim=-1), 1e-3).mean()
    
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
        pass



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
        x_new = xs + self.params[...,0]*localx + self.params[...,1]*localy + self.params[...,2]
        y_new = ys + self.params[...,3]*localx + self.params[...,4]*localy + self.params[...,5]
        return x_new, y_new
    
    def smoothness_tv(self):
        groups = {
            "A": (0, 4),
            "uv": (4, 6),
            "origin": (6, 8)
        }
        tv_terms = {}
        for key, (a, b) in groups.items():
            dy = self.params[1:, :, a:b] - self.params[:-1, :, a:b]
            dx = self.params[:, 1:, a:b] - self.params[:, :-1, a:b]
            tv_terms[key] = charbonnier_loss(torch.linalg.norm(dy, dim=-1), 1e-3).mean() + \
                            charbonnier_loss(torch.linalg.norm(dx, dim=-1), 1e-3).mean()
        tv = sum(tv_terms.values())
        return tv
    
    def get_origin_reg(self):
        aff = self.params
        origin_reg = torch.mean((aff[...,6] - self.xs_t)**2 + (aff[...,7] - self.ys_t)**2)
        return origin_reg

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
        pass