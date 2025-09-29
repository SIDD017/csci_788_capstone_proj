from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
import torch

from init_flow import calculate_initial_flow
from refine_utils import np_im_to_torch

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
    def warp_with_flow(self, xs, ys):
        pass

    @abstractmethod
    def epe_error(self):
        pass

    # Visualize the flow field
    @abstractmethod
    def visualize_params(self):
        pass

    # Log results to TensorBoard/MLFlow repository
    @abstractmethod
    def log_results(self):
        pass


class CustomLucasKanadeFlow(Flow):
    def __init__(self, image1, image2, gt_flow, init_params, use_opencv=False):
        super().__init__(image1, image2, gt_flow, init_params, use_opencv=use_opencv)
        self.init_flow = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        self.init_flow = torch.from_numpy(self.init_flow).to(init_params.get("device", "cpu"))
        self.params = self.init_flow.clone().requires_grad_(True)

    
    def warp_with_flow(self, xs, ys):
        xw = xs + self.params[...,0]
        yw = ys + self.params[...,1]
        return xw, yw
    
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

    def log_results(self):
        pass  
    


class AffineFlow(Flow):
    def __init__(self, image1, image2, init_params, use_opencv=False):
        super().__init__(image1, image2, init_params, use_opencv=use_opencv)
        # Initialize the affine parameters to identity + translation from init flow
        self.init_flow = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        H, W = self.init_flow.shape
        self.params = np.zeros((H, W, 6), dtype=np.float32)
        self.params[...,0] = 1.0
        self.params[...,1] = 0.0
        self.params[...,2] = self.init_flow[...,0]
        self.params[...,3] = 0.0
        self.params[...,4] = 1.0
        self.params[...,5] = self.init_flow[...,1]
        self.params = torch.from_numpy(self.params).to(init_params.get("device", "cpu"))
        self.params = self.params.clone().requires_grad_(True)

    
    def warp_with_flow(self, xs, ys):
        x_new = self.params[...,0]*xs + self.params[...,1]*ys + self.params[...,2]
        y_new = self.params[...,3]*xs + self.params[...,4]*ys + self.params[...,5]
        return x_new, y_new
    
    def epe_error(self):
        pass
    

    def angular_error(self):
        pass

    def visualize_params(self):
        pass

    def log_results(self):
        pass



class AffineFlowWithLocalOrigins(Flow):
    def __init__(self, image1, image2, init_params, use_opencv=False):
        super().__init__(image1, image2, init_params, use_opencv=use_opencv)
        # Initialize any additional parameters for affine flow and local origins
        self.init_flow = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        H, W = self.init_flow.shape
        self.params = np.zeros((H, W, 6), dtype=np.float32)
        self.params[...,0] = 1.0
        self.params[...,1] = 0.0
        self.params[...,2] = self.init_flow[...,0]
        self.params[...,3] = 0.0
        self.params[...,4] = 1.0
        self.params[...,5] = self.init_flow[...,1]
        self.params = torch.from_numpy(self.params).to(init_params.get("device", "cpu"))
        self.params = self.params.clone().requires_grad_(True)


    def warp_with_flow(self, xs, ys):
        ox = self.params[...,6]
        oy = self.params[...,7]
        localx = xs - ox
        localy = ys - oy
        x_new = xs + self.params[...,0]*localx + self.params[...,1]*localy + self.params[...,2]
        y_new = ys + self.params[...,3]*localx + self.params[...,4]*localy + self.params[...,5]
        return x_new, y_new
    
    def epe_error(self):
        pass
    

    def angular_error(self):
        pass

    def visualize_params(self):
        pass

    def log_results(self):
        pass