from abc import ABC, abstractmethod
import torch.nn.functional as F
import matplotlib.pyplot as plt

from flow_init import calculate_initial_flow
from refine import charbonnier_loss
from utils import *

# Abstract base class for optical flow (2 Param lucas kanade and 6/8 param affine)
class Flow(ABC):
    def __init__(self, image1_path, image2_path, gtimage_path, init_params):
        image1 = cv.imread(str(image1_path), cv.IMREAD_GRAYSCALE)
        image2 = cv.imread(str(image2_path), cv.IMREAD_GRAYSCALE)
        gt_flow = read_flo_file(gtimage_path)
        # Calculate the initial flow and setup everything to be torch tensors
        # on device, ready for refinement
        self.init_flow = calculate_initial_flow(image1, image2, init_params)
        self.image1 = np_im_to_torch(uint8_to_float32(image1)).to(init_params.get("device", "cpu"))
        self.image2 = np_im_to_torch(uint8_to_float32(image2)).to(init_params.get("device", "cpu"))
        self.gt_flow = torch.from_numpy(gt_flow.astype(np.float32)).to(init_params.get("device", "cpu"))
        self.is_flow_refined = False
        self.log_metrics = {"data_loss_log": [],
                            "smoothness_loss_log": [],
                            "loss_log": [],
                            "epe_log": [],
                            "angular_log": []}
    
    def epe_error(self):
        disp = self.pred_flow()
        epe = torch.linalg.norm(disp - self.gt_flow, dim=-1)
        return epe.mean().item()
    

    def angular_error(self):
        disp = self.pred_flow()
        u, v   = disp[..., 0], disp[..., 1]
        ug, vg = self.gt_flow[..., 0], self.gt_flow[..., 1]

        num = u * ug + v * vg
        epsilon = 1e-8
        pred_mag = torch.sqrt(u * u + v * v + epsilon)
        gt_mag   = torch.sqrt(ug * ug + vg * vg + epsilon)
        den = pred_mag * gt_mag
        ang = torch.acos(torch.clamp(num / den, -1.0, 1.0))
        return ang.mean().item()
    
    # Helper funtion to debug and visualize flow
    def visualize_flow(self):
        cv.imshow("Image1", convert_torch_to_cv(self.image1))
        cv.imshow("Image2", convert_torch_to_cv(self.image2))
        disp = self.pred_flow()
        disp_np = disp.detach().cpu().numpy()
        disp_hsv = visualize_flow_hsv(disp_np)
        cv.imshow("Dense refined flow", disp_hsv)
        disp_init = visualize_flow_hsv(self.init_flow)
        gt_disp = visualize_gt_flow_hsv(self.gt_flow.cpu().numpy())
        cv.imshow("Optical flow (initial)", disp_init)
        cv.imshow("Optical flow (ground truth)", gt_disp)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Warp an image using the flow field
    @abstractmethod
    def warped_coords(self, xs, ys):
        pass

    @abstractmethod
    def pred_flow(self):
        pass

    @abstractmethod
    def smoothness_tv(self):
        pass

    # Visualize the flow field
    @abstractmethod
    def visualize_params(self):
        pass


class CustomLucasKanadeFlow(Flow):
    def __init__(self, image1, image2, gt_flow, init_params):
        super().__init__(image1, image2, gt_flow, init_params)
        self.log_metrics["uv_tv_log"] = []
        temp = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        temp = torch.from_numpy(temp).to(init_params.get("device", "cpu"))
        self.params = temp.clone().requires_grad_(True)
    
    def warped_coords(self, xs, ys):
        xw = xs + self.params[...,0]
        yw = ys + self.params[...,1]
        return xw, yw

    def pred_flow(self):
        return self.params
    
    def smoothness_tv(self):
        dy = torch.linalg.norm(self.params[1:, :, :] - self.params[:-1, :, :], dim=-1)   # (H-1) x W x 2
        dx = torch.linalg.norm(self.params[:, 1:, :] - self.params[:, :-1, :], dim=-1)   # H x (W-1) x 2
        return {"uv": charbonnier_loss(dy, 1e-3).mean() + charbonnier_loss(dx, 1e-3).mean()}
    
    def visualize_params(self):
        pass


class AffineFlow(Flow):
    def __init__(self, image1, image2, gt_flow, init_params):
        super().__init__(image1, image2, gt_flow, init_params)
        self.log_metrics["A_tv_log"] = []
        self.log_metrics["uv_tv_log"] = []
        self.log_metrics["A_det_tv_log"] = []
        # Initialize the affine parameters to identity + translation from init flow
        temp = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        H, W, _ = temp.shape
        # create a meshgrid for pixel coordinates (note: x = column index, y = row index)
        ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                            np.arange(W, dtype=np.float32),
                            indexing="ij")
        self.params = np.zeros((H, W, 6), dtype=np.float32)
        self.params[...,0] = 0.0
        self.params[...,1] = 0.0
        self.params[...,2] = temp[...,0]
        self.params[...,3] = 0.0
        self.params[...,4] = 0.0
        self.params[...,5] = temp[...,1]
        self.params = torch.from_numpy(self.params).to(init_params.get("device", "cpu"))
        self.params = self.params.clone().requires_grad_(True)
        self.xs_t = torch.from_numpy(xs).to(init_params.get("device", "cpu"))
        self.ys_t = torch.from_numpy(ys).to(init_params.get("device", "cpu"))
        
    def warped_coords(self, xs, ys):
        x_new = xs + self.params[...,2]
        y_new = ys + self.params[...,5]
        return x_new, y_new
    
    def pred_flow(self):
        x_new, y_new = self.warped_coords(self.xs_t, self.ys_t)
        warped_coords = torch.stack([x_new, y_new], dim=-1)
        grid = torch.stack([self.xs_t, self.ys_t], dim=-1)
        return warped_coords - grid
        
    def smoothness_tv(self):
        # uv and A
        flow = self.params[..., [2, 5]]
        A = torch.stack([self.params[..., 0:2], self.params[..., 3:5]], dim=-2)  # [H, W, 2, 2]

        # NOTE: downweighting flow smoothness at edges (to avoid blurriness)
        I = self.image1.squeeze(0).squeeze(0)
        I_x = I[:, 1:] - I[:, :-1]  # [H, W-1]
        I_y = I[1:, :] - I[:-1, :]  # [H-1, W]
        edge_weight_x = torch.exp(-torch.abs(I_x) / 0.1)  # [H, W-1]
        edge_weight_y = torch.exp(-torch.abs(I_y) / 0.1)  # [H-1, W]
        
        # expected_flow[i+1,j] = flow[i,j] + A[i,j] @ [1, 0]
        expected_flow_right = flow[:, :-1] + A[:, :-1, :, 0]
        flow_diff_x = flow[:, 1:] - expected_flow_right
        # expected_flow[i,j+1] = flow[i,j] + A[i,j] @ [0, 1]
        expected_flow_down = flow[:-1, :] + A[:-1, :, :, 1]
        flow_diff_y = flow[1:, :] - expected_flow_down
        uv_tv = (edge_weight_x * charbonnier_loss(torch.linalg.norm(flow_diff_x, dim=-1), 1e-3)).mean() + \
                (edge_weight_y * charbonnier_loss(torch.linalg.norm(flow_diff_y, dim=-1), 1e-3)).mean()
        
        # A smoothness with edge weights
        A_diff_x = A[:, 1:] - A[:, :-1]  # [H, W-1, 2, 2]
        A_diff_y = A[1:, :] - A[:-1, :]  # [H-1, W, 2, 2]
        # Flatten last two dims for norm computation
        A_norm_x = torch.linalg.norm(A_diff_x.reshape(A_diff_x.shape[0], A_diff_x.shape[1], 4), dim=-1)  # [H, W-1]
        A_norm_y = torch.linalg.norm(A_diff_y.reshape(A_diff_y.shape[0], A_diff_y.shape[1], 4), dim=-1)  # [H-1, W]
        A_tv = (edge_weight_x * charbonnier_loss(A_norm_x, 1e-3)).mean() + \
            (edge_weight_y * charbonnier_loss(A_norm_y, 1e-3)).mean()
        
        # frobenius norm of A - I 
        # I = torch.eye(2, device=A.device, dtype=A.dtype).view(1,1,2,2)
        # A_diff = A - I
        # A_mag = charbonnier_loss(torch.linalg.norm(A_diff.reshape(-1, 4), dim=-1), 1e-3).mean()
        A_frob_norm = torch.linalg.norm(A.reshape(-1, 4), dim=-1)  # [H*W]
        A_det = charbonnier_loss(A_frob_norm, 1e-3).mean()

        return {"A": A_tv, "uv": uv_tv, "A_det": A_det}
    

    def visualize_params(self):
        # Make small reference patch (letter R)
        ref_np = np.zeros((16, 16), dtype=np.uint8)
        cv.putText(ref_np, "R", (1, 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv.LINE_AA)
        ref = torch.tensor(ref_np, dtype=torch.float32, device=self.params.device) / 255.0  # [H,W] float
        ref = ref.unsqueeze(0).unsqueeze(0)  # [N=1, C=1, H, W]
        ref_h, ref_w = ref.shape[2:]

        param_grid = self.params.detach().clone()  # [H, W, 6]
        H, W = param_grid.shape[:2]
        
        # Extract A matrix components (for visualization of local linear model)
        # A = [[a11, a12], [a21, a22]] from params[..., [0,1,3,4]]
        A = torch.stack([param_grid[..., 0:2], param_grid[..., 3:5]], dim=-2)  # [H, W, 2, 2]
        
        # For visualizing A, we'll apply it to the reference patch
        # Create 2x3 affine matrices: [A | 0] (no translation for this viz)
        M = torch.zeros(H * W, 2, 3, device=self.params.device)
        M[:, :, :2] = A.reshape(-1, 2, 2)  # Set the A part
        M[:, :, 2] = 0  # No translation
        
        # Add identity to make it I + A (so zero A = identity transform)
        M[:, 0, 0] += 1.0  # Add identity to top-left
        M[:, 1, 1] += 1.0  # Add identity to bottom-right

        # Build normalized sampling grids for each affine transform
        grids = F.affine_grid(M, size=(H*W, 1, ref_h, ref_w), align_corners=False)  # [H*W, ref_h, ref_w, 2]

        # Apply all warps in parallel
        warped = F.grid_sample(ref.expand(H*W, -1, -1, -1), grids, align_corners=False)  # [H*W, 1, ref_h, ref_w]

        # Tile into big canvas
        warped = warped.squeeze(1)  # [H*W, ref_h, ref_w]
        canvas = warped.view(H, W, ref_h, ref_w).permute(0, 2, 1, 3).reshape(H*ref_h, W*ref_w)

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(canvas.detach().cpu(), cmap="gray")
        plt.title("A Matrix Visualization (Local Linear Model)\nEach R shows (I+A) transformation")
        plt.axis("off")
        
        # Visualize translation parameters (indices 2 and 5)
        translation = self.params[..., [2, 5]].detach().cpu().numpy()  # shape: [H, W, 2]
        translation_hsv = visualize_flow_hsv(translation)
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv.cvtColor(translation_hsv, cv.COLOR_BGR2RGB))
        plt.title("Translation (u, v) - HSV Flow Visualization")
        plt.axis("off")
        
        # Visualize Frobenius norm of A (magnitude of local variation)
        A_frob = torch.linalg.norm(A.reshape(H, W, 4), dim=-1).detach().cpu().numpy()  # [H, W]
        
        plt.subplot(1, 3, 3)
        im = plt.imshow(A_frob, cmap="hot")
        plt.colorbar(im)
        plt.title("||A||_F (Magnitude of Local Flow Variation)")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # Optional: Show some statistics
        print(f"Translation stats:")
        print(f"  u (horizontal): mean={translation[...,0].mean():.3f}, std={translation[...,0].std():.3f}")
        print(f"  v (vertical): mean={translation[...,1].mean():.3f}, std={translation[...,1].std():.3f}")
        print(f"A matrix stats:")
        print(f"  ||A||_F: mean={A_frob.mean():.6f}, max={A_frob.max():.6f}")
        print(f"  A components: a11={param_grid[...,0].mean().item():.6f}, "
            f"a12={param_grid[...,1].mean().item():.6f}, "
            f"a21={param_grid[...,3].mean().item():.6f}, "
            f"a22={param_grid[...,4].mean().item():.6f}")




class AffineFlowWithLocalOrigins(Flow):
    def __init__(self, image1, image2, gt_flow, init_params):
        super().__init__(image1, image2, gt_flow, init_params)
        self.log_metrics["A_tv_log"] = []
        self.log_metrics["uv_tv_log"] = []
        self.log_metrics["origin_tv_log"] = []
        # Initialize any additional parameters for affine flow and local origins
        temp = np.nan_to_num(self.init_flow.astype(np.float32), nan=0.0)
        H, W, _ = temp.shape
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


    def warped_coords(self, xs, ys):
        ox = self.params[...,6]
        oy = self.params[...,7]
        localx = xs - ox
        localy = ys - oy
        x_new = ox + self.params[...,0]*localx + self.params[...,1]*localy + self.params[...,2]
        y_new = oy + self.params[...,3]*localx + self.params[...,4]*localy + self.params[...,5]
        return x_new, y_new
    
    def pred_flow(self):
        x_new, y_new = self.warped_coords(self.xs_t, self.ys_t)
        warped_coords = torch.stack([x_new, y_new], dim=-1)
        grid = torch.stack([self.xs_t, self.ys_t], dim=-1)
        return warped_coords - grid

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


    def visualize_params(self):
        # Make small reference patch (letter R)
        ref_np = np.zeros((16, 16), dtype=np.uint8)
        cv.putText(ref_np, "R", (1, 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv.LINE_AA)
        ref = torch.tensor(ref_np, dtype=torch.float32, device="cuda") / 255.0  # [H,W] float
        ref = ref.unsqueeze(0).unsqueeze(0)  # [N=1, C=1, H, W]
        ref_h, ref_w = ref.shape[2:]

        param_grid = self.params[:,:,:6].detach().clone()  # [H, W, 6]
        H, W = param_grid.shape[:2]
        param_grid[...,2] = 0.0
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
