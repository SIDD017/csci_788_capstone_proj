from abc import abstractmethod, ABC
from typing import Optional
import torch
import torch.nn.functional as F

def gauss_avg(t: torch.Tensor, dim: int = -1):
    # TODO - verify this too
    d = t.shape[dim]
    x = torch.linspace(-2, 2, d, device=t.device)
    wt = torch.exp(-x * x / 2)
    wt = wt / torch.sum(wt)
    new_shape = [1] * t.ndim
    new_shape[dim] = d
    return torch.sum(t * wt.view(*new_shape), dim)


def sliding_windowa_ar1_helper(t: torch.Tensor, win_size: int, cost_fn):
    """Helper function to take a sliding window over a tensor (could be uv, could be b, etc) and
    calculate the weighted average of a robust cost function of differences between a value and
    itd neighborhood
    
    Input 't' must have shape (h, w, ...) and all '...' dimensions will be flattened into a vector
    """
    t = t.flatten(start_dim=2)
    h, w, d = t.shape

    # Unfold with sliding windoes --> get a (win_size, win_size) region around each pixel, shaped
    # as a (h, w, d, win_size, win_size) tensor
    sliding_windows = t.unfold(0, win_size, 1).unfold(1, win_size, 1)
    assert sliding_windows.shape == (
        h - win_size + 1,
        w - win_size + 1,
        d,
        win_size,
        win_size
    )

    # Get the 'center' value for each windoe by simply cropping out edges
    left, right = win_size // 2, w - win_size // 2
    top, bottom = win_size // 2, h - win_size // 2
    central_values = t[top:bottom, left:right]

    # By adding singleton dimensions to the central values, they are broadcast to each window
    # in the diff:
    diff = sliding_windows - central_values.unsqueeze(-1).unsqueeze(-1)
    assert diff.shape == sliding_windows.shape

    # Compute the Euclidean norm along the 'd' dimension; now we have a scalar 'difference' for each
    # pixel relative to other pixels in its neighborhood
    norm_diff = torch.linalg.norm(diff, dim=2)
    assert norm_diff.shape == (h - win_size + 1, w - win_size + 1, win_size, win_size)

    # Compute a [robust] cost for every diff
    costs = cost_fn(norm_diff)

    # Finally, take a Gaussian-weighted average within the windows and then a mean across all pixels
    avg_cost_per_pixel = gauss_avg(gauss_avg(costs, dim=-1), dim=-1)
    assert avg_cost_per_pixel.shape == (h - win_size + 1, w - win_size + 1)

    return torch.mean(avg_cost_per_pixel)
                          

class BaseFlow(ABC):
    def __init__(self, 
                 shape_hw: tuple[int, int], 
                 device: torch.device = "cpu",
                 init_flow: Optional[torch.Tensor] = None):
        self.ys, self.xs = torch.meshgrid(torch.arange(shape_hw[0], dtype=torch.float32),
                                 torch.arange(shape_hw[1], dtype=torch.float32),
                                 indexing="ij")
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
        self.device = device
        self.params: torch.Tensor = self.init_params(shape_hw, device, init_flow).requires_grad_(True)


    @abstractmethod
    def init_params(self, 
                    shape_hw: tuple[int, int], 
                    device: torch.device,
                    init_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Initialize the parameters for this flow, using init_flow if provided, else zeros."""

    @abstractmethod
    def named_params(self) -> dict[str, torch.Tensor]:
        """Return named portion of parameters tensor"""

    @abstractmethod
    def ar0_terms(self, cost_fn) -> dict[str, torch.Tensor]:
        """Calculate magnitude penalties on parameters using the given [robust] cost function.
        Returns a collection of named penalties, each shape (h, w), representing the magnitude
        penalty applied to each parameter at each pixel."""

    @abstractmethod
    def ar1_terms(self, win_size: int, cost_fn) -> dict[str, torch.Tensor]:
        """Calculate local smothness penalties (analogous to AR1). Returns a collection of named
        penalties, each will be shape(h,w) representing the average of cost_fn applied to local
        *differences* in the parameters
        """

    def warp_image(self, im1: torch.tensor) -> torch.Tensor:
        #TODO: Abstract this? warped coords could be different for uv and affine cases
        warped_coords_x = self.xs + self.uv[..., 0]
        warped_coords_y = self.ys + self.uv[..., 1]

        if im1 is None:
            raise ValueError("Image to warp is None")
        B, C, H, W = im1.shape
        gx = (warped_coords_x / (W - 1)) * 2 - 1
        gy = (warped_coords_y / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], -1).unsqueeze(0)
        return F.grid_sample(im1, 
                             grid, 
                             align_corners=True, 
                             mode="bilinear", 
                             padding_mode="border")


    def serialize(self):
        return {"xs": self.xs, "ys": self.ys, "params":self.params}
    
    def deserialize(self, data):
        self.xs  = data["xs"]
        self.ys = data["ys"]
        self.uv = data["uv"]
        self.params = data["params"]



class Flow2p(BaseFlow):
    def init_params(self,
                    shape_hw: tuple[int, int],
                    device: torch.device = "cpu",
                    init_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        if init_flow is None:
            return torch.zeros(*shape_hw, 2, device=device)
        else:
            assert init_flow.shape == (*shape_hw, 2)
            return init_flow.clone().to(device)
        
    def named_params(self) -> dict[str, torch.Tensor]:
        """Return named portions of parameters tensor"""
        return {"uv": self.uv}

    @property
    def uv(self) -> torch.Tensor:
        return self.params
    
    def ar0_terms(self, cost_fn) -> dict[str, torch.Tensor]:
        dy = torch.linalg.norm(self.params[1:, :, :] - self.params[:-1, :, :], dim=-1)   # (H-1) x W x 2
        dx = torch.linalg.norm(self.params[:, 1:, :] - self.params[:, :-1, :], dim=-1)   # H x (W-1) x 2
        return {"uv": cost_fn(dy, 1e-3).mean() + cost_fn(dx, 1e-3).mean()}
    
    def ar1_terms(self, win_size: int, cost_fn):
        return {
            "uv": sliding_windowa_ar1_helper(self.uv, win_size, cost_fn)
        }
    


class Flow6p(BaseFlow):

    def named_params(self) -> dict[str, torch.Tensor]:
        """Return named portions of parameters tensor"""
        return {"A": self.A, "uv": self.uv}

    @property
    def uv(self) -> torch.Tensor:
        return self.params[..., [2,5]]
    
    @property
    def A(self) -> torch.Tensor:
        # Unflattening last dimension into 2x2 matrices
        return self.params[..., [0,1,3,4]].unflatten(-1, (2, 2))
    


class Metrics:
    @staticmethod
    def epe(true_flo: torch.Tensor, pred_flo: torch.Tensor) -> float:
        """Compute the end point error"""
        return torch.linalg.norm(true_flo - pred_flo, dim=-1).mean().item()
    
    @staticmethod
    def angular_error(true_flo: torch.Tensor, pred_flo: torch.Tensor) -> float:
        """Compute the angular error"""
        u_true, v_true = true_flo[...,0], true_flo[...,1]
        u_pred, v_pred = pred_flo[...,0], pred_flo[...,1]
        num = u_true * u_pred + v_true * v_pred
        eps = 1e-8
        denom = torch.sqrt(u_pred * u_pred + v_pred * v_pred + eps) \
            * torch.sqrt(u_true * u_true + v_true * v_true + eps)
        return torch.acos(torch.clamp(num / denom, -1.0, 1.0)).mean().item()