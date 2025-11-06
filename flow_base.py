from abc import abstractclassmethod, abstractmethod, ABC
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


def sliding_window_ar1_helper(t: torch.Tensor, win_size: int, cost_fn):
    """Helper function to take a sliding window over a tensor (could be uv, could be b, etc) and
    calculate the weighted-average of a robust cost function of differences between a value and
    its neighborhood.

    Input 't' must have shape (h, w, ...) and all '...' dimensions will be flattened into a vector
    """
    t = t.flatten(start_dim=2)
    h, w, d = t.shape

    # 'Unfold' with sliding windows --> get a (win_size, win_size) region around each pixel, shaped
    # as a (h, w, d, win_size, win_size) tensor
    sliding_windows = t.unfold(0, win_size, 1).unfold(1, win_size, 1)
    assert sliding_windows.shape == (
        h - win_size + 1,
        w - win_size + 1,
        d,
        win_size,
        win_size,
    )

    # Get the 'center' value for each window by simply cropping out edges
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


class Metrics:
    @staticmethod
    def epe(true_flo, pred_flo): ...

    @staticmethod
    def angle_err(true_flo, pred_flo): ...


class BaseFlow(ABC):
    def __init__(self, shape_hw: tuple[int, int], device: torch.device = "cpu"):
        self.xs = ...
        self.ys = ...

        # params must be instantiated by a base class and will be the thing that requires grad
        self.params: torch.Tensor = self.init_params(shape_hw, device)
        self.device = device

    @classmethod
    @abstractmethod
    def init_params(
        cls, shape_hw, device, initial_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Initialize the parameters for this flow, using initial_flow if provided."""

    @abstractmethod
    def named_params(self) -> dict[str, torch.Tensor]:
        """Return named portions of parameters tensor."""

    def warp_image(self, im1):
        new_x = self.xs + self.uv[..., 0]
        new_y = self.ys + self.uv[..., 1]

        # TODO do the grid_sample stuff

        return F.grid_sample(im1, ...)

    @abstractmethod
    def ar0_terms(self, cost_fn) -> dict[str, torch.Tensor]:
        """Calculate magnitude penalties on parameters using the given [robust] cost function.
        Returns a collection of named penalties, each shape (h, w), representing the magnitude
        penalty applied to each parameter at each pixel."""

    @abstractmethod
    def ar1_terms(self, win_size: int, cost_fn) -> dict[str, torch.Tensor]:
        """Calculate local smoothness penalties (analogous to AR1). Returns a collection of named
        penalties, each will be shape (h, w) representing the average of cost_fn applied to local
        *differences* in the parameters
        """

    def serialize(self):
        return {"xs": self.xs, "ys": self.ys, "uv": self.uv, "params": ...}

    def deserialize(self, data):
        self.xs = data["xs"]
        self.ys = data["ys"]
        self.uv = data["uv"]
        self.flow = data["params"]


class Flow2p(BaseFlow):
    @classmethod
    def init_params(
        cls, shape_hw, device, initial_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Initialize the parameters for this flow, using initial_flow if provided."""
        if initial_flow is None:
            return torch.zeros(*shape_hw, 2, device=device)
        else:
            assert initial_flow.shape == (*shape_hw, 2)
            return initial_flow.clone().to(device)

    def named_params(self) -> dict[str, torch.Tensor]:
        """Return named portions of parameters tensor."""
        return {"uv": self.uv}

    @property
    def uv(self):
        return self.params

    def ar0_terms(self, cost_fn):
        return {
            "uv": cost_fn(torch.linalg.norm(self.uv.flatten(start_dim=2), dim=-1)),
        }

    def ar1_terms(self, win_size: int, cost_fn):
        return {
            "uv": sliding_window_ar1_helper(self.uv, win_size, cost_fn),
        }


class Flow6p(BaseFlow):
    @classmethod
    def init_params(
        cls, shape_hw, device, initial_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Initialize the parameters for this 6p flow, using initial_flow if provided."""
        flo = torch.zeros(*shape_hw, 6, device=device)
        if initial_flow is not None:
            assert initial_flow.shape == (*shape_hw, 2)
            flo[..., [2, 5]] = initial_flow
        return flo

    def named_params(self) -> dict[str, torch.Tensor]:
        """Return named portions of parameters tensor."""
        return {"uv": self.uv, "b": self.affine_b}

    @property
    def uv(self):
        return self.params[..., [2, 5]]

    @property
    def affine_b(self):
        return self.params[..., [0, 1, 3, 4]].unflatten(-1, (2, 2))

    def _affine_predicted_flow_in_windows(self, win_size: int):
        # TODO - verify!
        h, w, _ = self.uv.shape
        # dx and dy have shape (win_size, win_size)
        dy, dx = torch.meshgrid(
            torch.arange(win_size, dtype=torch.float) - win_size // 2,
            torch.arange(win_size, dtype=torch.float) - win_size // 2,
            indexing="ij",
        )

        # delta has shape (win_size, win_size, 2)
        delta = torch.stack([dx, dy], dim=-1)

        # calculate the B @ delta terms everywhere all at once
        b_times_delta = torch.einsum("rcij,abj->rciab", self.affine_b, delta)

        affine_flow_in_sliding_windows = self.uv.unsqueeze(-1).unsqueeze(-1) + b_times_delta
        left, right = win_size//2, w - win_size//2
        top, bottom = win_size//2, h - win_size//2
        return affine_flow_in_sliding_windows[top:bottom, left:right, :, :, :]

    def ar0_terms(self, cost_fn):
        return {
            "uv": cost_fn(torch.linalg.norm(self.uv.flatten(start_dim=2), dim=-1)),
            "b": cost_fn(torch.linalg.norm(self.affine_b.flatten(start_dim=2), dim=-1)),
        }

    def ar1_terms(self, win_size: int, cost_fn):
        # Compute basic 'windowed AR1' for UV and AffineB separately
        basic_uv_term = sliding_window_ar1_helper(self.uv, win_size, cost_fn)
        basic_b_term = sliding_window_ar1_helper(self.affine_b, win_size, cost_fn)

        # *Also* compute the fancier term which uses the affine transform to predict small local
        # differences in uv terms. Compare with sliding_window_ar1_helper above...
        uv_sliding_window_actual = self.uv.unfold(0, win_size, 1).unfold(1, win_size, 1)
        uv_sliding_window_predictions = self._affine_predicted_flow_in_windows(win_size)
        norm_diff = torch.linalg.norm(
            uv_sliding_window_actual - uv_sliding_window_predictions, dim=2
        )
        costs = cost_fn(norm_diff)
        avg_cost_per_pixel = gauss_avg(gauss_avg(costs, dim=-1), dim=-1)
        avg_cost = torch.mean(avg_cost_per_pixel)

        # We now have 3 AR1 terms; return all of them so we can decide later which turn out to be
        # useful
        return {
            "uv": basic_uv_term,
            "b": basic_b_term,
            "affine_uv": avg_cost,
        }


if __name__ == "__main__":
    def cost_fn(x):
        # TODO - charbonnier
        return x**2

    # DEBUG/TEST
    uv = torch.randn(100, 200, 2)
    print(sliding_window_ar1_helper(uv, 3, cost_fn))

    f2p = Flow2p(shape_hw=(10, 20))
    f2p.ar0_terms(cost_fn)
    f2p.ar1_terms(3, cost_fn)

    f6p = Flow6p(shape_hw=(10, 20))
    f6p.ar0_terms(cost_fn)
    f6p.ar1_terms(3, cost_fn)