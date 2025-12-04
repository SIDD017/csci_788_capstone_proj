import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import read_flo_file


# ---------- affine GT from dense flow ----------

def fit_affine_patch(flow: np.ndarray, i: int, j: int, r: int) -> np.ndarray:
    """Fit 6-param affine model in a (2r+1)x(2r+1) patch centered at (i, j).

    Model:
        [u_i]   [dx]   [a11 a12] [x - x0]
        [v_i] = [dy] + [a21 a22] [y - y0]

    Returns
    -------
    theta : np.ndarray, shape (6,)
        [dx, dy, a11, a12, a21, a22]
        returns NaNs if fit is not possible.
    """
    H, W, _ = flow.shape

    ys = np.arange(i - r, i + r + 1)
    xs = np.arange(j - r, j + r + 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # flatten
    u = flow[yy, xx, 0].reshape(-1)
    v = flow[yy, xx, 1].reshape(-1)

    # ignore "unknown" flow values
    valid = (np.abs(u) < 1e9) & (np.abs(v) < 1e9)
    if valid.sum() < 3:  # not enough points to fit
        return np.full(6, np.nan, dtype=np.float32)

    dx = (xx - j).reshape(-1)[valid]
    dy = (yy - i).reshape(-1)[valid]
    u = u[valid]
    v = v[valid]

    # design matrix rows for u and v components
    # u_i = dx + a11 * dx + a12 * dy
    # v_i = dy + a21 * dx + a22 * dy
    N = dx.shape[0]
    A = np.zeros((2 * N, 6), dtype=np.float32)
    b = np.zeros((2 * N,), dtype=np.float32)

    # u rows
    A[0::2, 0] = 1.0                 # dx (translation x)
    A[0::2, 2] = dx                  # a11
    A[0::2, 3] = dy                  # a12
    b[0::2] = u

    # v rows
    A[1::2, 1] = 1.0                 # dy (translation y)
    A[1::2, 4] = dx                  # a21
    A[1::2, 5] = dy                  # a22
    b[1::2] = v

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return theta.astype(np.float32)


def compute_affine_gt(flow: np.ndarray, patch_radius: int = 5) -> np.ndarray:
    """Compute dense 6-param affine GT from dense flow.

    Parameters
    ----------
    flow : np.ndarray, shape (H, W, 2)
    patch_radius : int
        radius r so patch size is (2r+1)x(2r+1)

    Returns
    -------
    affine_gt : np.ndarray, shape (H, W, 6)
        channels are [dx, dy, a11, a12, a21, a22]
    """
    H, W, _ = flow.shape
    affine_gt = np.full((H, W, 6), np.nan, dtype=np.float32)

    for i in range(patch_radius, H - patch_radius):
        for j in range(patch_radius, W - patch_radius):
            affine_gt[i, j] = fit_affine_patch(flow, i, j, patch_radius)

    return affine_gt


# ---------- main script ----------

def main():
    parser = argparse.ArgumentParser(
        description="Convert dense .flo flow to 6-param affine ground truth."
    )
    parser.add_argument("flo_path", type=str, help="Path to input .flo file")
    parser.add_argument(
        "--patch_radius", type=int, default=5,
        help="Patch radius r (patch size = (2r+1)x(2r+1))"
    )
    parser.add_argument(
        "--out", type=str, default="affine_gt.npy",
        help="Output .npy filename for affine GT"
    )
    args = parser.parse_args()

    print(f"Reading flow from {args.flo_path} ...")
    flow = read_flo_file(args.flo_path)

    print("Computing 6-parameter affine ground truth ...")
    affine_gt = compute_affine_gt(flow, patch_radius=args.patch_radius)

    print(f"Saving affine GT to {args.out} ...")
    np.save(args.out, affine_gt)

    # ----- display heatmap of affine parameter magnitude -----
    # use only the affine part A (a11, a12, a21, a22)
    A = affine_gt[..., 2:]                         # (H, W, 4)
    mag = np.linalg.norm(A, axis=-1)               # (H, W)

    plt.imshow(mag, cmap="viridis")
    plt.title("Affine params magnitude (|A|)")
    plt.colorbar()
    png_path = args.out.replace('.npy', '_magnitude.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved magnitude heatmap to {png_path}")
    plt.show()
    


if __name__ == "__main__":
    main()
