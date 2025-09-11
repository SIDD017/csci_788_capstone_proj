import numpy as np
import torch
import torch.nn.functional as F

# def _to_t(img_np):
#     return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)

def _sobel_mag(t):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    # Adding small epsilon here to avoid dividing by zero in edge weighting
    return torch.sqrt(gx*gx + gy*gy)

def _warp_with_flow(I, flow):
    """
    I: 1x1xHxW
    flow: HxWx2 (u,v) in pixels
    returns I warped by reverse mapping using grid_sample
    """
    B, C, H, W = I.shape
    dev, dt = I.device, I.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    xw = xs + flow[...,0]
    yw = ys + flow[...,1]
    gx = (xw / (W - 1)) * 2 - 1
    gy = (yw / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(I, grid, align_corners=True, mode="bilinear", padding_mode="border")

def refine_dense_flow(
    image1_np,            # HxW float32 in [0,1]
    image2_np,            # HxW float32 in [0,1]
    init_flow_uv,         # HxWx2 float32 from your LK init
    steps=300,
    lr=1e-1,
    edge_beta=20.0,
    eps=1e-3,
    lambda_smooth=0.1,    # try 0.05 to 0.2
    device="cpu",
):
    """
    Returns:
      flow_opt: HxWx2 numpy
      I2_warp:  HxW numpy
    """
    assert image1_np.ndim == 2 and image2_np.ndim == 2
    H, W = image1_np.shape
    I1 = torch.from_numpy(image1_np).unsqueeze(0).unsqueeze(0).to(device)
    I2 = torch.from_numpy(image2_np).unsqueeze(0).unsqueeze(0).to(device)
    # I1 = _to_t(image1_np).to(device)
    # I2 = _to_t(image2_np).to(device)

    # edge downweighting on I1
    w_edge = 1.0 / (1.0 + edge_beta * _sobel_mag(I1))      # 1x1xHxW
    w_edge = w_edge.detach()

    # init flow from LK
    init = np.nan_to_num(init_flow_uv.astype(np.float32), nan=0.0)
    flow = torch.from_numpy(init).to(device)                # HxWx2
    flow = flow.clone().requires_grad_(True)

    opt = torch.optim.Adam([flow], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        I2w = _warp_with_flow(I2, flow)                    # 1x1xHxW
        resid = (I2w - I1) * w_edge                        # edge-weighted residual
        data = charbonnier_loss(resid, eps=eps).mean()

        # smoothness on flow with Charbonnier TV (forward differences)
        # flow: H x W x 2  (axis 0 = y/rows, axis 1 = x/cols)
        dy = flow[1:, :, :] - flow[:-1, :, :]   # (H-1) x W x 2
        dx = flow[:, 1:, :] - flow[:, :-1, :]   # H x (W-1) x 2

        tv = charbonnier_loss(dy, 1e-3).mean() + charbonnier_loss(dx, 1e-3).mean()

        loss = data + lambda_smooth * tv
        loss.backward()
        opt.step()

    with torch.no_grad():
        I2w = _warp_with_flow(I2, flow)
        return flow.detach().cpu().numpy(), I2w.squeeze().cpu().numpy()



# # patch_affine_refine.py
# import numpy as np
# import torch
# import torch.nn.functional as F

# # ---------------- utils ----------------
# def _to_t(img_np):  # HxW float32 [0,1] -> 1x1xHxW
#     return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

# def _sobel_mag(t):
#     kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
#     ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=t.dtype, device=t.device).view(1,1,3,3)/8
#     gx = F.conv2d(t, kx, padding=1); gy = F.conv2d(t, ky, padding=1)
#     return torch.sqrt(gx*gx + gy*gy + 1e-12)

# def _charb(x, eps=1e-3):
#     return torch.sqrt(x*x + eps*eps)

# def _affine_warp(I, theta):
#     """
#     I: 1x1xHxW, theta: 2x3 tensor
#     returns (Iw: 1x1xHxW, flow: HxWx2)
#     """
#     B, C, H, W = I.shape
#     dev, dt = I.device, I.dtype
#     ys, xs = torch.meshgrid(torch.arange(H, device=dev),
#                             torch.arange(W, device=dev), indexing="ij")
#     ones = torch.ones_like(xs, dtype=dt)
#     X = torch.stack([xs, ys, ones], -1).view(-1,3).T      # 3x(HW)
#     A = theta[:, :2]; t = theta[:, 2:].view(2,1)
#     Y = (A @ X[:2,:]) + t
#     xw, yw = Y[0].view(H,W), Y[1].view(H,W)
#     gx = (xw/(W-1))*2 - 1; gy = (yw/(H-1))*2 - 1
#     grid = torch.stack([gx,gy], -1).unsqueeze(0)
#     Iw = F.grid_sample(I, grid, align_corners=True, mode="bilinear", padding_mode="border")
#     flow = torch.stack([xw - xs, yw - ys], -1)
#     return Iw, flow

# # -------------- main API --------------
# def refine_patch_affine(
#     image1_np, image2_np, init_flow_uv,
#     patch=32, stride=None,
#     steps=300, lr=5e-2,
#     edge_beta=20.0, eps=1e-3,
#     lambda_smooth=1e-3,
#     device="cpu",
# ):
#     """
#     Local 6-param affine per patch on a grid.

#     image1_np, image2_np: HxW float32 in [0,1]
#     init_flow_uv: HxWx2 float32 (your 2-param LK init)
#     patch: patch size (square)
#     stride: step between patch centers (default = patch for non-overlap; use <patch for overlap)
#     returns:
#       thetas: P x 2 x 3 numpy (P = number of patches)
#       flow: HxWx2 numpy (blended piecewise flow)
#       I2w: HxW numpy (I2 warped by blended locals)
#       layout: list of (y0,y1,x0,x1) patch boxes, same order as thetas
#     """
#     H, W = image1_np.shape
#     if stride is None: stride = patch

#     I1 = _to_t(image1_np).to(device)
#     I2 = _to_t(image2_np).to(device)

#     # downweight edges: w = 1 / (1 + beta * |∇I1|)
#     w_edge = 1.0 / (1.0 + edge_beta * _sobel_mag(I1))     # 1x1xHxW
#     w_edge = w_edge.detach()

#     # grid of patches
#     boxes = []
#     for y in range(0, H, stride):
#         for x in range(0, W, stride):
#             y0 = y
#             x0 = x
#             y1 = min(y0 + patch, H)
#             x1 = min(x0 + patch, W)
#             boxes.append((y0, y1, x0, x1))
#     P = len(boxes)

#     # init per-patch theta: identity + local median(u,v)
#     uv = np.nan_to_num(init_flow_uv, nan=0.0)
#     thetas = []
#     for (y0,y1,x0,x1) in boxes:
#         sub = uv[y0:y1, x0:x1, :]
#         if sub.size > 0:
#             tx = float(np.median(sub[...,0]))
#             ty = float(np.median(sub[...,1]))
#         else:
#             tx, ty = 0.0, 0.0
#         thetas.append([[1.0, 0.0, tx],[0.0, 1.0, ty]])
#     theta = torch.tensor(thetas, dtype=I1.dtype, device=device, requires_grad=True)  # P x 2 x 3

#     opt = torch.optim.Adam([theta], lr=lr)

#     # make soft masks for blending when stride < patch (raised cosine feather)
#     with torch.no_grad():
#         blend = torch.zeros((P, 1, 1, H, W), dtype=I1.dtype, device=device)  # P x 1 x 1 x H x W
#         for i, (y0, y1, x0, x1) in enumerate(boxes):
#             h = y1 - y0
#             w = x1 - x0

#             wy = torch.ones(h, dtype=I1.dtype, device=device)
#             wx = torch.ones(w, dtype=I1.dtype, device=device)

#             if stride is not None and stride < patch:
#                 # nominal feather half widths
#                 sy_nom = max(1, (patch - stride) // 2)
#                 sx_nom = max(1, (patch - stride) // 2)
#                 # cap to available half-size of this (possibly truncated) patch
#                 sy = int(min(sy_nom, h // 2))
#                 sx = int(min(sx_nom, w // 2))

#                 if sy > 0:
#                     ramp_y = 0.5 - 0.5 * torch.cos(
#                         torch.linspace(0.0, np.pi, steps=sy, device=device, dtype=I1.dtype)
#                     )
#                     wy[:sy] = ramp_y
#                     wy[-sy:] = ramp_y.flip(0)

#                 if sx > 0:
#                     ramp_x = 0.5 - 0.5 * torch.cos(
#                         torch.linspace(0.0, np.pi, steps=sx, device=device, dtype=I1.dtype)
#                     )
#                     wx[:sx] = ramp_x
#                     wx[-sx:] = ramp_x.flip(0)

#             # outer product to get 2D window
#             mask = torch.outer(wy, wx)[None, None, :, :]  # 1 x 1 x h x w
#             blend[i, :, :, y0:y1, x0:x1] = mask

#         denom = blend.sum(dim=0).clamp_min(1e-6)   # 1 x 1 x H x W
#         norm_blend = blend / denom                 # P x 1 x 1 x H x W

#     # training loop
#     for _ in range(steps):
#         opt.zero_grad()
#         total = 0.0
#         # data term: sum over patches of Charbonnier on residual inside patch
#         for i,(y0,y1,x0,x1) in enumerate(boxes):
#             I2w_i, _ = _affine_warp(I2, theta[i])
#             resid_i = (I2w_i - I1) * w_edge * norm_blend[i]  # only contributes inside patch
#             # normalize by patch support so losses are comparable
#             norm = norm_blend[i].sum() + 1e-6
#             total = total + _charb(resid_i, eps=eps).sum() / norm

#         # smoothness between neighboring patch parameters (keeps seams small)
#         if lambda_smooth > 0 and P > 1:
#             # 4-neighborhood on the implicit grid
#             sm = 0.0
#             cols = max(1, (W + stride - 1) // stride)
#             for i in range(P):
#                 r = i // cols; c = i % cols
#                 for dr,dc in [(1,0),(0,1)]:
#                     rr, cc = r+dr, c+dc
#                     if 0 <= rr and 0 <= cc and rr*cols+cc < P:
#                         j = rr*cols + cc
#                         sm = sm + (theta[i] - theta[j]).pow(2).mean()
#             total = total + lambda_smooth * sm

#         total.backward()
#         opt.step()

#     # compose final warped image and flow via normalized blending
#     with torch.no_grad():
#         I2w_acc = torch.zeros_like(I1)
#         flow_acc = torch.zeros((H, W, 2), dtype=I1.dtype, device=device)
#         for i in range(P):
#             I2w_i, flow_i = _affine_warp(I2, theta[i])
#             w_i = norm_blend[i]
#             I2w_acc += I2w_i * w_i
#             flow_acc[...,0] += flow_i[...,0] * w_i.squeeze(0).squeeze(0)
#             flow_acc[...,1] += flow_i[...,1] * w_i.squeeze(0).squeeze(0)

#         return (theta.detach().cpu().numpy(),
#                 flow_acc.cpu().numpy(),
#                 I2w_acc.squeeze().cpu().numpy(),
#                 boxes)
