import cv2 as cv
import numpy as np
import mlflow

import flow
from utils import convert_torch_to_cv, visualize_flow_hsv, visualize_gt_flow_hsv, warp_image_with_flow

def log_artifacts():
    cv.imshow("Image1", convert_torch_to_cv(flow.image1))
    cv.imshow("Image2", convert_torch_to_cv(flow.image2))
    disp = flow.pred_flow()
    disp_np = disp.detach().cpu().numpy()
    disp_hsv = visualize_flow_hsv(disp_np)
    cv.imshow("Dense refined flow", disp_hsv)
    disp_init = visualize_flow_hsv(flow.init_flow)
    gt_disp = visualize_gt_flow_hsv(flow.gt_flow.cpu().numpy())
    cv.imshow("Optical flow (initial)", disp_init)
    cv.imshow("Optical flow (ground truth)", gt_disp)
    cv.waitKey(0)
    cv.destroyAllWindows()


def log_mlflow_images(flow):
    ''' Log images to MLflow '''
    flow_refined = flow.pred_flow().detach().cpu().numpy()
    I2_warp = warp_image_with_flow(flow).squeeze().detach().cpu().numpy()
    # Convert I2_warp to uint8 (assuming it's in [0, 1] range)
    I2_warp_uint8 = (np.clip(I2_warp, 0, 1) * 255).astype(np.uint8)
    result_img = {"I2_warp": I2_warp_uint8, 
                  "image1": convert_torch_to_cv(flow.image1), 
                  "image2": convert_torch_to_cv(flow.image2),
                  "flow": visualize_flow_hsv(flow_refined), 
                  "init_flow": visualize_flow_hsv(flow.init_flow),
                  "gt_flow": visualize_gt_flow_hsv(flow.gt_flow.cpu().numpy())}
    for name, img in result_img.items():
        plot_title = f"{name}.png"
        cv.imwrite(plot_title, img)
        mlflow.log_artifact(plot_title, artifact_path="outputs")