import os
from driver import read_input_ground_truth, read_input_images
from flow_interface import AffineFlowWithLocalOrigins, CustomLucasKanadeFlow
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from refine_utils import convert_torch_to_cv, refine_flow
from utils import visualize_flow_hsv, visualize_gt_flow_hsv

#list of images to test on
image_list = [
    (".\\images\\Venus\\frame10.png", ".\\images\\Venus\\frame11.png", ".\\ground_truth_images\\Venus\\flow10.flo"),
    (".\\images\\Dimetrodon\\frame10.png", ".\\images\\Dimetrodon\\frame11.png", ".\\ground_truth_images\\Dimetrodon\\flow10.flo"),
    (".\\images\\RubberWhale\\frame10.png", ".\\images\\RubberWhale\\frame11.png", ".\\ground_truth_images\\RubberWhale\\flow10.flo"),
    (".\\images\\Hydrangea\\frame10.png", ".\\images\\Hydrangea\\frame11.png", ".\\ground_truth_images\\Hydrangea\\flow10.flo"),
    (".\\images\\Grove2\\frame10.png", ".\\images\\Grove2\\frame11.png", ".\\ground_truth_images\\Grove2\\flow10.flo"),
    (".\\images\\Urban2\\frame10.png", ".\\images\\Urban2\\frame11.png", ".\\ground_truth_images\\Urban2\\flow10.flo"),
    (".\\images\\Urban3\\frame10.png", ".\\images\\Urban3\\frame11.png", ".\\ground_truth_images\\Urban3\\flow10.flo"),
    (".\\images\\Grove3\\frame10.png", ".\\images\\Grove3\\frame11.png", ".\\ground_truth_images\\Grove3\\flow10.flo")
]

init_params = {
        "levels": 5,
        "window_size": 7,
        "alpha": 1e-3,
        "goodness_threshold": 2.0,
        "device": "cuda"
    }
        

def save_epe_plots():
    pass

def init_method_comparison():    
    # Lists to store results
    farneback_epe = []
    assignment_epe = []
    image_names = []
    farneback_angular_error = []
    assignment_angular_error = []
    output_dir = "benchmarks/init_comparson/images/"

    for img1_path, img2_path, flo_path in image_list:
        curr_output_dir = output_dir + img1_path.split("\\")[-2]
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        image_names.append(img1_path.split("\\")[-2])
        image1, image2 = read_input_images(img1_path, img2_path)
        gt_flow = read_input_ground_truth(flo_path)
        
        # Farneback Initialization
        flow_farneback = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=True)
        farneback_epe.append(flow_farneback.epe_error())
        farneback_angular_error.append(flow_farneback.angular_error())

        # custom lucas kanade Initialization
        flow_assignment = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=False)
        assignment_epe.append(flow_assignment.epe_error())
        assignment_angular_error.append(flow_assignment.angular_error())

        # save image1, image2, gt_glow, init_flow to output_dir
        result_img = {"image1": convert_torch_to_cv(flow_farneback.image1), 
                      "image2": convert_torch_to_cv(flow_farneback.image2),
                      "init_flow_farneback": visualize_flow_hsv(flow_farneback.init_flow),
                      "init_flow_assignment": visualize_flow_hsv(flow_assignment.init_flow),
                      "gt_flow": visualize_gt_flow_hsv(flow_farneback.gt_flow.cpu().numpy())}
        for name, img in result_img.items():
            plot_title = os.path.join(curr_output_dir, f"{name}.png")
            cv.imwrite(plot_title, img)


    # epe for hydrangea, dimetredon and rubberwhale separate coz their vvlaues are too high
     # Plot bar chart: each image has two bars: farneback and assignment
    x = np.arange(len(image_names))
    width = 0.35

    # same plots for angular error
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, farneback_angular_error, width, label="Farneback")
    plt.bar(x + width/2, assignment_angular_error, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("Angular Error")
    plt.title("Angular Error Comparison: Farneback vs Assignment Initialization")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/init_comparson/error_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "angular_error.png"))
    plt.show()

    # get indexs of rubberwhale, dimetredon and hydrangea from images_list
    selected_indices = [1, 2, 3]
    # separate charts for selected and remaining images
    selected_image_names = [image_names[i] for i in selected_indices]
    selected_farneback_epe = [farneback_epe[i] for i in selected_indices]
    selected_assignment_epe = [assignment_epe[i] for i in selected_indices]
    
    remaining_image_names = [image_names[i] for i in range(len(image_names)) if i not in selected_indices]
    remaining_farneback_epe = [farneback_epe[i] for i in range(len(image_names)) if i not in selected_indices]
    remaining_assignment_epe = [assignment_epe[i] for i in range(len(image_names)) if i not in selected_indices]

    # Plot selected images bar chart
    x_selected = np.arange(len(selected_image_names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x_selected - width/2, selected_farneback_epe, width, label="Farneback")
    plt.bar(x_selected + width/2, selected_assignment_epe, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("EPE")
    plt.title("EPE Comparison (Selected Images)")
    plt.xticks(x_selected, selected_image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/init_comparson/error_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "large_epe.png"))
    plt.show()
    
    # Plot remaining images bar chart
    x_remaining = np.arange(len(remaining_image_names))
    plt.figure(figsize=(10, 6))
    plt.bar(x_remaining - width/2, remaining_farneback_epe, width, label="Farneback")
    plt.bar(x_remaining + width/2, remaining_assignment_epe, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("EPE")
    plt.title("EPE Comparison (Remaining Images)")
    plt.xticks(x_remaining, remaining_image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/init_comparson/error_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "small_epe.png"))
    plt.show()




def init_methods_plus_2param_comparison():
    refine_params = {
        "steps": 8000,
        "lr": 1e-2,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1, 
    }
    # Lists to store results
    farneback_2param_epe = []
    assignment_2param_epe = []
    image_names = []
    farneback_2param_angular_error = []
    assignment_2param_angular_error = []
    output_dir = "benchmarks/2_params/images/"

    for img1_path, img2_path, flo_path in image_list:
        curr_output_dir = output_dir + img1_path.split("\\")[-2]
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        image_names.append(img1_path.split("\\")[-2])
        image1, image2 = read_input_images(img1_path, img2_path)
        gt_flow = read_input_ground_truth(flo_path)
        
        # Farneback Initialization
        flow_farnedback = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=True)
        flow_refined_farnedback, I2_warp_farneback = refine_flow(flow_farnedback, refine_params)
        farneback_2param_epe.append(flow_farnedback.epe_error())
        farneback_2param_angular_error.append(flow_farnedback.angular_error())

        # custom lucas kanade Initialization
        flow_assignment = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=False)
        flow_refined_assignment, I2_warp_assignment = refine_flow(flow_assignment, refine_params)
        assignment_2param_epe.append(flow_assignment.epe_error())
        assignment_2param_angular_error.append(flow_assignment.angular_error())

        I2_warp_uint8_farneback = (np.clip(I2_warp_farneback, 0, 1) * 255).astype(np.uint8)
        I2_warp_uint8_assignment = (np.clip(I2_warp_assignment, 0, 1) * 255).astype(np.uint8)
        # save image1, image2, gt_glow, init_flow to output_dir
        result_img = {"I2_warp_farneback": I2_warp_uint8_farneback, 
                      "I2_warp_assignment": I2_warp_uint8_assignment,
                      "image1": convert_torch_to_cv(flow_assignment.image1), 
                    "image2": convert_torch_to_cv(flow_assignment.image2),
                    "flow_farneback": visualize_flow_hsv(flow_refined_farnedback),
                    "flow_assignment": visualize_flow_hsv(flow_refined_assignment),
                    "init_flow_farneback": visualize_flow_hsv(flow_farnedback.init_flow),
                    "init_flow_assignment": visualize_flow_hsv(flow_assignment.init_flow),
                    "gt_flow": visualize_gt_flow_hsv(flow_assignment.gt_flow.cpu().numpy())}
        for name, img in result_img.items():
            plot_title = os.path.join(curr_output_dir, f"{name}.png")
            cv.imwrite(plot_title, img)


    # epe for hydrangea, dimetredon and rubberwhale separate coz their vvlaues are too high
     # Plot bar chart: each image has two bars: farneback and assignment
    x = np.arange(len(image_names))
    width = 0.35

    # same plots for angular error
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, farneback_2param_angular_error, width, label="Farneback")
    plt.bar(x + width/2, assignment_2param_angular_error, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("Angular Error")
    plt.title("Angular Error Comparison: Farneback vs Assignment Initialization")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/2_params/error_plots/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "angular_error.png"))
    plt.show()

    # get indexs of rubberwhale, dimetredon and hydrangea from images_list
    selected_indices = [1, 2, 3]
    # separate charts for selected and remaining images
    selected_image_names = [image_names[i] for i in selected_indices]
    selected_farneback_epe = [farneback_2param_epe[i] for i in selected_indices]
    selected_assignment_epe = [assignment_2param_epe[i] for i in selected_indices]
    
    remaining_image_names = [image_names[i] for i in range(len(image_names)) if i not in selected_indices]
    remaining_farneback_epe = [farneback_2param_epe[i] for i in range(len(image_names)) if i not in selected_indices]
    remaining_assignment_epe = [assignment_2param_epe[i] for i in range(len(image_names)) if i not in selected_indices]

    # Plot selected images bar chart
    x_selected = np.arange(len(selected_image_names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x_selected - width/2, selected_farneback_epe, width, label="Farneback")
    plt.bar(x_selected + width/2, selected_assignment_epe, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("EPE")
    plt.title("EPE Comparison (Selected Images)")
    plt.xticks(x_selected, selected_image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/2_params/error_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "large_epe.png"))
    plt.show()
    
    # Plot remaining images bar chart
    x_remaining = np.arange(len(remaining_image_names))
    plt.figure(figsize=(10, 6))
    plt.bar(x_remaining - width/2, remaining_farneback_epe, width, label="Farneback")
    plt.bar(x_remaining + width/2, remaining_assignment_epe, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("EPE")
    plt.title("EPE Comparison (Remaining Images)")
    plt.xticks(x_remaining, remaining_image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/2_params/error_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "small_epe.png"))
    plt.show()
    
    

def init_methods_plus_8param_comparison():
    refine_params = {
        "steps": 8000,
        "lr": 1e-2,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1, 
    }
    # Lists to store results
    farneback_2param_epe = []
    assignment_2param_epe = []
    image_names = []
    farneback_2param_angular_error = []
    assignment_2param_angular_error = []
    output_dir = "benchmarks/8_params/images/"

    for img1_path, img2_path, flo_path in image_list:
        curr_output_dir = output_dir + img1_path.split("\\")[-2]
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        image_names.append(img1_path.split("\\")[-2])
        image1, image2 = read_input_images(img1_path, img2_path)
        gt_flow = read_input_ground_truth(flo_path)
        
        # Farneback Initialization
        flow_farnedback = AffineFlowWithLocalOrigins(image1, image2, gt_flow, init_params, use_opencv=True)
        flow_refined_farnedback, I2_warp_farneback = refine_flow(flow_farnedback, refine_params)
        farneback_2param_epe.append(flow_farnedback.epe_error())
        farneback_2param_angular_error.append(flow_farnedback.angular_error())

        # custom lucas kanade Initialization
        flow_assignment = AffineFlowWithLocalOrigins(image1, image2, gt_flow, init_params, use_opencv=False)
        flow_refined_assignment, I2_warp_assignment = refine_flow(flow_assignment, refine_params)
        assignment_2param_epe.append(flow_assignment.epe_error())
        assignment_2param_angular_error.append(flow_assignment.angular_error())

        I2_warp_uint8_farneback = (np.clip(I2_warp_farneback, 0, 1) * 255).astype(np.uint8)
        I2_warp_uint8_assignment = (np.clip(I2_warp_assignment, 0, 1) * 255).astype(np.uint8)
        # save image1, image2, gt_glow, init_flow to output_dir
        result_img = {"I2_warp_farneback": I2_warp_uint8_farneback, 
                      "I2_warp_assignment": I2_warp_uint8_assignment,
                      "image1": convert_torch_to_cv(flow_assignment.image1), 
                    "image2": convert_torch_to_cv(flow_assignment.image2),
                    "flow_farneback": visualize_flow_hsv(flow_refined_farnedback),
                    "flow_assignment": visualize_flow_hsv(flow_refined_assignment),
                    "init_flow_farneback": visualize_flow_hsv(flow_farnedback.init_flow),
                    "init_flow_assignment": visualize_flow_hsv(flow_assignment.init_flow),
                    "gt_flow": visualize_gt_flow_hsv(flow_assignment.gt_flow.cpu().numpy())}
        for name, img in result_img.items():
            plot_title = os.path.join(curr_output_dir, f"{name}.png")
            cv.imwrite(plot_title, img)


    # epe for hydrangea, dimetredon and rubberwhale separate coz their vvlaues are too high
     # Plot bar chart: each image has two bars: farneback and assignment
    x = np.arange(len(image_names))
    width = 0.35

    # same plots for angular error
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, farneback_2param_angular_error, width, label="Farneback")
    plt.bar(x + width/2, assignment_2param_angular_error, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("Angular Error")
    plt.title("Angular Error Comparison: Farneback vs Assignment Initialization")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/8_params/error_plots/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "angular_error.png"))
    plt.show()

    # get indexs of rubberwhale, dimetredon and hydrangea from images_list
    selected_indices = [1, 2, 3]
    # separate charts for selected and remaining images
    selected_image_names = [image_names[i] for i in selected_indices]
    selected_farneback_epe = [farneback_2param_epe[i] for i in selected_indices]
    selected_assignment_epe = [assignment_2param_epe[i] for i in selected_indices]
    
    remaining_image_names = [image_names[i] for i in range(len(image_names)) if i not in selected_indices]
    remaining_farneback_epe = [farneback_2param_epe[i] for i in range(len(image_names)) if i not in selected_indices]
    remaining_assignment_epe = [assignment_2param_epe[i] for i in range(len(image_names)) if i not in selected_indices]

    # Plot selected images bar chart
    x_selected = np.arange(len(selected_image_names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x_selected - width/2, selected_farneback_epe, width, label="Farneback")
    plt.bar(x_selected + width/2, selected_assignment_epe, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("EPE")
    plt.title("EPE Comparison (Selected Images)")
    plt.xticks(x_selected, selected_image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/8_params/error_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "large_epe.png"))
    plt.show()
    
    # Plot remaining images bar chart
    x_remaining = np.arange(len(remaining_image_names))
    plt.figure(figsize=(10, 6))
    plt.bar(x_remaining - width/2, remaining_farneback_epe, width, label="Farneback")
    plt.bar(x_remaining + width/2, remaining_assignment_epe, width, label="Assignment")
    plt.xlabel("Image")
    plt.ylabel("EPE")
    plt.title("EPE Comparison (Remaining Images)")
    plt.xticks(x_remaining, remaining_image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    output_dir = "benchmarks/8_params/error_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "small_epe.png"))
    plt.show()

def fixed_farneback_init_plus_different_refinement_comparison():
    refine_params = {
        "steps": 8000,
        "lr": 1e-2,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1, 
    }
    # Lists to store results
    iternations_list = [0, 2000, 4000, 6000, 8000]
    uv_epe = []
    affine_epe = []
    image_names = []
    uv_angular_error = []
    affine_angular_error = []
    output_dir = "benchmarks/farneback_fix_diff_refinement/images/"

    for img1_path, img2_path, flo_path in image_list:
        curr_output_dir = output_dir + img1_path.split("\\")[-2]
        if img1_path.split("\\")[-2] in ["Dimetrodon", "RubberWhale", "Hydrangea"]:
            continue
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        image_names.append(img1_path.split("\\")[-2])
        image1, image2 = read_input_images(img1_path, img2_path)
        gt_flow = read_input_ground_truth(flo_path)
        
        # Farneback Initialization for 2 param refinement
        uv_flow = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=True)
        uv_flow_refined, I2_warp_uv = refine_flow(uv_flow, refine_params)
        # get epe and angular error from uv_flow.log_metrics. this shoudl build a list of errors from uv_flow.log_metrics where it stores the errors for evry 50 iterations. the list should only contsain errors at iternations in iternations_list
        epe = []
        ang_err = []
        for i, x in enumerate(uv_flow.log_metrics["epe_log"]):
            if i * 50 in iternations_list:
                epe.append(x)
        for i, x in enumerate(uv_flow.log_metrics["angular_log"]):
            if i * 50 in iternations_list:
                ang_err.append(x)
        uv_epe.append(epe)
        uv_angular_error.append(ang_err)


        # custom lucas kanade Initialization
        affine_flow = AffineFlowWithLocalOrigins(image1, image2, gt_flow, init_params, use_opencv=True)
        affine_flow_refined, I2_warp_affine = refine_flow(affine_flow, refine_params)
        epe = []
        ang_err = []
        for i, x in enumerate(affine_flow.log_metrics["epe_log"]):
            if i * 50 in iternations_list:
                epe.append(x)
        for i, x in enumerate(affine_flow.log_metrics["angular_log"]):
            if i * 50 in iternations_list:
                ang_err.append(x)
        affine_epe.append(epe)
        affine_angular_error.append(ang_err)

        I2_warp_uint8_uv = (np.clip(I2_warp_uv, 0, 1) * 255).astype(np.uint8)
        I2_warp_uint8_affine = (np.clip(I2_warp_affine, 0, 1) * 255).astype(np.uint8)
        # save image1, image2, gt_glow, init_flow to output_dir
        result_img = {"I2_warp_uv": I2_warp_uint8_uv, 
                      "I2_warp_affine": I2_warp_uint8_affine,
                      "image1": convert_torch_to_cv(uv_flow.image1), 
                    "image2": convert_torch_to_cv(uv_flow.image2),
                    "flow_uv": visualize_flow_hsv(uv_flow_refined),
                    "flow_affine": visualize_flow_hsv(affine_flow_refined),
                    "init_flow_uv": visualize_flow_hsv(uv_flow.init_flow),
                    "init_flow_affine": visualize_flow_hsv(affine_flow.init_flow),
                    "gt_flow": visualize_gt_flow_hsv(affine_flow.gt_flow.cpu().numpy())}
        for name, img in result_img.items():
            plot_title = os.path.join(curr_output_dir, f"{name}.png")
            cv.imwrite(plot_title, img)

    print(uv_epe)
    print(affine_epe)
    print(uv_angular_error)
    print(affine_angular_error)


    num_images = len(image_names)
    x = np.arange(num_images)
    bar_total_width = 0.8   # total width reserved for bars per group

    # --- Plot UV EPE ---
    num_iters_uv_epe = min(len(lst) for lst in uv_epe)
    bar_width = bar_total_width / num_iters_uv_epe
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_uv_epe):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [uv_epe[i][j] for i in range(num_images)]
        # if j exceeds iternations_list length, show 'Iter?'
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("EPE (UV)")
    plt.title("UV EPE vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    uv_epe_out = os.path.join("benchmarks", "farneback_fix_diff_refinement", "uv_epe.png")
    os.makedirs(os.path.dirname(uv_epe_out), exist_ok=True)
    plt.savefig(uv_epe_out)
    plt.show()

    # --- Plot Affine EPE ---
    num_iters_affine_epe = min(len(lst) for lst in affine_epe)
    bar_width = bar_total_width / num_iters_affine_epe
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_affine_epe):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [affine_epe[i][j] for i in range(num_images)]
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("EPE (Affine)")
    plt.title("Affine EPE vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    affine_epe_out = os.path.join("benchmarks", "farneback_fix_diff_refinement", "affine_epe.png")
    os.makedirs(os.path.dirname(affine_epe_out), exist_ok=True)
    plt.savefig(affine_epe_out)
    plt.show()

    # --- Plot UV Angular Error ---
    num_iters_uv_ang = min(len(lst) for lst in uv_angular_error)
    bar_width = bar_total_width / num_iters_uv_ang
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_uv_ang):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [uv_angular_error[i][j] for i in range(num_images)]
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("Angular Error (UV)")
    plt.title("UV Angular Error vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    uv_ang_out = os.path.join("benchmarks", "farneback_fix_diff_refinement", "uv_angular_error.png")
    os.makedirs(os.path.dirname(uv_ang_out), exist_ok=True)
    plt.savefig(uv_ang_out)
    plt.show()

    # --- Plot Affine Angular Error ---
    num_iters_affine_ang = min(len(lst) for lst in affine_angular_error)
    bar_width = bar_total_width / num_iters_affine_ang
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_affine_ang):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [affine_angular_error[i][j] for i in range(num_images)]
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("Angular Error (Affine)")
    plt.title("Affine Angular Error vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    affine_ang_out = os.path.join("benchmarks", "farneback_fix_diff_refinement", "affine_angular_error.png")
    os.makedirs(os.path.dirname(affine_ang_out), exist_ok=True)
    plt.savefig(affine_ang_out)
    plt.show()


def fixed_lk_init_plus_different_refinement_comparison():
    refine_params = {
        "steps": 8000,
        "lr": 1e-2,
        "edge_beta": 20.0,
        "eps": 1e-3,
        "lambda_smooth": 0.1, 
    }
    iternations_list = [0, 2000, 4000, 6000, 8000]
    uv_epe = []
    affine_epe = []
    image_names = []
    uv_angular_error = []
    affine_angular_error = []
    output_dir = "benchmarks/lk_fix_diff_refinement/images/"

    for img1_path, img2_path, flo_path in image_list:
        if img1_path.split("\\")[-2] in ["Dimetrodon", "RubberWhale", "Hydrangea"]:
            continue
        curr_output_dir = output_dir + img1_path.split("\\")[-2]
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        image_names.append(img1_path.split("\\")[-2])
        image1, image2 = read_input_images(img1_path, img2_path)
        gt_flow = read_input_ground_truth(flo_path)
        
        # Farneback Initialization for 2 param refinement
        uv_flow = CustomLucasKanadeFlow(image1, image2, gt_flow, init_params, use_opencv=False)
        uv_flow_refined, I2_warp_uv = refine_flow(uv_flow, refine_params)
        # get epe and angular error from uv_flow.log_metrics. this shoudl build a list of errors from uv_flow.log_metrics where it stores the errors for evry 50 iterations. the list should only contsain errors at iternations in iternations_list
        epe = []
        ang_err = []
        for i, x in enumerate(uv_flow.log_metrics["epe_log"]):
            if i * 50 in iternations_list:
                epe.append(x)
        for i, x in enumerate(uv_flow.log_metrics["angular_log"]):
            if i * 50 in iternations_list:
                ang_err.append(x)
        uv_epe.append(epe)
        uv_angular_error.append(ang_err)


        # custom lucas kanade Initialization
        affine_flow = AffineFlowWithLocalOrigins(image1, image2, gt_flow, init_params, use_opencv=False)
        affine_flow_refined, I2_warp_affine = refine_flow(affine_flow, refine_params)
        epe = []
        ang_err = []
        for i, x in enumerate(affine_flow.log_metrics["epe_log"]):
            if i * 50 in iternations_list:
                epe.append(x)
        for i, x in enumerate(affine_flow.log_metrics["angular_log"]):
            if i * 50 in iternations_list:
                ang_err.append(x)
        affine_epe.append(epe)
        affine_angular_error.append(ang_err)

        I2_warp_uint8_uv = (np.clip(I2_warp_uv, 0, 1) * 255).astype(np.uint8)
        I2_warp_uint8_affine = (np.clip(I2_warp_affine, 0, 1) * 255).astype(np.uint8)
        # save image1, image2, gt_glow, init_flow to output_dir
        result_img = {"I2_warp_uv": I2_warp_uint8_uv, 
                      "I2_warp_affine": I2_warp_uint8_affine,
                      "image1": convert_torch_to_cv(uv_flow.image1), 
                    "image2": convert_torch_to_cv(uv_flow.image2),
                    "flow_uv": visualize_flow_hsv(uv_flow_refined),
                    "flow_affine": visualize_flow_hsv(affine_flow_refined),
                    "init_flow_uv": visualize_flow_hsv(uv_flow.init_flow),
                    "init_flow_affine": visualize_flow_hsv(affine_flow.init_flow),
                    "gt_flow": visualize_gt_flow_hsv(affine_flow.gt_flow.cpu().numpy())}
        for name, img in result_img.items():
            plot_title = os.path.join(curr_output_dir, f"{name}.png")
            cv.imwrite(plot_title, img)

    print(uv_epe)
    print(affine_epe)
    print(uv_angular_error)
    print(affine_angular_error)


    num_images = len(image_names)
    x = np.arange(num_images)
    bar_total_width = 0.8   # total width reserved for bars per group

    # --- Plot UV EPE ---
    num_iters_uv_epe = min(len(lst) for lst in uv_epe)
    bar_width = bar_total_width / num_iters_uv_epe
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_uv_epe):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [uv_epe[i][j] for i in range(num_images)]
        # if j exceeds iternations_list length, show 'Iter?'
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("EPE (UV)")
    plt.title("UV EPE vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    uv_epe_out = os.path.join("benchmarks", "lk_fix_diff_refinement", "uv_epe.png")
    os.makedirs(os.path.dirname(uv_epe_out), exist_ok=True)
    plt.savefig(uv_epe_out)
    plt.show()

    # --- Plot Affine EPE ---
    num_iters_affine_epe = min(len(lst) for lst in affine_epe)
    bar_width = bar_total_width / num_iters_affine_epe
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_affine_epe):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [affine_epe[i][j] for i in range(num_images)]
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("EPE (Affine)")
    plt.title("Affine EPE vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    affine_epe_out = os.path.join("benchmarks", "lk_fix_diff_refinement", "affine_epe.png")
    os.makedirs(os.path.dirname(affine_epe_out), exist_ok=True)
    plt.savefig(affine_epe_out)
    plt.show()

    # --- Plot UV Angular Error ---
    num_iters_uv_ang = min(len(lst) for lst in uv_angular_error)
    bar_width = bar_total_width / num_iters_uv_ang
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_uv_ang):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [uv_angular_error[i][j] for i in range(num_images)]
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("Angular Error (UV)")
    plt.title("UV Angular Error vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    uv_ang_out = os.path.join("benchmarks", "lk_fix_diff_refinement", "uv_angular_error.png")
    os.makedirs(os.path.dirname(uv_ang_out), exist_ok=True)
    plt.savefig(uv_ang_out)
    plt.show()

    # --- Plot Affine Angular Error ---
    num_iters_affine_ang = min(len(lst) for lst in affine_angular_error)
    bar_width = bar_total_width / num_iters_affine_ang
    plt.figure(figsize=(12, 6))
    for j in range(num_iters_affine_ang):
        offsets = x - bar_total_width/2 + j * bar_width + bar_width/2
        values = [affine_angular_error[i][j] for i in range(num_images)]
        label = f"Iter {iternations_list[j]}" if j < len(iternations_list) else f"Iter {j}"
        plt.bar(offsets, values, width=bar_width, label=label)
    plt.xlabel("Image")
    plt.ylabel("Angular Error (Affine)")
    plt.title("Affine Angular Error vs Iterations")
    plt.xticks(x, image_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    affine_ang_out = os.path.join("benchmarks", "lk_fix_diff_refinement", "affine_angular_error.png")
    os.makedirs(os.path.dirname(affine_ang_out), exist_ok=True)
    plt.savefig(affine_ang_out)
    plt.show()


def main():
    # init_method_comparison() 
    init_methods_plus_2param_comparison()
    # init_methods_plus_8param_comparison()
    # fixed_farneback_init_plus_different_refinement_comparison()
    # fixed_lk_init_plus_different_refinement_comparison()


if __name__ == "__main__":
    main()


