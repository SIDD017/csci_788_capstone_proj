# from utils import *
# from ground_truth_flo import visualize_gt_flow_hsv, read_flo_file

# def main():
#     args = process_args()

#     # Compute forward optical flow (from image1 to image2)
#     forward_flow = cv.calcOpticalFlowFarneback(
#         args.image1,
#         args.image2,
#         None,
#         pyr_scale=0.5,
#         levels=args.levels,
#         winsize=args.window_size,
#         iterations=3,
#         poly_n=5,
#         poly_sigma=1.2,
#         flags=0,
#     )

#     # Compute reverse optical flow (from image2 to image1)
#     reverse_flow = cv.calcOpticalFlowFarneback(
#         args.image2,
#         args.image1,
#         None,
#         pyr_scale=0.5,
#         levels=args.levels,
#         winsize=args.window_size,
#         iterations=3,
#         poly_n=5,
#         poly_sigma=1.2,
#         flags=0,
#     )

#     try:
#         gt_flow = read_flo_file(args.gtimage)
#     except Exception as e:
#         exit(1)

#     # Consistency check: locations with large disagreement between flows are set to NaN.
#     consistency = np.linalg.norm(forward_flow + reverse_flow, axis=2)
#     is_bad = consistency > args.goodness_threshold  # set goodness_threshold in process_args
#     # Combine forward and reverse flows to get the final estimate.
#     flow = (forward_flow - reverse_flow) / 2
#     flow[is_bad] = np.nan

#     flow_viz = visualize_flow_hsv(flow)
#     cv.imshow("Dense flow (Farneback)", flow_viz)
#     gt_display = visualize_gt_flow_hsv(gt_flow)
#     cv.imshow("Optical flow (ground truth)", gt_display)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# # TODO List:
# # opencv optical flow - benchmark against old assignment code
# # See if we should have sep[arate gradient descent for affine and translation parameters
# # Make the debugging viualizatyions for all 8 paramters as told by lange
# # Make the interface thingy that Lange wanted
