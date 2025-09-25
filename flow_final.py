from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv

from init_flow import calculate_initial_flow
from refine import refine_dense_flow, refine_dense_affine_flow
from utils import visualize_flow_hsv

# Abstract base class for optical flow types (2 Param lucas kanade and 8 param affine)
class Flow(ABC):
    def __init__(self, image1, image2, init_params, use_opencv=False):
        self.params = calculate_initial_flow(image1, 
                                             image2, 
                                             init_params["levels"], 
                                             init_params["window_size"], 
                                             init_params["alpha"], 
                                             init_params["goodness_threshold"], 
                                             use_opencv)

    # Refine the calculated initialflow using gradient descent
    @abstractmethod
    def refine_flow(self, image1, image2, ground_truth, refine_params):
        pass

    # Warp an image using the flow field
    def warp_with_flow(self, image):
        pass

    # Visualize the flow field
    @abstractmethod
    def visualize_flow(self):
        pass


class CustomLucasKanadeFlow(Flow):
    def __init__(self):
        super().__init__()


    def refine_flow(self, image1, image2, ground_truth, refine_params):
        return refine_dense_flow(image1, image2, self.params, refine_params["steps"], ground_truth)
    
    def warp_with_flow(self, image):
        

    def visualize_flow(self):
        
    


class AffineFlow(Flow):
    def __init__(self):
        super().__init__()
        # Initialize any additional parameters for affine flow and local origins



    def calculate_initial_flow(self, image1, image2, levels, window_size, alpha, goodness_threshold):