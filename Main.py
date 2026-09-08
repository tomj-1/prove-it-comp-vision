import cv2
import numpy as np
import argparse
import torch
import argparse
from Visualizer import Visualizer
from YoloPV2ExternalStuff.tools.PointsChoosing import camera_calibration
def main():
    
    #arguments to pass in command line like image path 
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--conf-thres", type=float, default=0.8)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    opt = parser.parse_args()

    image = Image_Input(opt.input)

    #load model
    model = torch.jit.load("YoloPV2ExternalStuff/weights/yolopv2.pt", map_location=opt.device)
    model.eval()



    result = Visualizer(
        image,
        model,
        opt
    ).visualize()

    #load image
    cv2.imshow("Visualizer Test", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Image_Input(input_path):
        image = cv2.imread(input_path)
        
        if image is None:
            print("couldn't process image")
            exit()
            
        return image 
    
if __name__ == "__main__":
    main()
        