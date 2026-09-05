import cv2
import numpy as np
import argparse
import torch
import argparse
from Visualizer import Visualizer
from YoloPV2ExternalStuff.tools.PointsChoosing import camera_calibration
def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--conf-thres", type=float, default=0.8)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")

    opt = parser.parse_args()

    image = Image_Input()

    model = torch.jit.load("YoloPV2ExternalStuff/weights/yolopv2.pt", map_location="cpu")
    model.eval()



    result = Visualizer(
        image,
        model,
        opt
    ).visualize()

    cv2.imshow("Visualizer Test", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Image_Input():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required = True)
        
        args = parser.parse_args()
        image = cv2.imread(args.input)
        
        if image is None:
            print("couldn't process image")
            exit()
            
        return image 
    
if __name__ == "__main__":
    main()
        