import cv2
import numpy as np
import argparse

from Visualizer import display_from_list
from ImageProcessing import ImageProcessing 

def main():
    
    
    # Show result
    cv2.imshow("Visualizer Test", imgProcessing.img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
        