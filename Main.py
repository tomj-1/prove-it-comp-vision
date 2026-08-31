import cv2
import argparse 
class Main:
    
    def Image_Input(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required = True)
        
        args = parser.parse_args()
        image = cv2.imread(args.input)
        
        if image is None:
            print("couldn't process image")
            exit()
            
        return image 
        