import torch
import argparse
from YoloPV2ExternalStuff.utils.utils import select_device, LoadImages
class ImageProcessing():
        
    
    def __init__(self, opt):
        
        self.opt = opt
        self.source = self.opt.input
        self.half = opt.device != "cpu"
               

    def load_image(self):
        vid_path, vid_writer = None, None
        self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        return self.dataset

    def preprocess_image(self,img):
        
        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)
        img = img.copy()
        img = torch.from_numpy(img)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        img = img.unsqueeze(0)
            
        return img
           

            

        
    