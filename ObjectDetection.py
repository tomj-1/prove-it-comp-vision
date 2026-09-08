from ImageProcessing import ImageProcessing
from YoloPV2ExternalStuff.utils.utils import letterbox, split_for_trace_model, non_max_suppression 

import cv2
import torch
class ObjectDetection():
    
    def __init__(self,model,opt, img):
        
        self.model = model
        self.opt = opt
        self.img = img  
    
    def detectobject(self):

        processor = ImageProcessing(self.opt)

        img = self.img.copy()

        img = letterbox(
            img,
            (640, 640),
            auto=False,
            stride=32
        )[0]

        img = processor.preprocess_image(img)

        # Inference
        [pred,anchor_grid],_,_ = self.model(img)

        pred = split_for_trace_model(pred,anchor_grid)

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, 
                                   classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        
        return pred[0]
