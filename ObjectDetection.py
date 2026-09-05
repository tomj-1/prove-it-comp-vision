from YoloPV2ExternalStuff.utils.utils import scale_coords, time_synchronized, \
split_for_trace_model, non_max_suppression \

from ImageProcessing import ImageProcessing
import cv2
import torch
class ObjectDetection():
    
    def __init__(self,model,opt, img):
        
        self.model = model
        self.opt = opt
        self.img = img  
    
    def detectobject(self):
        img = self.img

        img = cv2.resize(img, (640, 640))
        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)
        img = img.copy()

        img = torch.from_numpy(img)
        img = img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        [pred,anchor_grid],_,_ = self.model(img)

        pred = split_for_trace_model(pred,anchor_grid)

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, 
                                   classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        
        return pred[0]
