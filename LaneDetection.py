from YoloPV2ExternalStuff.utils.utils import lane_line_mask
import torch 
import cv2
class LaneDetection():
    def __init__(self,model, img):
        
        self.model = model
        self.img = img
    
    def detectlane(self):
        img = self.img

        img = cv2.resize(img, (640, 640))
        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)
        img = img.copy()

        img = torch.from_numpy(img)
        img = img.float()
        img /= 255.0

        img = img.unsqueeze(0)
        # Inference
        _,_,ll = self.model(img)

        ll_seg_mask = lane_line_mask(ll)
        
        return ll_seg_mask
    
