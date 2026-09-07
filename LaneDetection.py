from YoloPV2ExternalStuff.utils.utils import lane_line_mask, letterbox
import torch 
import cv2
class LaneDetection():
    def __init__(self,model, img):
        
        self.model = model
        self.img = img
    
    def detectlane(self):
        
        original_h, original_w = self.img.shape[:2]
        
        
        img = self.img.copy()
        
        img = letterbox(img,640, stride=32)[0]

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
        print("lane mask:", ll_seg_mask.shape)
        print("original image:", self.img.shape)
                

        # Resize lane mask back to original image size
        ll_seg_mask = cv2.resize(
            ll_seg_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        return ll_seg_mask
    
