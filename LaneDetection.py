from YoloPV2ExternalStuff.utils.utils import lane_line_mask, letterbox
import cv2
from ImageProcessing import ImageProcessing
class LaneDetection():
    def __init__(self, model, img, opt):
        
        self.model = model
        self.img = img
        self.opt = opt
    
    def detectlane(self):
        
        original_h, original_w = self.img.shape[:2]
        
        img = self.img.copy()
        
    
        img = letterbox(img, (640, 640),  stride=32)[0]
        
    
        processor = ImageProcessing(self.opt)
        
        img = processor.preprocess_image(img)
        
        
        # Inference
        _,_,ll = self.model(img)

        ll_seg_mask = lane_line_mask(ll)
                

        # Resize lane mask back to original image size
        ll_seg_mask = cv2.resize(
            ll_seg_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        return ll_seg_mask
    
