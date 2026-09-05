from YoloPV2ExternalStuff.utils.utils import lane_line_mask
from ImageProcessing import ImageProcessing
class LaneDetection():
    def __init__(self,model, img):
        
        self.model = model
        self.img = ImageProcessing(img) 
    
    def detectlane(self):
        # Inference
        _,_,ll = self.model(self.img)

        ll_seg_mask = lane_line_mask(ll)
        
        return ll_seg_mask
    
