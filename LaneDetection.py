from YoloPV2ExternalStuff.utils.utils import lane_line_mask
class LaneDetection():
    def __init__(self,model, img):
        
        self.model = model
        self.img = img   
    
    def detectlane(self):
        # Inference
        _,_,ll = self.model(self.img)

        ll_seg_mask = lane_line_mask(ll)
        
        return ll_seg_mask
    
