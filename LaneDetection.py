from YoloPV2ExternalStuff.utils.utils import driving_area_mask,lane_line_mask
class LaneDetection():
    def __init__(self,model,opt, img):
        
        self.model = model
        self.opt = opt
        self.img = img   
    
    def detectlane(self):
        # Inference
        _,_,ll = self.model(self.img)

        ll_seg_mask = lane_line_mask(ll)
        
        return ll_seg_mask