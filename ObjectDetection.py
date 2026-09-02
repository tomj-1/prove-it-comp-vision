from YoloPV2ExternalStuff.utils.utils import scale_coords, time_synchronized, \
split_for_trace_model, non_max_suppression \

class ObjectDetection():
    
    def __init__(self,model,opt, img):
        
        self.model = model
        self.opt = opt
        self.img = img   
    
    def detectobject(self):
        # Inference
        [pred,anchor_grid],_,_ = self.model(self.img)

        pred = split_for_trace_model(pred,anchor_grid)

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, 
                                   classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        
        return pred
    
    