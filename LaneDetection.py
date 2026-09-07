from YoloPV2ExternalStuff.utils.utils import lane_line_mask
import torch 
import cv2
class LaneDetection():
    def __init__(self,model, img):
        
        self.model = model
        self.img = img
    
    def detectlane(self):
        img = self.img

        img = letterbox(img,640)
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
    
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]  # height, width

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = (
        int(round(shape[1] * ratio)),
        int(round(shape[0] * ratio))
    )

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    img = cv2.resize(
        img,
        new_unpad,
        interpolation=cv2.INTER_LINEAR
    )

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img