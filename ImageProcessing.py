import torch
import argparse
import Main
from YoloPV2ExternalStuff.utils.utils import select_device, LoadImages
class ImageProcessing():
        
    
    def __init__(self):
        
          
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='YoloPV2ExternalStuff/weights/yolopv2.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='inference/vid2', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        # setting and directories
        self.source, self.weights, self.save_txt, self.imgsz = self.opt.source, self.opt.weights,  self.opt.save_txt, self.opt.img_size
        save_img = not self.opt.nosave and not self.source.endswith('.txt')  # save inference images
        
        # Load model
        self.device = select_device(self.opt.device)
        self.stride =32
        self.model  = torch.jit.load(self.weights, map_location=self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = self.model.to(self.device)
               

    def load_image(self):
        vid_path, vid_writer = None, None
        self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        return self.dataset

    def preprocess_image(self,img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img
           

            

        
    