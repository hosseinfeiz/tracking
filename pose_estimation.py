import os
import torch
from ultralytics import YOLO
from config import DEVICE, KEYPOINTS

if DEVICE != 'cpu' and torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

class Yolov8PoseModel:
    def __init__(self, device: str, person_conf, kpts_threshold):
        self.person_conf = person_conf
        self.kpts_threshold = kpts_threshold
        self.model = YOLO('yolov8l.pt')

        
    def run_inference(self, image):
        results = self.model(image)
        return results
    
    def get_filtered_bboxes_by_confidence(self, image):
        results = self.run_inference(image)

        conf_filtered_bboxes = [] #01_April_2023_10_53_54_00_00_00_00_01_19
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            # all_kpts = result.keypoints
            for i, box in enumerate(boxes):
                
                if box.conf[0] > self.person_conf and box.cls[0] == 0:
                    

                    conf_filtered_bboxes.append( box.xyxy[0].astype(int))
        
        return conf_filtered_bboxes
    
    
    def get_filtered_bboxes_by_size(self, bboxes, image, percentage=10):
        image_size = image.shape[:2]
        min_bbox_width = image_size[0] * (percentage/100)
        min_bbox_height = image_size[1] * (percentage/100)

        filtered_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width >= min_bbox_width and bbox_height >= min_bbox_height:
                filtered_bboxes.append(bbox)

        return size_filtered_bboxes