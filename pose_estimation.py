import os
import torch
from ultralytics import YOLO
from config import DEVICE, KEYPOINTS

if DEVICE != 'cpu' and torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    # IoU
    iou = intersection_area / union_area
    return iou

class Yolov8PoseModel:
    def __init__(self, device: str, person_conf, kpts_threshold):
        self.person_conf = person_conf
        self.kpts_threshold = kpts_threshold
        self.model = YOLO('yolov8m.pt')
    def run_inference(self, image):
        results = self.model(image, classes=[0,1])
        return results 
    def get_filtered_bboxes_by_confidence(self, image):
        results = self.run_inference(image)
        
        conf_filtered_bboxes = []
        conf = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            # all_kpts = result.keypoints
            for i, box in enumerate(boxes):
                # single_kpts_conf = all_kpts[i].conf
                
                # r_sho_proba = single_kpts_conf[0].cpu().numpy()[KEYPOINTS.index("right_shoulder")]
                # l_sho_proba = single_kpts_conf[0].cpu().numpy()[KEYPOINTS.index("left_shoulder")]
                # r_hip_proba = single_kpts_conf[0].cpu().numpy()[KEYPOINTS.index("right_hip")]
                # l_hip_proba = single_kpts_conf[0].cpu().numpy()[KEYPOINTS.index("left_hip")]
                
                if box.conf[0] > self.person_conf:
                    conf_filtered_bboxes.append(box.xyxy[0].astype(int))
                    conf.append(box.conf[0])
        
        return conf_filtered_bboxes
    
    
    def get_filtered_bboxes_by_size(self, bboxes, image, percentage=6):
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

        return filtered_bboxes


    def remove_similar_bboxes(self, bboxes):
        """
        Remove every two bounding boxes with an IoU greater than 0.2,
        keeping only the one with the bigger area.
        """
        filtered_bboxes = []
        num_bboxes = len(bboxes)

        # Create a list to keep track of which bounding boxes to remove
        to_remove = [False] * num_bboxes

        for i in range(num_bboxes):
            if not to_remove[i]:
                bbox1 = bboxes[i]
                for j in range(i + 1, num_bboxes):
                    if not to_remove[j]:
                        bbox2 = bboxes[j]
                        iou = calculate_iou(bbox1, bbox2)
                        if iou > 0.6:
                            # Compare areas and mark the smaller one for removal
                            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                            if area1 < area2:
                                to_remove[i] = True
                            else:
                                to_remove[j] = True
                            break

        # Add bounding boxes that are not marked for removal to the filtered list
        for i in range(num_bboxes):
            if not to_remove[i]:
                filtered_bboxes.append(bboxes[i])

        return filtered_bboxes
