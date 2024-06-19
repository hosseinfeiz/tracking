# -*- coding: utf-8 -*-
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from config import (DEVICE, INFERENCE_SIZE, IOU_THRESHOLD, KPTS_CONF,
                    MAX_OBJECT_CNT, PERSON_CONF, XMEM_CONFIG, YOLO_EVERY)
from inference.inference_utils import (add_new_classes_to_dict,
                                       generate_colors_dict,
                                       get_iou_filtered_yolo_mask_bboxes,
                                       merge_masks, overlay_mask_on_image)
from inference.interact.interactive_utils import torch_prob_to_numpy_mask
from tracker import Tracker
from pose_estimation import Yolov8PoseModel
import xml.etree.ElementTree as ET

def write_xml_file(data, path, frame_idx, frame_height, frame_width):
    try:
        track_file = os.path.dirname(path)
        if not os.path.exists(track_file):
            os.makedirs(track_file)
        
        # Check if the file exists
        if os.path.isfile(path):
            # File exists, open it for appending
            tree = ET.parse(path)
            root = tree.getroot()
        else:
            # File doesn't exist, create a new root element
            root = ET.Element("mocap")
            tree = ET.ElementTree(root)

    except FileNotFoundError:
        root = ET.Element("mocap")
        tree = ET.ElementTree(root)

    keyframe = ET.SubElement(root, "keyframe", key="{:06}".format(frame_idx))
    for person_id, bbox in data.items():
        person_key = ET.SubElement(keyframe, "key")
        person_key.set("personID", str(person_id))
        scale_y = frame_height / INFERENCE_SIZE[1]
        scale_x = frame_width / INFERENCE_SIZE[0]
        rescaled_person_data = [bbox[0] * scale_x, bbox[1] * scale_x, bbox[2] * scale_y, bbox[3] * scale_y, 0.9]
        person_key.set("bbox", " ".join(f"{value:.2f}" for value in rescaled_person_data))
    keyframe.tail = "\n"
    
    # Open the file in write mode if it doesn't exist, and append mode if it does
    with open(path, 'wb') as file:  
        tree.write(file)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str,
                        required=True, help='Path to input video')
    parser.add_argument(
        '--width', type=int, default=INFERENCE_SIZE[0], required=False, help='Inference width')
    parser.add_argument(
        '--height', type=int, default=INFERENCE_SIZE[1], required=False, help='Inference height')
    parser.add_argument('--frames_to_propagate', type=int,
                        default=None, required=False, help='Frames to propagate')
    parser.add_argument('--output_video_path', type=str, default=None,
                        required=False, help='Output video path to save')
    parser.add_argument('--device', type=str, default=DEVICE,
                        required=False, help='GPU id')
    parser.add_argument('--person_conf', type=float, default=PERSON_CONF,
                        required=False, help='YOLO person confidence')
    parser.add_argument('--kpts_conf', type=float, default=KPTS_CONF,
                        required=False, help='YOLO keypoints confidence')
    parser.add_argument('--iou_thresh', type=float, default=IOU_THRESHOLD,
                        required=False, help='IOU threshold to find new persons bboxes')
    parser.add_argument('--yolo_every', type=int, default=YOLO_EVERY,
                        required=False, help='Find new persons with YOLO every N frames')
    parser.add_argument('--output_track', type=str,
                        default='tracking_results.xml', required=False, help='Output filepath')

    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    cap = cv2.VideoCapture(args.video_path)
    df = pd.DataFrame(
        columns=['frame_id', 'person_id', 'x1', 'y1', 'x2', 'y2'])

    if args.output_video_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), fps, (args.width, args.height))

    yolov8pose_model = Yolov8PoseModel(DEVICE, PERSON_CONF, KPTS_CONF)
    tracker = Tracker(XMEM_CONFIG, MAX_OBJECT_CNT, DEVICE)
    persons_in_video = False

    class_color_mapping = generate_colors_dict(MAX_OBJECT_CNT+1)

    current_frame_index = 0
    class_label_mapping = {}
    
with torch.cuda.amp.autocast(enabled=True):

    current_frame_index = 0
    persons_in_video = False
    class_label_mapping = {}
    filtered_bboxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or (args.frames_to_propagate is not None and current_frame_index == args.frames_to_propagate):
            break

        frame_height, frame_width, _ = frame.shape
        frame_c = frame.copy()
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

        if current_frame_index % args.yolo_every == 0 and current_frame_index < 300:
            yolo_filtered_bboxes = yolov8pose_model.get_filtered_bboxes_by_confidence(frame_c)
            rescaled_bboxes = []
            scale_y = INFERENCE_SIZE[1] / frame_height
            scale_x = INFERENCE_SIZE[0] / frame_width
            for bbox in yolo_filtered_bboxes:
                xmin, ymin, xmax, ymax = bbox
                rescaled_bboxes.append([
                    int(xmin * scale_x),
                    int(ymin * scale_x),
                    int(xmax * scale_y),
                    int(ymax * scale_y),
                ])

        if len(rescaled_bboxes) > 0:
            persons_in_video = True
        else:
            masks = []
            mask_bboxes_with_idx = []

        if persons_in_video:
            if len(class_label_mapping) == 0:  # First persons in video
                mask = tracker.create_mask_from_img(frame, rescaled_bboxes, device='0')
                unique_labels = np.unique(mask)
                class_label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                mask = np.array([class_label_mapping[label] for label in mask.flat]).reshape(mask.shape)
                prediction = tracker.add_mask(frame, mask)
            elif len(filtered_bboxes) > 0:  # Additional/new persons in video
                mask = tracker.create_mask_from_img(frame, filtered_bboxes, device='0')
                unique_labels = np.unique(mask)
                mask_image = Image.fromarray(mask, mode='L')
                class_label_mapping = add_new_classes_to_dict(unique_labels, class_label_mapping)
                mask = np.array([class_label_mapping[label] for label in mask.flat]).reshape(mask.shape)
                merged_mask = merge_masks(masks.squeeze(0), torch.tensor(mask))
                prediction = tracker.add_mask(frame, merged_mask.squeeze(0).numpy())
                filtered_bboxes = []
            else:  # Only predict
                prediction = tracker.predict(frame)

            masks = torch.tensor(torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
            mask_bboxes_with_idx = tracker.masks_to_boxes_with_ids(masks)
            tracking_results = {}
            for box_idx, box in enumerate(mask_bboxes_with_idx):
                person_id = box[0]
                person_data = box[1:]
                tracking_results[person_id] = person_data  # Store tracking results for this frame

            if current_frame_index % args.yolo_every == 0 and current_frame_index < 300:
                filtered_bboxes = get_iou_filtered_yolo_mask_bboxes(rescaled_bboxes, mask_bboxes_with_idx, iou_threshold=args.iou_thresh)
            write_xml_file(tracking_results, args.output_track, current_frame_index, frame_height, frame_width)

        # VISUALIZATION
        if args.output_video_path is not None:
            if len(mask_bboxes_with_idx) > 0:
                for bbox in mask_bboxes_with_idx:
                    cv2.rectangle(frame, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), (255, 255, 0), 2)
                    cv2.putText(frame, f'{bbox[0]}', (int(bbox[1])-10, int(bbox[2])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                visualization = overlay_mask_on_image(frame, masks, class_color_mapping, alpha=0.75)
                visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
                result.write(visualization)
            else:
                result.write(frame)

        current_frame_index += 1

if args.output_video_path is not None:
    result.release()