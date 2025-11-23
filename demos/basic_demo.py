"""
Demo: Pose Estimation Pipeline
This demo shows how to use the pose estimation pipeline
"""

import sys
import os
# Add local easy_ViTPose to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "easy_ViTPose"))

import cv2
import numpy as np
import time
import json
from google.colab.patches import cv2_imshow
from ultralytics import YOLO
from easy_ViTPose import VitInference

def run_demo():
    print("🚀 POSE ESTIMATION DEMO")
    print("=" * 30)
    
    # Initialize models
    yolo_model = YOLO("models/yolov8s.pt")
    
    vitpose_model = VitInference(
        "models/vitpose-b.pth",
        "models/yolov8s.pt", 
        "b", 
        dataset="coco", 
        yolo_size=320, 
        is_video=False, 
        device="cuda"
    )
    
    # Load video frame
    video_path = "data/input/campus_walk.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not load frame")
        return
    
    print(f"✅ Loaded frame: {frame.shape}")
    
    # YOLO detection
    yolo_start = time.time()
    results = yolo_model(frame, verbose=False)
    yolo_time = time.time() - yolo_start
    
    # Get largest bbox
    detections = results[0]
    bboxes = []
    if detections.boxes is not None:
        boxes = detections.boxes.xyxy.cpu().numpy()
        confidences = detections.boxes.conf.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confidences, classes):
            if cls_id == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                bboxes.append(([x1, y1, x2, y2], area))
    
    if not bboxes:
        print("❌ No people detected")
        return
    
    largest_bbox, largest_area = max(bboxes, key=lambda x: x[1])
    x1, y1, x2, y2 = largest_bbox
    print(f"✅ Largest bbox: Area {largest_area}")
    
    # ViTPose on largest bbox
    crop = frame[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    vitpose_start = time.time()
    keypoints_dict = vitpose_model.inference(crop_rgb)
    vitpose_time = time.time() - vitpose_start
    
    # Display results
    vis_frame = frame.copy()
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    if keypoints_dict:
        keypoints = list(keypoints_dict.values())[0]
        keypoints[:, 0] += y1
        keypoints[:, 1] += x1
        
        # Draw keypoints
        for kp in keypoints:
            y, x, conf = kp
            if conf > 0.3:
                cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    cv2_imshow(vis_frame)
    
    print(f"📊 Performance:")
    print(f"   • YOLO: {yolo_time*1000:.1f}ms")
    print(f"   • ViTPose: {vitpose_time*1000:.1f}ms")
    print(f"   • Total: {(yolo_time + vitpose_time)*1000:.1f}ms")

if __name__ == "__main__":
    run_demo()
