"""
Demo Stage 1: YOLOv8 Person Detection and BBox Export
"""

import sys
import os
import json
import cv2
import numpy as np
import time
from ultralytics import YOLO

def stage1_detection():
    """Stage 1: Run YOLOv8 detection and save bboxes to JSON"""
    print("🚀 STAGE 1: YOLOv8 Person Detection")
    print("=" * 40)
    
    # Initialize YOLO model
    yolo_model = YOLO("models/yolov8s.pt")
    
    # Warm-up run to load model and initialize GPU
    print("🔥 Warming up YOLO model...")
    warm_up_start = time.time()
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    _ = yolo_model(dummy_image, verbose=False)
    warm_up_time = time.time() - warm_up_start
    print(f"✅ YOLO warm-up completed in {warm_up_time:.2f}s")
    
    # Open video
    video_path = os.path.join(os.path.dirname(__file__), "..", "data", "input", "campus_walk.mp4")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open video file")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video: {total_frames} frames, {fps:.2f} FPS")
    
    # Storage for bboxes
    bboxes_data = {}
    frame_times = []
    
    print("\n🔍 Processing frames...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # YOLO detection
        frame_start = time.time()
        results = yolo_model(frame, verbose=False)
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        # Process detections
        detections = results[0]
        frame_bboxes = []
        
        if detections.boxes is not None:
            boxes = detections.boxes.xyxy.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            
            person_bboxes = []
            for box, conf, cls_id in zip(boxes, confidences, classes):
                if cls_id == 0 and conf > 0.5:  # person class with confidence > 0.5
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    person_bboxes.append(([x1, y1, x2, y2], area, float(conf)))
            
            # Sort by area and take top 3 largest
            person_bboxes.sort(key=lambda x: x[1], reverse=True)
            top_bboxes = person_bboxes[:3]
            
            # Store bbox info
            for bbox, area, conf in top_bboxes:
                frame_bboxes.append({
                    "bbox": bbox,
                    "area": int(area),
                    "confidence": conf
                })
        
        bboxes_data[frame_count] = frame_bboxes
        
        # Progress reporting every 50 frames
        if frame_count % 50 == 0:
            avg_time = np.mean(frame_times[-50:]) if len(frame_times) >= 50 else np.mean(frame_times)
            current_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"   Frame {frame_count}/{total_frames} - Avg: {avg_time*1000:.1f}ms, FPS: {current_fps:.1f}")
        
        frame_count += 1
    
    cap.release()
    total_time = time.time() - start_time
    
    # Save bboxes to JSON
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "stage1_bboxes.json")
    with open(output_path, 'w') as f:
        json.dump(bboxes_data, f, indent=2)
    
    # Performance summary
    print("\n✅ STAGE 1 COMPLETED")
    print("=" * 40)
    print(f"📊 Performance Summary:")
    print(f"   • Total frames processed: {frame_count}")
    print(f"   • Total time: {total_time:.2f}s")
    print(f"   • Average time per frame: {np.mean(frame_times)*1000:.1f}ms")
    print(f"   • Average FPS: {1.0/np.mean(frame_times):.1f}")
    print(f"   • Bboxes saved to: {output_path}")
    
    return bboxes_data

if __name__ == "__main__":
    print("🎯 Easy Pose Pipeline - Stage 1 Only")
    print("=" * 50)
    stage1_detection()
