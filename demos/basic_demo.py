"""
Demo: Pose Estimation Pipeline - Two Stage Processing
Stage 1: YOLOv8 person detection and bbox export
Stage 2: ViTPose pose estimation using pre-computed bboxes
"""

import sys
import os
import json
import cv2
import numpy as np
import time
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Add local easy_ViTPose to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "easy_ViTPose"))
from easy_ViTPose.pose_only import VitPoseOnly

def stage1_detection():
    """Stage 1: Run YOLOv8 detection and save bboxes to JSON"""
    print("🚀 STAGE 1: YOLOv8 Person Detection")
    print("=" * 40)
    
    # Initialize YOLO model
    yolo_model = YOLO("models/yolov8s.pt")
    
    # Open video
    video_path = "data/input/campus_walk.mp4"
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
    output_path = "stage1_bboxes.json"
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

def stage2_pose_estimation(bboxes_data, frame_number=None):
    """Stage 2: Run ViTPose pose estimation using pre-computed bboxes"""
    print("\n🚀 STAGE 2: ViTPose Pose Estimation")
    print("=" * 40)
    
    # Initialize ViTPose model
    vitpose_start = time.time()
    vitpose_model = VitPoseOnly(
        "models/vitpose-b.pth",
        "b", 
        dataset="coco", 
        device="cuda"
    )
    model_load_time = time.time() - vitpose_start
    print(f"✅ ViTPose model loaded in {model_load_time:.2f}s")
    
    # Select frame if not provided
    if frame_number is None:
        available_frames = [f for f in bboxes_data.keys() if bboxes_data[f]]
        if available_frames:
            frame_number = np.random.choice(available_frames)
        else:
            print("❌ No frames with detections found")
            return
    
    print(f"📸 Processing frame {frame_number}")
    
    # Load the specific frame
    video_path = "data/input/campus_walk.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not load frame {frame_number}")
        return
    
    # Get bboxes for this frame
    frame_bboxes = bboxes_data.get(frame_number, [])
    
    if not frame_bboxes:
        print(f"❌ No bboxes found for frame {frame_number}")
        return
    
    print(f"✅ Found {len(frame_bboxes)} person(s) in frame")
    
    # Process each bbox with ViTPose
    vis_frame = frame.copy()
    inference_times = []
    
    for i, bbox_info in enumerate(frame_bboxes):
        bbox = bbox_info["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Draw bbox
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Person {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Crop and process with ViTPose
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # ViTPose inference
        inference_start = time.time()
        keypoints = vitpose_model.inference_bbox(frame, bbox)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        # Draw keypoints
            if keypoints.size > 0:
            keypoints = list(keypoints_dict.values())[0]
            keypoints[:, 0] += y1  # Adjust y coordinates
            keypoints[:, 1] += x1  # Adjust x coordinates
            
            # Draw keypoints
            for kp in keypoints:
                y, x, conf = kp
                if conf > 0.3:
                    cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    # Save output image
    output_image = f"stage2_frame_{frame_number}_output.jpg"
    cv2.imwrite(output_image, vis_frame)
    
    # Display results
    print(f"\n🖼️ Displaying results for frame {frame_number}:")
    cv2_imshow(vis_frame)
    
    # Performance summary
    print(f"\n✅ STAGE 2 COMPLETED")
    print("=" * 40)
    print(f"📊 Performance Summary:")
    print(f"   • Frame processed: {frame_number}")
    print(f"   • Persons processed: {len(frame_bboxes)}")
    if inference_times:
        print(f"   • Average ViTPose inference time: {np.mean(inference_times)*1000:.1f}ms per person")
        print(f"   • Total ViTPose time: {np.sum(inference_times)*1000:.1f}ms")
    print(f"   • Output saved to: {output_image}")

def main():
    """Main function to run both stages"""
    print("🎯 Easy Pose Pipeline - Two Stage Demo")
    print("=" * 50)
    
    # Run Stage 1
    bboxes_data = stage1_detection()
    
    # Run Stage 2 with a random frame
    if bboxes_data:
        stage2_pose_estimation(bboxes_data)

if __name__ == "__main__":
    main()
