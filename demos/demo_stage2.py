"""
Demo Stage 2: ViTPose Pose Estimation using Pre-computed BBoxes
"""

import sys
import os
import json
import cv2
import numpy as np
import time
from google.colab.patches import cv2_imshow

# Add local easy_ViTPose to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "easy_ViTPose"))
from easy_ViTPose.pose_only import VitPoseOnly

def stage2_pose_estimation(frame_number=None):
    """Stage 2: Run ViTPose pose estimation using pre-computed bboxes"""
    print("🚀 STAGE 2: ViTPose Pose Estimation")
    print("=" * 40)
    
    # Load bboxes from Stage 1
    bboxes_path = "stage1_bboxes.json"
    if not os.path.exists(bboxes_path):
        print(f"❌ Bboxes file not found: {bboxes_path}")
        print("💡 Please run Stage 1 first to generate bboxes")
        return
    
    with open(bboxes_path, 'r') as f:
        bboxes_data = json.load(f)
    
    print(f"✅ Loaded bboxes for {len(bboxes_data)} frames")
    
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
        available_frames = [int(f) for f in bboxes_data.keys() if bboxes_data[f]]
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
    frame_bboxes = bboxes_data.get(str(frame_number), [])
    
    if not frame_bboxes:
        print(f"❌ No bboxes found for frame {frame_number}")
        return
    
    print(f"✅ Found {len(frame_bboxes)} person(s) in frame")
    
    # Process each bbox with ViTPose
    vis_frame = frame.copy()
    inference_times = []
    all_keypoints = []
    
    for i, bbox_info in enumerate(frame_bboxes):
        bbox = bbox_info["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Draw bbox
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Person {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ViTPose inference
        inference_start = time.time()
        keypoints = vitpose_model.inference_bbox(frame, bbox)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        # Store keypoints
        all_keypoints.append(keypoints)
        
        # Draw keypoints
        if keypoints.size > 0:
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

if __name__ == "__main__":
    print("🎯 Easy Pose Pipeline - Stage 2 Only")
    print("=" * 50)
    
    # You can specify a frame number here, or leave as None for random frame
    # stage2_pose_estimation(frame_number=254)  # Specific frame
    stage2_pose_estimation()  # Random frame
