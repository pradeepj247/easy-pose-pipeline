"""
Demo Stage 2 Video: ViTPose Pose Estimation on Multiple Frames
Processes multiple frames and generates output video with pose estimation
"""

import sys
import os
import json
import cv2
import numpy as np
import time
import argparse
from google.colab.patches import cv2_imshow

# Add local easy_ViTPose to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "easy_ViTPose"))
from easy_ViTPose.pose_only import VitPoseOnly

def parse_arguments():
    parser = argparse.ArgumentParser(description='ViTPose Video Processing')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to process')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame number')
    return parser.parse_args()

def stage2_video_processing(num_frames=100, start_frame=0):
    """Stage 2: Run ViTPose pose estimation on multiple frames and generate video"""
    print("🚀 STAGE 2 VIDEO: ViTPose Pose Estimation on Multiple Frames")
    print("=" * 60)
    
    # Load bboxes from Stage 1
    bboxes_path = os.path.join(os.path.dirname(__file__), "outputs", "stage1_bboxes.json")
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
    
    # Warm-up run to initialize GPU and cache
    print("🔥 Warming up ViTPose model...")
    warm_up_start = time.time()
    dummy_image = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
    dummy_bbox = [0, 0, 192, 256]
    _ = vitpose_model.inference_bbox(dummy_image, dummy_bbox)
    warm_up_time = time.time() - warm_up_start
    print(f"✅ ViTPose warm-up completed in {warm_up_time:.2f}s")
    
    # Open video
    video_path = os.path.join(os.path.dirname(__file__), "..", "data", "input", "campus_walk.mp4")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open video file")
        return
    
    # Get video properties for output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    output_video_path = os.path.join(os.path.dirname(__file__), "outputs", "stage2_video_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"📹 Processing {num_frames} frames starting from frame {start_frame}")
    print(f"🎬 Output video: {output_video_path}")
    print(f"   • Resolution: {width}x{height}")
    print(f"   • FPS: {fps}")
    
    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_times = []
    frame_count = 0
    processed_count = 0
    keypoints_data = {}  # Store keypoints for each frame
    
    print("\n🔍 Processing frames...")
    start_time = time.time()
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = start_frame + frame_count
        
        # Get bboxes for this frame
        frame_bboxes = bboxes_data.get(str(current_frame), [])
        
        vis_frame = frame.copy()
        frame_keypoints = []
        
        if frame_bboxes:
            # Get the largest bbox (by area)
            largest_bbox_info = max(frame_bboxes, key=lambda x: x["area"])
            bbox = largest_bbox_info["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Draw bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Person (Largest)", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # ViTPose inference on largest bbox
            inference_start = time.time()
            keypoints = vitpose_model.inference_bbox(frame, bbox)
            inference_time = time.time() - inference_start
            frame_times.append(inference_time)
            
            # Store keypoints data
            if keypoints.size > 0:
                for kp in keypoints:
                    y_coord, x_coord, conf = kp
                    frame_keypoints.append({
                        "x": float(x_coord),
                        "y": float(y_coord), 
                        "confidence": float(conf)
                    })
            
            # Draw keypoints
            if keypoints.size > 0:
                for kp in keypoints:
                    y, x, conf = kp
                    if conf > 0.3:
                        cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            
            processed_count += 1
        
        # Store keypoints for this frame
        keypoints_data[str(current_frame)] = frame_keypoints
        
        # Write frame to output video
        out.write(vis_frame)
        
        # Progress reporting every 50 frames
        if frame_count % 50 == 0 and frame_count > 0:
            avg_time = np.mean(frame_times[-50:]) if len(frame_times) >= 50 else np.mean(frame_times)
            current_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"   Frame {frame_count}/{num_frames} - Avg: {avg_time*1000:.1f}ms, FPS: {current_fps:.1f}")
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Save keypoints to JSON file
    keypoints_output_path = os.path.join(os.path.dirname(__file__), "outputs", "stage2_2dkps.json")
    with open(keypoints_output_path, 'w') as f:
        json.dump(keypoints_data, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Performance summary
    print(f"\n✅ STAGE 2 VIDEO COMPLETED")
    print("=" * 60)
    print(f"📊 Performance Summary:")
    print(f"   • Frames processed: {frame_count}")
    print(f"   • Persons processed: {processed_count}")
    print(f"   • Total time: {total_time:.2f}s")
    if frame_times:
        print(f"   • Average time per frame: {np.mean(frame_times)*1000:.1f}ms")
        print(f"   • Average FPS: {1.0/np.mean(frame_times):.1f}")
    print(f"   • Output video: {output_video_path}")
    print(f"   • 2D Keypoints JSON: {keypoints_output_path}")

if __name__ == "__main__":
    print("🎯 Easy Pose Pipeline - Stage 2 Video Processing")
    print("=" * 70)
    
    args = parse_arguments()
    stage2_video_processing(num_frames=args.frames, start_frame=args.start_frame)