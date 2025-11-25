# Easy Pose Pipeline
A streamlined pipeline for 2D human pose estimation using YOLOv8 for detection and ViTPose for pose estimation.

## 🚀 Features
- **Stage 1**: YOLOv8 for fast and accurate person detection
- **Stage 2**: ViTPose-B for high-quality 2D pose estimation
- **Optimized**: Two-stage processing for maximum efficiency
- **Easy to Use**: Simple API and demo scripts

## 📦 Installation

Clone this repository:
```bash
git clone https://github.com/pradeepj247/easy-pose-pipeline.git
cd easy-pose-pipeline
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download ViTPose-B model:
```bash
python download_model.py
```

Additional packages for Google Colab:
```bash
pip install ffmpeg-python
```

Verify installation:
```bash
python verify_installation.py
```

## 🎯 Quick Start

### Two-Stage Demo

**Stage 1 - Person Detection**:
```bash
python demos/stage1_yolo_test.py
```
Processes video frames with YOLOv8 and exports bounding boxes to JSON.

**Stage 2 - Pose Estimation**:
```bash
# Single image test
python demos/stage2_image_test.py --frame_number 254

# Video processing
python demos/stage2_video_test.py --frames=100 --start_frame=0
```
Uses pre-computed bounding boxes to run ViTPose pose estimation.

## 📁 Project Structure
```
easy_pose_pipeline/
├── demos/                 # Demo scripts
│   ├── stage1_yolo_test.py     # Stage 1: YOLOv8 detection
│   ├── stage2_image_test.py    # Stage 2: ViTPose pose estimation (single image test)
│   ├── stage2_video_test.py    # Stage 2: ViTPose video processing
│   └── outputs/            # Output directory for results
├── models/                # Model weights directory
├── data/                  # Input data directory
├── easy_ViTPose/          # ViTPose implementation
└── utils/                 # Utility functions
```

## 🔧 Model Setup
**Note**: There are two models directories in the project:
- `models/` (root) - Contains our working models (`vitpose-b.pth`, `yolov8s.pt`)
- `easy_ViTPose/models/` - Contains original easy_ViTPose download scripts (not used in our pipeline)

Download the required model weights:
- **YOLOv8s**: Automatically downloaded by ultralytics
- **ViTPose-B**: Use `python download_model.py` to download from GitHub releases

## 🎮 Usage Examples

### Basic Two-Stage Pipeline
```bash
# Stage 1: Run detection on all frames
python demos/stage1_yolo_test.py

# Stage 2: Run pose estimation on specific frame
python demos/stage2_image_test.py --frame_number 254

# Stage 2: Process multiple video frames
python demos/stage2_video_test.py --frames=100 --start_frame=0
```

### Output Files
- `stage1_bboxes.json`: Bounding boxes from Stage 1
- `stage2_2dkps.json`: 2D keypoints with confidence scores from Stage 2
- `stage2_video_output.mp4`: Video with pose estimation overlay

## 📊 Performance
- **YOLOv8 Detection**: 80-110 FPS
- **ViTPose Estimation**: ~10 FPS
- **Hardware**: Tested on NVIDIA T4 GPU (Google Colab)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License
This project is licensed under the MIT License.