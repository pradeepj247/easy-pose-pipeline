# Easy Pose Pipeline

A streamlined pipeline for 2D human pose estimation using YOLOv8 for detection and ViTPose for pose estimation.

## 🚀 Features

- **Stage 1**: YOLOv8 for fast and accurate person detection
- **Stage 2**: ViTPose-B for high-quality 2D pose estimation
- **Optimized**: Processes only the largest detected person for efficiency
- **Easy to Use**: Simple API and demo scripts

## 📦 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd easy_pose_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python verify_installation.py
```

## 🎯 Quick Start

Run the basic demo:
```bash
python demos/basic_demo.py
```

## 📁 Project Structure

```
easy_pose_pipeline/
├── demos/              # Demo scripts
│   └── basic_demo.py   # Basic pose estimation demo
├── data/               # Data directory
│   ├── input/          # Input videos/images
│   └── output/         # Processed outputs
├── models/             # Model weights (download separately)
├── src/                # Source code
├── utils/              # Utility functions
├── configs/            # Configuration files
├── requirements.txt    # Python dependencies
└── verify_installation.py  # Installation checker
```

## 🔧 Model Setup

Download the required model weights:

1. **YOLOv8s**: Automatically downloaded by ultralytics
2. **ViTPose-B**: Download from [official repository](https://github.com/JunkyByte/easy_ViTPose)
   - Place in `models/vitpose-b.pth`

## 📊 Performance

- **YOLOv8s detection**: ~10-15ms per frame
- **ViTPose-B inference**: ~20-30ms per frame
- **Total pipeline**: ~30-45ms per frame
- **Estimated FPS**: 20-30 FPS on CUDA

## 🎮 Usage Examples

### Basic Pipeline Usage
```python
from src.pose_pipeline import PosePipeline

# Initialize pipeline
pipeline = PosePipeline(
    yolo_model_path="models/yolov8s.pt",
    vitpose_model_path="models/vitpose-b.pth"
)

# Process frame
keypoints, yolo_time, vitpose_time = pipeline.process_frame(frame)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Easy-ViTPose](https://github.com/JunkyByte/easy_ViTPose)
- [ViTPose](https://github.com/ViTAE-3D/ViTPose)

## 📄 License

This project is licensed under the MIT License.

**Note**: The `easy_ViTPose/` directory contains code from [JunkyByte/easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) which is also MIT licensed.
