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
wget -O models/vitpose-b.pth https://github.com/pradeepj247/easy-pose-pipeline/releases/download/v1.0/vitpose-b.pth
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
python demos/demo_stage1.py
```
Processes video frames with YOLOv8 and exports bounding boxes to JSON.

**Stage 2 - Pose Estimation**:
```bash
python demos/demo_stage2.py
```
Uses pre-computed bounding boxes to run ViTPose pose estimation on a random frame.

## 📁 Project Structure
```
easy_pose_pipeline/
├── demos/                 # Demo scripts
│   ├── demo_stage1.py          # Stage 1: YOLOv8 detection
│   ├── demo_stage2_image_test.py  # Stage 2: ViTPose pose estimation (single image test)
│   └── outputs/            # Output directory for results
├── data/                 # Data directory
│   ├── input/           # Input videos/images
│   └── output/          # Processed outputs
├── models/              # Model weights
├── src/                 # Source code
├── utils/               # Utility functions
├── configs/             # Configuration files
├── requirements.txt     # Python dependencies
└── verify_installation.py  # Installation checker
```

## 🔧 Model Setup
**Note**: There are two models directories in the project:
- `models/` (root) - Contains our working models (`vitpose-b.pth`, `yolov8s.pt`)
- `easy_ViTPose/models/` - Contains original easy_ViTPose download scripts (not used in our pipeline)



Download the required model weights:
- **YOLOv8s**: Automatically downloaded by ultralytics
- **ViTPose-B**: Download from GitHub releases and place in `models/vitpose-b.pth`

## 📊 Performance
- **YOLOv8s detection**: ~8-12ms per frame
- **ViTPose-B inference**: ~100ms per person
- **Total pipeline**: Efficient two-stage processing
- **Estimated FPS**: 80-110 FPS for detection, ~10 FPS for pose estimation

## 🎮 Usage Examples

### Basic Two-Stage Pipeline
```python
# Stage 1: Run detection on all frames
python demos/demo_stage1.py

# Stage 2: Run pose estimation on specific frame
python demos/demo_stage2_image_test.py --frame_number 254
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License
This project is licensed under the MIT License.

## 🙏 Acknowledgments
- Ultralytics YOLOv8
- Easy-ViTPose
- ViTPose

## 🔧 Model Download
The ViTPose-B model is hosted on GitHub Releases and will be downloaded to the `models/` directory.

### Manual Download
If automatic download fails, you can manually:
- Download from [GitHub Releases](https://github.com/pradeepj247/easy-pose-pipeline/releases/download/v1.0/vitpose-b.pth)
- Place the model in `models/vitpose-b.pth`
