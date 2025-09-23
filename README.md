# 🔍 Brand Eye - AI-Powered Brand Detection Suite

**A comprehensive YOLO-based brand detection system with professional GUI applications for real-world deployment.**

## 🌟 Complete Solution Overview

Brand Eye combines cutting-edge YOLO object detection models with intuitive PyQt6 GUI applications, providing a complete brand recognition solution from training to deployment.

**🎯 Core Components:**
- **YOLO Model**: Custom-trained brand detection model
- **Image GUI**: Professional single-image analysis application  
- **Video GUI**: Advanced video processing with smart object tracking
- **GPU Acceleration**: Automatic CUDA optimization for maximum performance

---

## � Quick Start Guide

### 📦 Installation

1. **Clone Repository**
```bash
git clone <repository-url>
cd brand-eye
```

2. **Automated Setup** (Recommended)
```bash
python setup.py
```

3. **Manual Setup**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 🎮 Launch Applications


**Command Line:**
```bash
python main_gui.py    # Image GUI
python video_gui.py   # Video GUI
```

### 📁 Model Setup
Place your trained YOLO model at:
```
model/best.pt
```

---

## 💻 System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 1GB storage

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- CUDA-compatible GPU
- SSD storage for video processing

**GPU Acceleration:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- 5-10x performance improvement

---

### 📈 Training Metrics

![Training Metrics](images/metrics.png)

---

## 🖥️ GUI Applications

### 🖼️ Image Detection GUI (`main_gui.py`)

**Professional single-image brand analysis with real-time processing**

**Key Features:**
- **📂 Smart File Selection**: Support for JPG, PNG, BMP, TIFF formats
- **🔍 One-Click Detection**: Instant YOLO inference with visual feedback
- **💾 Export Annotations**: Save processed images with bounding boxes
- **⚙️ Device Selection**: Automatic GPU/CPU detection and switching
- **📊 Detailed Results**: Confidence scores, coordinates, and object dimensions

**Interface Highlights:**
- Side-by-side original and processed image display
- Real-time confidence scoring and classification results
- Modern PyQt6 design with intuitive controls
- Progress indicators and status monitoring

### 🎬 Video Processing GUI (`video_gui.py`)

**Advanced video analysis with intelligent object tracking system**

**Revolutionary Features:**
- **🧠 Smart Tracking**: IoU-based temporal filtering eliminates duplicate counting
- **📹 Multi-Format Support**: MP4, AVI, MOV, MKV, WMV, FLV processing
- **⚡ Real-Time Preview**: Live frame processing with progress monitoring  
- **📊 Brand Analytics**: Comprehensive counting and statistical analysis
- **🎯 Unique Detection**: Same object tracked across frames, counted only once
- **💾 Video Export**: Save annotated videos with detection overlays

**Smart Tracking Technology:**
```python
# Problem Solved: Car passing for 5 seconds = 1 count (not 150 frames)
ObjectTracker(
    max_distance=150,      # Movement tolerance
    max_frames_missing=30, # Temporal persistence  
    iou_threshold=0.3     # Overlap matching
)
```

**Performance Metrics:**
- **CPU Processing**: Reliable baseline performance
- **GPU Acceleration**: 5-10x speed improvement with CUDA
- **Memory Efficient**: Optimized for large video files
- **Real-Time Analytics**: Live brand counting and statistics

---

### 🎯 Model Accuracy

![Confusion Matrix](images/confusion_matrix_normalized.png)

### 📋 Validation Examples

| Ground Truth Labels | Model Predictions |
|:-------------------:|:-----------------:|
| ![Labels](images/val_batch0_labels.jpg) | ![Predictions](images/val_batch0_pred.jpg) |
| *Ground truth annotations* | *Model predictions with confidence scores* |

## 🎬 Demo Results

See the model in action! The images above show:

- **Training Metrics**: Comprehensive performance charts showing loss curves, precision, recall, and mAP scores over 50 epochs
- **Confusion Matrix**: Normalized confusion matrix displaying model accuracy across different brand classes
- **Validation Examples**: Side-by-side comparison of ground truth labels vs. model predictions on validation data

The model demonstrates excellent performance with:
- ✅ High precision in brand detection
- ✅ Strong recall across different brand categories  
- ✅ Accurate bounding box predictions
- ✅ Reliable confidence scoring

## 🚀 Quick Start

### Installation

```bash
pip install ultralytics huggingface_hub torch torchvision opencv-python pillow
```

### Download Model from Hugging Face

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download the trained model
model_path = hf_hub_download(
    repo_id="haydarkadioglu/brand-eye",  
    filename="brandeye.pt"
)

# Load the model
model = YOLO(model_path)
```

### Basic Usage

#### Single Image Detection

```python
import cv2
from PIL import Image

def detect_brands(image_path, conf_threshold=0.25):
    """
    Detect brands in a single image
    
    Args:
        image_path (str): Path to the image file
        conf_threshold (float): Confidence threshold (0.0-1.0)
    
    Returns:
        results: Detection results with bounding boxes and labels
    """
    results = model(image_path, conf=conf_threshold)
    
    # Display results
    results[0].show()
    
    # Get detection details
    boxes = results[0].boxes
    if boxes is not None:
        print(f"Found {len(boxes)} brand detections:")
        for box in boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = model.names[cls]
            print(f"  - {class_name}: {conf:.3f} confidence")
    
    return results

# Example usage
results = detect_brands("path/to/your/image.jpg")
```

#### Batch Processing

```python
import os
from pathlib import Path

def process_folder(input_folder, output_folder="results", conf=0.25):
    """
    Process all images in a folder
    
    Args:
        input_folder (str): Path to folder containing images
        output_folder (str): Path to save results
        conf (float): Confidence threshold
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            print(f"Processing {img_file.name}...")
            
            # Run detection
            results = model(str(img_file), conf=conf)
            
            # Save annotated image
            save_path = output_path / f"detected_{img_file.name}"
            results[0].save(str(save_path))
            
            # Print summary
            boxes = results[0].boxes
            if boxes is not None:
                print(f"  ✅ Found {len(boxes)} brands")
            else:
                print(f"  ❌ No brands detected")

# Example usage
process_folder("input_images/", "detection_results/")
```

#### Real-time Detection (Webcam)

```python
import cv2

def real_time_detection():
    """
    Real-time brand detection using webcam
    """
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.3)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display frame
        cv2.imshow('Brand Detection', annotated_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
# real_time_detection()
```


## 📁 Project Structure

```
brand-eye/
├── README.md                          # This file
├── visualize.ipynb                     # Training results visualization
├── brandeye.pt                        # Trained model weights
├── model/
│   ├── train/
│   │   ├── results.csv               # Training metrics
│   │   ├── weights/
│   │   │   └── last.pt               # Final model checkpoint
│   │   └── *.png                     # Training plots
│   └── val/
│       ├── predictions.json          # Validation predictions
│       └── *.png                     # Validation visualizations
├── confusion_matrix_normalized.png    # Model confusion matrix
└── val_batch*_*.jpg                  # Validation batch examples
```

## 🎯 Use Cases

- **Brand Monitoring**: Track brand presence in social media images
- **Market Research**: Analyze brand visibility in retail environments  
- **Advertising Analysis**: Measure brand exposure in marketing materials
- **Quality Control**: Verify proper brand logo placement
- **Content Moderation**: Detect unauthorized brand usage

## 🔧 Advanced Usage

### Custom Confidence Thresholds

```python
# High precision (fewer false positives)
results_high_conf = model("image.jpg", conf=0.7)

# High recall (catch more brands, may include false positives)
results_low_conf = model("image.jpg", conf=0.1)
```

### Export to Different Formats

```python
# Export detections to JSON
import json

def export_detections(image_path, output_json):
    results = model(image_path)
    detections = []
    
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            detection = {
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            }
            detections.append(detection)
    
    with open(output_json, 'w') as f:
        json.dump(detections, f, indent=2)

export_detections("image.jpg", "detections.json")
```

---

## 🖥️ GUI Applications - User Guide

### 🎯 Why Use GUI Applications?

**Professional Deployment:** Transform your YOLO model into production-ready applications with intuitive interfaces.

**Key Benefits:**
- **🚀 No Coding Required**: Point-and-click brand detection
- **📊 Real-Time Analytics**: Live statistics and progress monitoring  
- **💾 Export Ready**: Professional-quality outputs
- **⚡ GPU Accelerated**: Automatic hardware optimization
- **🧠 Smart Features**: Advanced tracking and filtering

### 🖼️ Image Detection GUI Features

**Professional Image Analysis:**
```python
# What the GUI does behind the scenes:
model = YOLO("model/best.pt")
results = model(image_path, device="cuda" if available else "cpu")
annotated_image = results[0].plot()
```

**Interface Capabilities:**
- **Side-by-Side Display**: Original vs processed images
- **Confidence Filtering**: Adjustable detection thresholds
- **Export Options**: Save annotated images in multiple formats
- **Batch Processing**: Handle multiple images efficiently
- **Real-Time Feedback**: Processing status and error handling

### 🎬 Video Processing GUI - Advanced Features

**Revolutionary Smart Tracking:**
```python
# Solves the duplicate counting problem
class ObjectTracker:
    """
    Problem: Car appears in 150 frames = counted 150 times ❌
    Solution: Car tracked across frames = counted 1 time ✅
    """
    def update_tracks(self, detections, frame_number):
        # IoU-based matching
        # Temporal persistence  
        # Unique object counting
```

**Advanced Analytics Dashboard:**
- **Real-Time Counting**: Live brand statistics as video processes
- **Progress Monitoring**: Frame-by-frame processing with ETA
- **Export Controls**: Save processed videos with annotations
- **Performance Metrics**: Processing speed and detection rates

### 🛠️ GUI Technical Implementation

**Architecture:**
```python
# Multi-threaded processing
class VideoProcessingThread(QThread):
    def run(self):
        # Background processing
        # UI remains responsive
        # Real-time updates via signals
        
# GPU/CPU Selection
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
```

**Performance Optimization:**
- **Threading**: Non-blocking UI with background processing
- **Memory Management**: Efficient handling of large video files  
- **Batch Processing**: Optimized frame-by-frame analysis
- **Error Recovery**: Robust error handling and user feedback

---

## 🎮 Complete Usage Workflow

### 1. **Setup** (One-time)
```bash
python setup.py  # Automated installation
```

### 2. **Image Analysis**
```bash
python main_gui.py
# 1. Select image → 2. Choose device → 3. Detect → 4. Save results
```

### 3. **Video Processing** 
```bash
python video_gui.py  
# 1. Select video → 2. Enable smart tracking → 3. Process → 4. Export
```

### 4. **Results Analysis**
- **Confidence Scores**: Validate detection quality
- **Bounding Boxes**: Precise object localization
- **Brand Statistics**: Comprehensive counting and analysis
- **Export Options**: Professional-quality outputs

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Hosted on [Hugging Face Hub](https://huggingface.co/)
- Training visualization powered by matplotlib and seaborn

## 📞 Contact & Support

- **🔧 GUI Issues**: [GitHub Issues](https://github.com/your-username/brand-eye/issues)
- **📈 Model Performance**: [Hugging Face Discussions](https://huggingface.co/haydarkadioglu/brand-eye/discussions)  
- **💡 Feature Requests**: Create detailed GitHub issues with use cases
- **🤝 Collaboration**: Open to partnerships and contributions

## 🏆 Project Highlights

**🎯 Complete Solution**: From model training to GUI deployment
**⚡ Performance**: GPU-accelerated processing with smart optimizations  
**🧠 Intelligence**: Advanced object tracking eliminates duplicate counting
**🎨 Professional**: Modern PyQt6 interface with export capabilities
**🌐 Accessible**: No coding required for end-users

---

**🚀 Ready to revolutionize your brand detection workflow? Get started with Brand Eye today!**

*Made with ❤️ and cutting-edge AI for professional brand recognition*