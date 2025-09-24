import sys
import os
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict, Counter
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QScrollArea, QFrame, QProgressBar, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QComboBox, QCheckBox, QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QFont
from ultralytics import YOLO
import torch
import math

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ModelDownloadThread(QThread):
    """Thread for downloading model from Hugging Face without blocking the UI"""
    download_complete = pyqtSignal(str)  # model_path
    download_progress = pyqtSignal(str)  # progress_message
    error_occurred = pyqtSignal(str)
    
    def __init__(self, repo_id="haydarkadioglu/brand-eye", filename="brandeye.pt", local_dir="model"):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        self.local_dir = local_dir
    
    def run(self):
        try:
            self.download_progress.emit("Connecting to Hugging Face...")
            
            if not HF_AVAILABLE:
                raise ImportError("huggingface_hub is not installed. Please install with: pip install huggingface_hub")
            
            self.download_progress.emit("Downloading model from Hugging Face...")
            
            # Create directory if it doesn't exist
            os.makedirs(self.local_dir, exist_ok=True)
            
            # Download model
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=self.local_dir,
                local_dir_use_symlinks=False
            )
            
            self.download_complete.emit(model_path)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class ObjectTracker:
    """Simple object tracker to avoid counting the same object multiple times"""
    
    def __init__(self, max_distance=100, max_frames_missing=30):
        self.tracked_objects = {}  # track_id -> {bbox, class_name, last_frame, center}
        self.next_track_id = 1
        self.max_distance = max_distance  # Maximum distance for object matching
        self.max_frames_missing = max_frames_missing  # Frames to keep track after disappearing
        self.counted_objects = set()  # Set of track_ids that were already counted
    
    def calculate_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between centers of two bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        # Get intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracks(self, detections, frame_number):
        """
        Update tracks with new detections
        detections: list of {'bbox': [x1,y1,x2,y2], 'class_name': str, 'confidence': float}
        Returns: list of track_ids for new unique objects
        """
        new_objects = []
        matched_tracks = set()
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track_info in self.tracked_objects.items():
            if frame_number - track_info['last_frame'] > self.max_frames_missing:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
        
        # Match detections with existing tracks
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            best_match = None
            best_score = 0
            
            # Find best matching existing track
            for track_id, track_info in self.tracked_objects.items():
                if track_info['class_name'] != class_name:
                    continue
                
                # Calculate similarity (combination of IoU and distance)
                iou = self.calculate_iou(bbox, track_info['bbox'])
                distance = self.calculate_distance(bbox, track_info['bbox'])
                
                # Weighted score: prioritize IoU, but also consider distance
                score = iou * 0.7 + (1.0 - min(distance / self.max_distance, 1.0)) * 0.3
                
                if score > best_score and score > 0.3:  # Minimum threshold for matching
                    best_score = score
                    best_match = track_id
            
            if best_match:
                # Update existing track
                self.tracked_objects[best_match].update({
                    'bbox': bbox,
                    'last_frame': frame_number,
                    'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                })
                matched_tracks.add(best_match)
            else:
                # Create new track for new object
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracked_objects[track_id] = {
                    'bbox': bbox,
                    'class_name': class_name,
                    'last_frame': frame_number,
                    'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    'first_frame': frame_number,
                    'confidence': confidence
                }
                
                new_objects.append(track_id)
                self.counted_objects.add(track_id)
        
        return new_objects
    
    def get_unique_count_by_class(self):
        """Get count of unique objects by class"""
        class_counts = defaultdict(int)
        for track_id in self.counted_objects:
            if track_id in self.tracked_objects:
                class_name = self.tracked_objects[track_id]['class_name']
                class_counts[class_name] += 1
        return dict(class_counts)


class VideoProcessingThread(QThread):
    """Thread for processing video without blocking the UI"""
    frame_processed = pyqtSignal(int, int)  # current_frame, total_frames
    detection_update = pyqtSignal(object)  # brand_counts dictionary
    processing_complete = pyqtSignal(object, str)  # final_results, output_video_path
    error_occurred = pyqtSignal(str)
    preview_frame = pyqtSignal(object)  # processed frame for preview
    
    def __init__(self, model_path, video_path, output_path=None, preview_enabled=True, device='auto', enable_tracking=True):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.preview_enabled = preview_enabled
        self.device = device
        self.enable_tracking = enable_tracking
        self.should_stop = False
    
    def stop_processing(self):
        self.should_stop = True
    
    def run(self):
        try:
            # Load YOLO model
            model = YOLO(self.model_path)
            
            # Set device
            if self.device != 'auto':
                model.to(self.device)
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer if output path is provided
            out = None
            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            # Initialize brand counting
            if self.enable_tracking:
                tracker = ObjectTracker(max_distance=150, max_frames_missing=30)
                unique_brand_counts = defaultdict(int)
            else:
                unique_brand_counts = defaultdict(int)
            
            total_detections_count = defaultdict(int)  # Frame-by-frame count
            frame_detections = []
            current_frame = 0
            
            while True:
                if self.should_stop:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame += 1
                
                # Run YOLO detection on frame
                results = model(frame, verbose=False, device=self.device)
                
                # Process detections
                if results[0].boxes is not None:
                    current_detections = []
                    
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = results[0].names[class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        
                        # Count all detections for statistics
                        if confidence > 0.5:
                            total_detections_count[class_name] += 1
                            
                            # Prepare detection for tracking
                            current_detections.append({
                                'bbox': bbox,
                                'class_name': class_name,
                                'confidence': confidence
                            })
                            
                            # Store detection info
                            frame_detections.append({
                                'frame': current_frame,
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': bbox
                            })
                    
                    # Update tracking and get unique counts
                    if self.enable_tracking and current_detections:
                        new_objects = tracker.update_tracks(current_detections, current_frame)
                        unique_brand_counts = tracker.get_unique_count_by_class()
                    else:
                        # Without tracking, count every detection (old behavior)
                        for detection in current_detections:
                            unique_brand_counts[detection['class_name']] += 1
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Write frame to output video if enabled
                if out is not None:
                    out.write(annotated_frame)
                
                # Send preview frame (every 10th frame to avoid overwhelming UI)
                if self.preview_enabled and current_frame % 10 == 0:
                    # Convert BGR to RGB for display
                    preview_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    self.preview_frame.emit(preview_frame_rgb)
                
                # Update progress
                self.frame_processed.emit(current_frame, total_frames)
                
                # Update brand counts every 30 frames
                if current_frame % 30 == 0:
                    self.detection_update.emit(dict(unique_brand_counts))
            
            # Clean up
            cap.release()
            if out is not None:
                out.release()
            
            # Prepare final results
            final_results = {
                'brand_counts': dict(unique_brand_counts),
                'total_detections_count': dict(total_detections_count),
                'total_frames': total_frames,
                'total_detections': len(frame_detections),
                'detections': frame_detections,
                'tracking_enabled': self.enable_tracking,
                'video_info': {
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'duration': total_frames / fps
                }
            }
            
            self.processing_complete.emit(final_results, self.output_path or "")
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = "model/best.pt"
        self.current_video_path = None
        self.processing_thread = None
        self.brand_counts = defaultdict(int)
        self.output_video_path = None
        self.available_devices = self.get_available_devices()
        self.current_device = self.available_devices[0] if self.available_devices else 'cpu'
        self.enable_tracking = True  # Default to enabled
        
        self.init_ui()
        self.check_model_exists()
    
    def get_available_devices(self):
        """Get list of available devices (CPU, CUDA GPUs)"""
        devices = ['cpu']
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                devices.append(f'cuda:{i}')
        
        return devices
    
    def get_device_display_name(self, device):
        """Get user-friendly device name"""
        if device == 'cpu':
            return "CPU"
        elif device.startswith('cuda:'):
            gpu_id = device.split(':')[1]
            try:
                gpu_name = torch.cuda.get_device_name(int(gpu_id))
                return f"GPU {gpu_id}: {gpu_name}"
            except:
                return f"GPU {gpu_id}"
        return device
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("YOLO Video Processing - Brand Counter")
        self.setGeometry(50, 50, 1600, 1000)
        self.setMinimumSize(1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Video preview and info
        left_widget = self.create_video_section()
        splitter.addWidget(left_widget)
        
        # Right side - Brand counts table
        right_widget = self.create_results_section()
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 2)  # Video section gets 2/3
        splitter.setStretchFactor(1, 1)  # Results section gets 1/3
        
        main_layout.addWidget(splitter)
        
        # Status bar at bottom
        self.status_label = QLabel("Ready to process video")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #1a1a1a;
                font-size: 13pt;
                font-weight: bold;
                padding: 12px;
                background-color: #ffffff;
                border: 2px solid #2196F3;
                border-radius: 6px;
            }
        """)
        main_layout.addWidget(self.status_label)
    
    def create_control_panel(self):
        """Create the control panel with buttons"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setMaximumHeight(90)
        
        layout = QHBoxLayout(frame)
        
        # Select video button
        self.select_video_button = QPushButton("📹 Select Video")
        self.select_video_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 13pt;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.select_video_button.clicked.connect(self.select_video)
        layout.addWidget(self.select_video_button)
        
        # Process video button
        self.process_button = QPushButton("⚡ Process Video")
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 13pt;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)
        
        # Save output video button
        self.save_video_button = QPushButton("💾 Save Processed Video")
        self.save_video_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 13pt;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #EF6C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.save_video_button.clicked.connect(self.save_processed_video)
        self.save_video_button.setEnabled(False)
        layout.addWidget(self.save_video_button)
        
        # Stop processing button
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 13pt;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Device selection
        device_label = QLabel("Device:")
        device_label.setStyleSheet("""
            QLabel {
                color: #1a1a1a;
                font-size: 12pt;
                font-weight: bold;
                padding: 5px;
            }
        """)
        layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        for device in self.available_devices:
            display_name = self.get_device_display_name(device)
            self.device_combo.addItem(display_name, device)
        
        self.device_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 8px;
                font-size: 11pt;
                font-weight: bold;
                color: #1a1a1a;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
                background: #4CAF50;
                border-radius: 3px;
            }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid white;
                width: 6px;
                height: 6px;
                border-top: none;
                border-right: none;
                transform: rotate(-45deg);
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 2px solid #4CAF50;
                selection-background-color: #e8f5e8;
            }
        """)
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        layout.addWidget(self.device_combo)
        
        # Object tracking checkbox
        self.tracking_checkbox = QCheckBox("Smart Tracking (Avoid Double Counting)")
        self.tracking_checkbox.setChecked(True)
        self.tracking_checkbox.setStyleSheet("""
            QCheckBox {
                color: #1a1a1a;
                font-size: 11pt;
                font-weight: bold;
                padding: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #4CAF50;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
            }
            QCheckBox::indicator:checked:after {
                content: "✓";
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        self.tracking_checkbox.toggled.connect(self.on_tracking_changed)
        layout.addWidget(self.tracking_checkbox)
        
        layout.addStretch()
        return frame
    
    def on_tracking_changed(self, checked):
        """Handle tracking option change"""
        self.enable_tracking = checked
        if checked:
            self.status_label.setText("Smart tracking enabled - objects will be counted once")
        else:
            self.status_label.setText("Smart tracking disabled - objects counted per frame")
    
    def on_device_changed(self):
        """Handle device selection change"""
        current_index = self.device_combo.currentIndex()
        if current_index >= 0:
            self.current_device = self.device_combo.itemData(current_index)
            device_name = self.get_device_display_name(self.current_device)
            self.status_label.setText(f"Device changed to: {device_name}")
    
    def create_video_section(self):
        """Create video preview and info section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Video info label
        self.video_info_label = QLabel("No video selected")
        self.video_info_label.setStyleSheet("""
            QLabel {
                font-size: 13pt;
                font-weight: bold;
                color: #1a1a1a;
                padding: 12px;
                background-color: #ffffff;
                border: 2px solid #4CAF50;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.video_info_label)
        
        # Video preview frame
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout(preview_frame)
        
        preview_title = QLabel("Video Preview")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_title.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #1a1a1a;
                padding: 10px;
                background-color: #e3f2fd;
                border-radius: 5px;
            }
        """)
        preview_layout.addWidget(preview_title)
        
        # Preview image area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 2px dashed #2196F3;
                color: #1a1a1a;
                min-height: 300px;
                font-size: 12pt;
                font-weight: bold;
            }
        """)
        self.preview_label.setText("Video frames will be displayed here during processing")
        
        scroll_area.setWidget(self.preview_label)
        preview_layout.addWidget(scroll_area)
        
        layout.addWidget(preview_frame)
        return widget
    
    def create_results_section(self):
        """Create brand counting results section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Results title
        results_title = QLabel("Brand Detection Results")
        results_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_title.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #1a1a1a;
                padding: 12px;
                background-color: #c8e6c9;
                border: 2px solid #4CAF50;
                border-radius: 6px;
            }
        """)
        layout.addWidget(results_title)
        
        # Brand counts table
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Brand", "Count"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 2px solid #4CAF50;
                border-radius: 6px;
                background-color: white;
                gridline-color: #e0e0e0;
                font-size: 11pt;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #e0e0e0;
                color: #1a1a1a;
            }
            QTableWidget::item:selected {
                background-color: #e8f5e8;
                color: #1a1a1a;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border: none;
                font-weight: bold;
                font-size: 12pt;
            }
        """)
        layout.addWidget(self.results_table)
        
        # Summary text area
        summary_title = QLabel("Processing Summary")
        summary_title.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #1a1a1a;
                padding: 8px 0;
                background-color: #f0f8f0;
                border-radius: 3px;
                padding-left: 8px;
            }
        """)
        layout.addWidget(summary_title)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #4CAF50;
                border-radius: 6px;
                background-color: #ffffff;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11pt;
                color: #1a1a1a;
                padding: 10px;
                line-height: 1.4;
            }
        """)
        self.summary_text.setPlainText("Processing summary will appear here...")
        layout.addWidget(self.summary_text)
        
        return widget
    
    def check_model_exists(self):
        """Check if the YOLO model file exists, if not offer to download from Hugging Face"""
        if not os.path.exists(self.model_path):
            reply = QMessageBox.question(
                self, 
                "Model Not Found", 
                f"YOLO model file not found: {self.model_path}\n\n"
                "Would you like to download the pre-trained model from Hugging Face?\n"
                "(https://huggingface.co/haydarkadioglu/brand-eye)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.download_model_from_hf()
            else:
                QMessageBox.information(
                    self, 
                    "Manual Setup Required", 
                    "Please place your trained model file at:\n" + self.model_path
                )
    
    def download_model_from_hf(self):
        """Download model from Hugging Face"""
        if not HF_AVAILABLE:
            QMessageBox.critical(
                self, 
                "Dependency Missing", 
                "huggingface_hub is required to download the model.\n\n"
                "Please install it with:\npip install huggingface_hub\n\n"
                "Then restart the application."
            )
            return
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Downloading model from Hugging Face...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        # Start download thread
        self.download_thread = ModelDownloadThread()
        self.download_thread.download_complete.connect(self.on_download_complete)
        self.download_thread.download_progress.connect(self.on_download_progress)
        self.download_thread.error_occurred.connect(self.on_download_error)
        self.download_thread.start()
        
        # Connect cancel button
        self.progress_dialog.canceled.connect(self.download_thread.terminate)
    
    def on_download_complete(self, model_path):
        """Handle successful model download"""
        self.progress_dialog.close()
        
        # Update model path to use the downloaded model
        if os.path.exists(os.path.join("model", "brandeye.pt")):
            self.model_path = os.path.join("model", "brandeye.pt")
        elif os.path.exists(model_path):
            self.model_path = model_path
        
        QMessageBox.information(
            self, 
            "Download Complete", 
            f"Model downloaded successfully!\n\n"
            f"Saved to: {self.model_path}\n\n"
            "You can now start processing videos."
        )
        
        self.status_label.setText("Model downloaded successfully - Ready to process videos")
    
    def on_download_progress(self, message):
        """Handle download progress updates"""
        self.progress_dialog.setLabelText(message)
    
    def on_download_error(self, error_message):
        """Handle download error"""
        self.progress_dialog.close()
        
        QMessageBox.critical(
            self, 
            "Download Failed", 
            f"Failed to download model from Hugging Face:\n\n{error_message}\n\n"
            "Please check your internet connection or manually place the model file at:\n"
            f"{self.model_path}"
        )
        
        self.status_label.setText("Model download failed")
    
    def select_video(self):
        """Open file dialog to select a video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File", 
            "", 
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;MP4 files (*.mp4);;AVI files (*.avi)"
        )
        
        if file_path:
            self.current_video_path = file_path
            self.process_button.setEnabled(True)
            self.save_video_button.setEnabled(False)
            
            # Get video info
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps
                
                info_text = f"Selected: {os.path.basename(file_path)}\n"
                info_text += f"Resolution: {width}x{height} | Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {total_frames}"
                
                self.video_info_label.setText(info_text)
                cap.release()
            
            # Clear previous results
            self.brand_counts.clear()
            self.update_results_table()
            self.summary_text.setPlainText("Select 'Process Video' to start brand detection...")
            self.preview_label.setText("Video frames will be displayed here during processing")
            
            self.status_label.setText(f"Video loaded: {os.path.basename(file_path)}")
    
    def process_video(self):
        """Start video processing"""
        if not self.current_video_path:
            QMessageBox.warning(self, "Warning", "Please select a video file first.")
            return
        
        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, "Error", f"Model file not found: {self.model_path}")
            return
        
        # Setup UI for processing
        self.process_button.setEnabled(False)
        self.select_video_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        device_name = self.get_device_display_name(self.current_device)
        self.status_label.setText(f"Processing video on {device_name}...")
        
        # Clear previous results
        self.brand_counts.clear()
        self.update_results_table()
        
        # Create output path
        video_name = Path(self.current_video_path).stem
        output_dir = Path(self.current_video_path).parent
        self.output_video_path = str(output_dir / f"{video_name}_processed.mp4")
        
        # Start processing thread
        self.processing_thread = VideoProcessingThread(
            self.model_path, 
            self.current_video_path, 
            self.output_video_path,
            preview_enabled=True,
            device=self.current_device,
            enable_tracking=self.enable_tracking
        )
        
        self.processing_thread.frame_processed.connect(self.update_progress)
        self.processing_thread.detection_update.connect(self.update_brand_counts)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.preview_frame.connect(self.update_preview)
        
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop video processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop_processing()
            self.processing_thread.wait(3000)  # Wait up to 3 seconds
            
        self.reset_ui_after_processing()
        self.status_label.setText("Processing stopped by user")
    
    def update_progress(self, current_frame, total_frames):
        """Update progress bar"""
        progress = int((current_frame / total_frames) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Processing frame {current_frame}/{total_frames} ({progress}%)")
    
    def update_brand_counts(self, brand_counts):
        """Update brand counts display"""
        self.brand_counts = brand_counts
        self.update_results_table()
    
    def update_preview(self, frame_rgb):
        """Update video preview with processed frame"""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to QPixmap
            pixmap = self.pil_to_qpixmap(pil_image)
            
            # Scale to fit preview area
            scaled_pixmap = pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation)
            
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.setText("")
            
        except Exception as e:
            print(f"Preview update error: {e}")
    
    def update_results_table(self):
        """Update the results table with current brand counts"""
        self.results_table.setRowCount(len(self.brand_counts))
        
        # Sort brands by count (descending)
        sorted_brands = sorted(self.brand_counts.items(), key=lambda x: x[1], reverse=True)
        
        for row, (brand, count) in enumerate(sorted_brands):
            # Brand name
            brand_item = QTableWidgetItem(str(brand))
            brand_item.setFlags(brand_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.results_table.setItem(row, 0, brand_item)
            
            # Count
            count_item = QTableWidgetItem(str(count))
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, 1, count_item)
    
    def on_processing_complete(self, results, output_path):
        """Handle completion of video processing"""
        try:
            self.reset_ui_after_processing()
            
            # Update final results
            self.brand_counts = results['brand_counts']
            self.update_results_table()
            
            # Update summary
            summary = f"Processing Complete!\n\n"
            summary += f"Total frames processed: {results['total_frames']:,}\n"
            summary += f"Total detections: {results['total_detections']:,}\n"
            summary += f"Video duration: {results['video_info']['duration']:.1f} seconds\n"
            
            if results.get('tracking_enabled', False):
                summary += f"✓ Smart Tracking: Unique objects counted\n"
                summary += f"Unique brands detected: {len(results['brand_counts'])}\n"
                if 'total_detections_count' in results:
                    total_frame_detections = sum(results['total_detections_count'].values())
                    summary += f"Total frame detections: {total_frame_detections:,}\n"
                    summary += f"Duplicate reduction: {total_frame_detections - sum(results['brand_counts'].values()):,}\n"
            else:
                summary += f"⚠ Smart Tracking: Disabled (objects counted per frame)\n"
                summary += f"Frame-by-frame brands: {len(results['brand_counts'])}\n"
            
            summary += f"Average detections per second: {results['total_detections']/results['video_info']['duration']:.1f}\n\n"
            
            if results['brand_counts']:
                if results.get('tracking_enabled', False):
                    summary += "Unique brands detected:\n"
                else:
                    summary += "Top brands (frame-by-frame):\n"
                sorted_brands = sorted(results['brand_counts'].items(), key=lambda x: x[1], reverse=True)
                for brand, count in sorted_brands[:10]:
                    summary += f"• {brand}: {count}\n"
            
            if output_path:
                summary += f"\nProcessed video saved to:\n{output_path}"
                self.save_video_button.setEnabled(True)
            
            self.summary_text.setPlainText(summary)
            self.status_label.setText("Video processing completed successfully!")
            
            # Show completion message
            QMessageBox.information(self, "Processing Complete", 
                                  f"Video processing completed!\n\n"
                                  f"Detected {len(results['brand_counts'])} unique brands\n"
                                  f"Total detections: {results['total_detections']:,}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing results: {str(e)}")
    
    def on_processing_error(self, error_message):
        """Handle processing error"""
        self.reset_ui_after_processing()
        QMessageBox.critical(self, "Processing Error", f"Error during video processing:\n{error_message}")
        self.status_label.setText("Video processing failed")
    
    def reset_ui_after_processing(self):
        """Reset UI elements after processing is complete or stopped"""
        self.process_button.setEnabled(True)
        self.select_video_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def save_processed_video(self):
        """Save the processed video to a custom location"""
        if not self.output_video_path or not os.path.exists(self.output_video_path):
            QMessageBox.warning(self, "Warning", "No processed video available to save.")
            return
        
        # Get save location
        video_name = Path(self.current_video_path).stem
        default_name = f"{video_name}_processed.mp4"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Video",
            default_name,
            "Video files (*.mp4);;All files (*.*)"
        )
        
        if save_path:
            try:
                # Copy the processed video to new location
                import shutil
                shutil.copy2(self.output_video_path, save_path)
                
                QMessageBox.information(self, "Success", f"Processed video saved to:\n{save_path}")
                self.status_label.setText(f"Saved: {os.path.basename(save_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving video:\n{str(e)}")
    
    def pil_to_qpixmap(self, pil_image):
        """Convert PIL Image to QPixmap with proper color format"""
        try:
            # Ensure the image is in RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL image to bytes
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Create QPixmap from bytes
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            return pixmap
        except Exception as e:
            print(f"Error converting PIL to QPixmap: {e}")
            return QPixmap()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use a modern style
    
    window = VideoProcessingApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()