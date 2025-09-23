import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QScrollArea, QFrame, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from ultralytics import YOLO
import torch


class YOLOInferenceThread(QThread):
    """Thread for running YOLO inference without blocking the UI"""
    inference_complete = pyqtSignal(object, object)  # results, annotated_image
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path, image_path, device='auto'):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.device = device
    
    def run(self):
        try:
            # Load model
            model = YOLO(self.model_path)
            
            # Set device
            if self.device != 'auto':
                model.to(self.device)
            
            # Run inference
            results = model(self.image_path, device=self.device)
            
            # Get annotated image
            annotated_image = results[0].plot()
            
            self.inference_complete.emit(results, annotated_image)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class YOLODetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = "model/best.pt"
        self.current_image_path = None
        self.inference_thread = None
        self.annotated_image_data = None  # Store processed image data for saving
        self.available_devices = self.get_available_devices()
        self.current_device = self.available_devices[0] if self.available_devices else 'cpu'
        
        self.init_ui()
        self.check_model_exists()
    
    def get_available_devices(self):
        """Get list of available devices (CPU, CUDA GPUs)"""
        devices = ['cpu']
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
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
        self.setWindowTitle("YOLO Object Detection - Brand Eye")
        self.setGeometry(50, 50, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Images display area
        images_layout = QHBoxLayout()
        
        # Original image display
        self.original_image_frame = self.create_image_frame("Original Image")
        images_layout.addWidget(self.original_image_frame)
        
        # Detection results display
        self.result_image_frame = self.create_image_frame("Detection Results")
        images_layout.addWidget(self.result_image_frame)
        
        main_layout.addLayout(images_layout)
        
        # Results text area
        self.results_label = QLabel("Results will be displayed here...")
        self.results_label.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
                color: #333333;
                line-height: 1.4;
            }
        """)
        self.results_label.setMinimumHeight(180)
        self.results_label.setMaximumHeight(220)
        self.results_label.setWordWrap(True)
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(self.results_label)
    
    def create_control_panel(self):
        """Create the control panel with buttons"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setMaximumHeight(80)
        
        layout = QHBoxLayout(frame)
        
        # Select image button
        self.select_button = QPushButton("📂 Select Image")
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)
        
        # Detect button
        self.detect_button = QPushButton("🔍 Detect Objects")
        self.detect_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
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
        self.detect_button.clicked.connect(self.run_detection)
        self.detect_button.setEnabled(False)
        layout.addWidget(self.detect_button)
        
        # Save result button
        self.save_button = QPushButton("💾 Save Result")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
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
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
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
                min-width: 120px;
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
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #2196F3;
                font-size: 12pt;
                font-weight: bold;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        return frame
    
    def create_image_frame(self, title):
        """Create a frame for displaying images"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 12pt;
                font-weight: bold;
                color: #333;
                text-align: center;
                padding: 5px;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Image display area with scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("""
            QLabel {
                background-color: #f9f9f9;
                border: 2px dashed #ccc;
                color: #666;
            }
        """)
        image_label.setText("Image will be displayed here once loaded")
        
        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)
        
        # Store reference to image label
        if "Original" in title:
            self.original_image_label = image_label
        else:
            self.result_image_label = image_label
        
        return frame
    
    def check_model_exists(self):
        """Check if the YOLO model file exists"""
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Model Not Found", 
                              f"YOLO model file not found: {self.model_path}\n"
                              "Please ensure the model file is in the correct location.")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.load_image(file_path, self.original_image_label)
            self.detect_button.setEnabled(True)
            self.save_button.setEnabled(False)  # Disable save button when new image is selected
            self.annotated_image_data = None  # Clear previous processed image
            self.status_label.setText(f"Selected file: {os.path.basename(file_path)}")
            
            # Clear previous results
            self.result_image_label.setText("Detection results will be displayed here")
            self.results_label.setText("Results will be displayed here...")
    
    def on_device_changed(self):
        """Handle device selection change"""
        current_index = self.device_combo.currentIndex()
        if current_index >= 0:
            self.current_device = self.device_combo.itemData(current_index)
            device_name = self.get_device_display_name(self.current_device)
            self.status_label.setText(f"Device changed to: {device_name}")
    
    def load_image(self, image_path, label):
        """Load and display an image in the given label"""
        try:
            # Load image with OpenCV to handle various formats
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # OpenCV loads images in BGR format, convert to RGB for proper display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to QPixmap
            pixmap = self.pil_to_qpixmap(pil_image)
            
            # Scale image to fit display area while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(500, 400, Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation)
            
            label.setPixmap(scaled_pixmap)
            label.setText("")  # Clear any placeholder text
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
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
            return QPixmap()  # Return empty pixmap on error
    
    def run_detection(self):
        """Run YOLO detection on the selected image"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        
        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, "Error", f"Model file not found: {self.model_path}")
            return
        
        # Disable button and update status
        self.detect_button.setEnabled(False)
        device_name = self.get_device_display_name(self.current_device)
        self.status_label.setText(f"Running detection on {device_name}...")
        
        # Start inference in separate thread
        self.inference_thread = YOLOInferenceThread(
            self.model_path, 
            self.current_image_path, 
            self.current_device
        )
        self.inference_thread.inference_complete.connect(self.on_inference_complete)
        self.inference_thread.error_occurred.connect(self.on_inference_error)
        self.inference_thread.start()
    
    def on_inference_complete(self, results, annotated_image):
        """Handle completion of YOLO inference"""
        try:
            # YOLO returns BGR format, convert to RGB for proper display
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Store the annotated image data for saving
            self.annotated_image_data = annotated_image_rgb
            
            # Convert annotated image (numpy array) to PIL Image
            annotated_pil = Image.fromarray(annotated_image_rgb)
            
            # Display annotated image
            pixmap = self.pil_to_qpixmap(annotated_pil)
            scaled_pixmap = pixmap.scaled(500, 400, Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation)
            self.result_image_label.setPixmap(scaled_pixmap)
            self.result_image_label.setText("")
            
            # Display detection results
            self.display_results(results[0])
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error displaying results: {str(e)}")
        
        finally:
            # Re-enable buttons and update status
            self.detect_button.setEnabled(True)
            self.save_button.setEnabled(True)  # Enable save button after successful detection
            self.status_label.setText("Detection completed")
    
    def on_inference_error(self, error_message):
        """Handle inference error"""
        QMessageBox.critical(self, "Detection Error", f"Error during detection:\n{error_message}")
        self.detect_button.setEnabled(True)
        self.status_label.setText("Detection error")
    
    def display_results(self, result):
        """Display detection results in text format"""
        try:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                self.results_label.setText("No objects detected.")
                return
            
            results_text = f"Detected Objects ({len(boxes)} items):\n\n"
            
            for i, box in enumerate(boxes):
                # Get class name
                class_id = int(box.cls[0])
                class_name = result.names[class_id] if result.names else f"Class {class_id}"
                
                # Get confidence
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                results_text += f"{i+1}. {class_name}\n"
                results_text += f"   Confidence: {confidence:.2f} ({confidence*100:.1f}%)\n"
                results_text += f"   Location: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})\n"
                results_text += f"   Size: {int(x2-x1)}x{int(y2-y1)}\n\n"
            
            self.results_label.setText(results_text)
            
        except Exception as e:
            self.results_label.setText(f"Error displaying results: {str(e)}")
    
    def save_result(self):
        """Save the processed image with detections"""
        if self.annotated_image_data is None:
            QMessageBox.warning(self, "Warning", "No processed image to save. Please run detection first.")
            return
        
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No original image selected.")
            return
        
        # Get the original filename and create default save name
        original_name = Path(self.current_image_path).stem
        original_ext = Path(self.current_image_path).suffix
        default_name = f"{original_name}_detected{original_ext}"
        
        # Open save dialog
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            default_name,
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;PNG files (*.png);;JPEG files (*.jpg *.jpeg)"
        )
        
        if save_path:
            try:
                # Convert RGB numpy array to PIL Image
                pil_image = Image.fromarray(self.annotated_image_data)
                
                # Save the image
                pil_image.save(save_path, quality=95)
                
                # Show success message
                QMessageBox.information(self, "Success", f"Image saved successfully:\n{save_path}")
                self.status_label.setText(f"Saved: {os.path.basename(save_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving image:\n{str(e)}")
                self.status_label.setText("Save failed")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use a modern style
    
    window = YOLODetectionApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()