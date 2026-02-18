import sys
import os
import numpy as np
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Slot
import requests
import threading
import subprocess
import time
from depth_processor import DepthProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Anything 3 Video Visualizer")
        self.resize(1200, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Image Display Labels
        self.image_layout = QHBoxLayout()
        self.rgb_label = QLabel("RGB Video")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.depth_label = QLabel("Depth Map")
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.rgb_label)
        self.image_layout.addWidget(self.depth_label)
        self.layout.addLayout(self.image_layout)

        # Controls
        self.controls_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        self.controls_layout.addWidget(self.load_button)
        self.controls_layout.addWidget(self.start_button)
        self.controls_layout.addWidget(self.stop_button)
        self.layout.addLayout(self.controls_layout)

        self.processor = None
        self.video_path = None
        
        # Start Server
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_script = os.path.join(script_dir, "viewer", "server.py")
        self.server_process = subprocess.Popen([sys.executable, server_script])
        self.statusBar().showMessage("3D Viewer Server started at http://localhost:8000")

    def closeEvent(self, event):
        if self.server_process:
            self.server_process.terminate()
        event.accept()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.video_path = file_path
            self.start_button.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")

    def start_processing(self):
        if self.video_path:
            self.processor = DepthProcessor(self.video_path)
            self.processor.frame_processed.connect(self.update_frames)
            self.processor.finished.connect(self.on_finished)
            self.processor.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.load_button.setEnabled(False)

    def stop_processing(self):
        if self.processor:
            self.processor.stop()

    def on_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.statusBar().showMessage("Processing Finished")

    @Slot(np.ndarray, np.ndarray)
    def update_frames(self, rgb_frame, depth_map):
        # Update RGB Frame
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_rgb = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.rgb_label.setPixmap(QPixmap.fromImage(q_rgb).scaled(580, 480, Qt.KeepAspectRatio))

        # Update Depth Map
        # Normalize depth for visualization (0-255)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            norm_depth = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
        else:
            norm_depth = depth_map * 0
        
        norm_depth = norm_depth.astype(np.uint8)
        # Apply colormap for better visualization
        depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_VIRIDIS)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        
        dh, dw, dch = depth_color.shape
        d_bytes_per_line = dch * dw
        q_depth = QImage(depth_color.data, dw, dh, d_bytes_per_line, QImage.Format_RGB888)
        self.depth_label.setPixmap(QPixmap.fromImage(q_depth).scaled(580, 480, Qt.KeepAspectRatio))

        # Send to 3D Viewer (Downsample for performance)
        threading.Thread(target=self.send_to_viewer, args=(rgb_frame, depth_map), daemon=True).start()

    def send_to_viewer(self, rgb, depth):
        try:
            # Downsample further if needed or just send
            # We'll send a downsampled version to the viewer for speed
            h, w = depth.shape
            scale = 0.5
            d_small = cv2.resize(depth, (int(w*scale), int(h*scale)))
            rgb_small = cv2.resize(rgb, (int(w*scale), int(h*scale)))
            
            data = {
                "depth": d_small.flatten().tolist(),
                "rgb": rgb_small.flatten().tolist(),
                "width": int(w*scale),
                "height": int(h*scale)
            }
            requests.post("http://localhost:8000/update", json=data, timeout=0.1)
        except Exception as e:
            print(f"Error sending to viewer: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
