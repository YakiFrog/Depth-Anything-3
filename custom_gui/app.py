import sys
import os
import numpy as np
import cv2
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Slot
import requests
import threading
import subprocess
import webbrowser
import queue
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

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "depth-anything/DA3-SMALL",
            "depth-anything/DA3-BASE",
            "depth-anything/DA3-LARGE-1.1",
            "depth-anything/DA3-GIANT-1.1",
            "depth-anything/DA3METRIC-LARGE",
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
        ])
        self.model_combo.setToolTip("Select DA3 Model")

        self.res_combo = QComboBox()
        self.res_combo.addItems(["504", "756", "1008"])
        self.res_combo.setCurrentText("756")
        self.res_combo.setToolTip("Processing Resolution")

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["saddle_balanced", "first", "middle"])
        self.strategy_combo.setToolTip("Reference View Strategy (Fast vs High Quality)")

        self.view_3d_button = QPushButton("Open 3D Viewer")
        self.view_3d_button.clicked.connect(self.open_3d_viewer)
        
        self.controls_layout.addWidget(self.load_button)
        self.controls_layout.addWidget(QLabel("Model:"))
        self.controls_layout.addWidget(self.model_combo)
        self.controls_layout.addWidget(QLabel("Res:"))
        self.controls_layout.addWidget(self.res_combo)
        self.controls_layout.addWidget(QLabel("Strategy:"))
        self.controls_layout.addWidget(self.strategy_combo)
        self.controls_layout.addWidget(self.start_button)
        self.controls_layout.addWidget(self.stop_button)
        self.controls_layout.addWidget(self.view_3d_button)
        self.layout.addLayout(self.controls_layout)

        self.processor = None
        self.video_path = None
        
        # 3D Viewer Update Management
        self.viewer_queue = queue.Queue(maxsize=1)
        self.viewer_thread = threading.Thread(target=self.viewer_update_worker, daemon=True)
        self.viewer_thread.start()
        
        # Start Server
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_script = os.path.join(script_dir, "viewer", "server.py")
        self.server_process = subprocess.Popen([sys.executable, server_script])
        self.statusBar().showMessage("3D Viewer Server started at http://localhost:8000")

    def closeEvent(self, event):
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait()
        if self.server_process:
            self.server_process.terminate()
        event.accept()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.video_path = file_path
            self.start_button.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")

    def open_3d_viewer(self):
        webbrowser.open("http://localhost:8000")

    def start_processing(self):
        if self.video_path:
            selected_model = self.model_combo.currentText()
            res = int(self.res_combo.currentText())
            strategy = self.strategy_combo.currentText()
            # For SMALL, we use batch_size=4 to boost speed. 
            # For larger models, keep it at 1 to avoid OOM or slow UI response.
            batch_size = 4 if "SMALL" in selected_model else 1
            
            self.processor = DepthProcessor(self.video_path, model_name=selected_model, process_res=res, batch_size=batch_size, strategy=strategy)
            self.processor.frame_processed.connect(self.update_frames)
            self.processor.finished.connect(self.on_finished)
            
            self.processor.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.load_button.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.res_combo.setEnabled(False)

    def stop_processing(self):
        if self.processor:
            self.processor.stop()
            self.processor.wait()
            self.on_finished()

    def on_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.res_combo.setEnabled(True)
        self.strategy_combo.setEnabled(True)
        self.statusBar().showMessage("Processing Finished or Stopped")

    @Slot(np.ndarray, np.ndarray, np.ndarray)
    def update_frames(self, rgb_frame, depth_map, extrinsics):
        # Update RGB Frame
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_rgb = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.rgb_label.setPixmap(QPixmap.fromImage(q_rgb).scaled(580, 480, Qt.KeepAspectRatio))

        # Update Depth Map
        # Robust normalization using percentiles to ignore outliers
        p_min = np.percentile(depth_map, 2)
        p_max = np.percentile(depth_map, 98)
        
        if p_max - p_min > 0:
            norm_depth = np.clip((depth_map - p_min) / (p_max - p_min) * 255.0, 0, 255)
        else:
            norm_depth = np.zeros_like(depth_map)
        
        norm_depth = norm_depth.astype(np.uint8)
        # Apply colormap (MAGMA is usually clearer for depth)
        depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
        
        dh, dw, dch = depth_color.shape
        d_bytes_per_line = dch * dw
        q_depth = QImage(depth_color.data, dw, dh, d_bytes_per_line, QImage.Format_RGB888)
        self.depth_label.setPixmap(QPixmap.fromImage(q_depth).scaled(580, 480, Qt.KeepAspectRatio))

        # Queue for 3D Viewer
        try:
            self.viewer_queue.put_nowait((rgb_frame, depth_map, extrinsics))
        except queue.Full:
            pass
        # print("DEBUG: update_frames called")

    def viewer_update_worker(self):
        while True:
            try:
                rgb, depth, ext = self.viewer_queue.get(timeout=1.0)
                self.send_to_viewer(rgb, depth, ext)
                self.viewer_queue.task_done()
                import time
                time.sleep(0.05) # Cap at ~20fps for viewer
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Viewer worker error: {e}")

    def send_to_viewer(self, rgb, depth, extrinsics):
        try:
            # Simple check to see if server is likely up (port check could be more robust)
            # We'll just try to send and catch errors silently if it's not yet ready
            h, w = depth.shape
            scale = 0.25
            d_small = cv2.resize(depth, (int(w*scale), int(h*scale)))
            rgb_small = cv2.resize(rgb, (int(w*scale), int(h*scale)))
            
            if extrinsics is not None:
                # print(f"Sending extrinsics: {extrinsics.shape}")
                if extrinsics.shape == (3, 4):
                    row = np.array([[0, 0, 0, 1]])
                    extrinsics = np.concatenate([extrinsics, row], axis=0)
                ext_list = extrinsics.flatten().tolist()
            else:
                ext_list = np.eye(4).flatten().tolist()
                
            data = {
                "depth": d_small.flatten().tolist(),
                "rgb": rgb_small.flatten().tolist(),
                "extrinsics": ext_list,
                "width": int(w*scale),
                "height": int(h*scale)
            }
            # print(f"DEBUG: Posting to viewer... {len(data['depth'])} points")
            resp = requests.post("http://127.0.0.1:8000/update", json=data, timeout=1.0)
            if resp.status_code != 200:
                print(f"DEBUG: Viewer post failed: {resp.status_code}")
        except Exception as e:
            print(f"DEBUG: Error sending to viewer: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
