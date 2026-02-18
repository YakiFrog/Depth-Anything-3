import cv2
import torch
import numpy as np
from PySide6.QtCore import QThread, Signal
from depth_anything_3.api import DepthAnything3

class DepthProcessor(QThread):
    frame_processed = Signal(np.ndarray, np.ndarray, np.ndarray)  # rgb_frame, depth_map, extrinsics
    finished = Signal()

    def __init__(self, video_path, model_name="depth-anything/DA3-SMALL", process_res=756, batch_size=4, strategy="saddle_balanced"):
        super().__init__()
        self.video_path = video_path
        self.model_name = model_name
        self.process_res = process_res
        self.batch_size = batch_size
        self.strategy = strategy
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def run(self):
        try:
            if self.model is None:
                print(f"Loading model: {self.model_name}")
                self.model = DepthAnything3.from_pretrained(self.model_name)
                self.model = self.model.to(device=self.device)
                print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.finished.emit()
            return

        cap = cv2.VideoCapture(self.video_path)
        self.running = True

        frames_bundle = []
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_bundle.append(frame_rgb)

            if len(frames_bundle) >= self.batch_size:
                with torch.no_grad():
                    prediction = self.model.inference(
                        frames_bundle, 
                        process_res=self.process_res,
                        ref_view_strategy=self.strategy
                    )
                
                for i in range(len(frames_bundle)):
                    if not self.running: break
                    ext = prediction.extrinsics[i] if prediction.extrinsics is not None else np.eye(4)
                    self.frame_processed.emit(frames_bundle[i], prediction.depth[i], ext)
                
                frames_bundle = []

        # Process remaining frames
        if self.running and frames_bundle:
            with torch.no_grad():
                prediction = self.model.inference(
                    frames_bundle, 
                    process_res=self.process_res,
                    ref_view_strategy=self.strategy
                )
            for i in range(len(frames_bundle)):
                ext = prediction.extrinsics[i] if prediction.extrinsics is not None else np.eye(4)
                self.frame_processed.emit(frames_bundle[i], prediction.depth[i], ext)

        cap.release()
        self.finished.emit()

    def stop(self):
        self.running = False
