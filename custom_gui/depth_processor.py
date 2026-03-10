import cv2
import torch
import numpy as np
import os
import sys
from PySide6.QtCore import QThread, Signal

# Add da3_streaming to sys.path to access official utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
da3_streaming_path = os.path.join(project_root, "da3_streaming")
if da3_streaming_path not in sys.path:
    sys.path.append(da3_streaming_path)

from depth_anything_3.api import DepthAnything3
from loop_utils.alignment_torch import (
    robust_weighted_estimate_sim3_torch,
    depth_to_point_cloud_optimized_torch,
    apply_sim3_direct_torch,
)
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    process_loop_list,
    compute_sim3_ab,
    accumulate_sim3_transforms
)

class DepthProcessor(QThread):
    frame_processed = Signal(np.ndarray, np.ndarray, np.ndarray)  # rgb_frame, depth_map, extrinsics
    optimization_finished = Signal(list) # list of optimized global extrinsics
    finished = Signal()

    def __init__(self, video_path, model_name="depth-anything/DA3-SMALL", process_res=756, batch_size=8, overlap=4, strategy="saddle_balanced"):
        super().__init__()
        # Ensure batch_size is at least 2 if we want overlap
        if batch_size < 2:
            print(f"Warning: batch_size {batch_size} is too small for trajectory. Increasing to 2.")
            batch_size = 2
        
        # Ensure overlap is less than batch_size
        if overlap >= batch_size:
            overlap = batch_size // 2
            print(f"Warning: overlap too large. Adjusted to {overlap}")

        self.video_path = video_path
        self.model_name = model_name
        self.process_res = process_res
        self.batch_size = batch_size
        self.overlap = overlap
        self.strategy = strategy
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Cumulative Sim3 transform (Scale, Rotation, Translation)
        self.cum_s = 1.0
        self.cum_R = np.eye(3, dtype=np.float32)
        self.cum_t = np.zeros(3, dtype=np.float32)
        
        # Data for Global Optimization
        self.sim3_list = []      # Relative Sim3 transforms between chunks
        self.chunk_indices = []  # [(start, end), ...] for each chunk
        self.prediction_paths = [] # Paths to saved .npy chunks
        self.temp_dir = os.path.join(project_root, "temp_chunks")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Buffer for overlap frames from previous batch
        self.prev_batch_data = None # (depths, intrinsics, extrinsics, confs)
        self.processed_frame_count = 0

    def apply_cum_sim3_to_ext(self, chunk_ext):
        """
        Apply cumulative Sim3 to a batch of extrinsics (N, 3, 4)
        transformed_c2w = S @ c2w
        transformed_c2w[:3, :3] /= s
        """
        N = chunk_ext.shape[0]
        global_exts = []
        
        # Construct 4x4 Sim3 matrix S
        S = np.eye(4, dtype=np.float32)
        S[:3, :3] = self.cum_s * self.cum_R
        S[:3, 3] = self.cum_t
        
        for i in range(N):
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = chunk_ext[i]
            c2w = np.linalg.inv(w2c)
            
            transformed_c2w = S @ c2w
            transformed_c2w[:3, :3] /= self.cum_s # Normalize rotation part
            
            # Convert back to W2C
            transformed_w2c = np.linalg.inv(transformed_c2w)
            global_exts.append(transformed_w2c[:3, :4])
            
        return np.stack(global_exts)

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

        frames_buffer = []
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_buffer.append(frame_rgb)

            # Wait until we have enough frames for a full batch
            if len(frames_buffer) >= self.batch_size:
                with torch.no_grad():
                    prediction = self.model.inference(
                        frames_buffer, 
                        process_res=self.process_res,
                        ref_view_strategy=self.strategy
                    )
                
                # Align batch
                if self.prev_batch_data is not None:
                    # Sequential alignment logic
                    # 1. Get overlapping frames from current prediction
                    cur_depth = prediction.depth[:self.overlap]
                    cur_intr = prediction.intrinsics[:self.overlap]
                    cur_extr = prediction.extrinsics[:self.overlap]
                    cur_conf = prediction.conf[:self.overlap]
                    
                    # 2. Get corresponding overlap from previous batch
                    prev_depth = self.prev_batch_data['depth'][-self.overlap:]
                    prev_intr = self.prev_batch_data['intrinsics'][-self.overlap:]
                    prev_extr = self.prev_batch_data['extrinsics'][-self.overlap:]
                    prev_conf = self.prev_batch_data['conf'][-self.overlap:]
                    
                    # 3. Project to point clouds in their LOCAL coordinates
                    pcd_cur = depth_to_point_cloud_optimized_torch(cur_depth, cur_intr, cur_extr)
                    pcd_prev = depth_to_point_cloud_optimized_torch(prev_depth, prev_intr, prev_extr)
                    
                    # Flatten for alignment
                    pcd_cur_flat = pcd_cur.reshape(-1, 3)
                    pcd_prev_flat = pcd_prev.reshape(-1, 3)
                    conf_cur_flat = cur_conf.reshape(-1)
                    conf_prev_flat = prev_conf.reshape(-1)
                    
                    # Combined weights based on confidence
                    # Prediction output is typically numpy already
                    weights = (conf_cur_flat * conf_prev_flat)
                    if hasattr(weights, 'cpu'): weights = weights.cpu()
                    if hasattr(weights, 'numpy'): weights = weights.numpy()
                    
                    # 4. robust estimate Sim3: pcd_cur -> pcd_prev
                    s_src = pcd_cur_flat
                    if hasattr(s_src, 'cpu'): s_src = s_src.cpu()
                    if hasattr(s_src, 'numpy'): s_src = s_src.numpy()
                    
                    s_tgt = pcd_prev_flat
                    if hasattr(s_tgt, 'cpu'): s_tgt = s_tgt.cpu()
                    if hasattr(s_tgt, 'numpy'): s_tgt = s_tgt.numpy()

                    s_rel, R_rel, t_rel = robust_weighted_estimate_sim3_torch(
                        s_src, 
                        s_tgt, 
                        weights,
                        align_method="sim3"
                    )
                    
                    # 5. Update cumulative transform
                    # T_cum_new = T_cum_prev * T_rel
                    # Note: These are Sim3 transforms in world space
                    R_new = self.cum_R @ R_rel
                    s_new = self.cum_s * s_rel
                    t_new = self.cum_s * (self.cum_R @ t_rel) + self.cum_t
                    
                    self.cum_R = R_new
                    self.cum_s = s_new
                    self.cum_t = t_new
                    
                    self.sim3_list.append((s_rel, R_rel, t_rel))

                # Store chunk info
                chunk_start = self.processed_frame_count
                chunk_end = chunk_start + len(frames_buffer)
                self.chunk_indices.append((chunk_start, chunk_end))
                
                # Save chunk data to disk to save RAM
                chunk_path = os.path.join(self.temp_dir, f"chunk_{len(self.prediction_paths)}.npy")
                np.save(chunk_path, prediction)
                self.prediction_paths.append(chunk_path)

                # Transform extrinsics to global coordinates
                global_exts = self.apply_cum_sim3_to_ext(prediction.extrinsics)
                
                # Emit frames (except the overlap portion if we want to avoid duplicates? 
                # Actually, indices are continuous in video, so we just emit all of them in order)
                # To keep it simple for real-time visualization, we emit the non-overlap part of the previous batch if we were buffering
                # Wait, the simplest way is to just emit the whole batch. 
                # But we need to handle sequential reading. 
                # If batch_size=8 and overlap=4, we read indices [0..7], then [4..11], then [8..15].
                # So we should only emit the UNIQUE frames.
                
                # Current simple implementation: Process batches of 8, with NO overlap in READ but use internal overlap?
                # No, official long-sequence processing REQUIRES overlap in input images.
                # Let's adjust the reading logic to support sliding window.
                
                emit_start = 0 if self.prev_batch_data is None else self.overlap
                for i in range(emit_start, len(frames_buffer)):
                    if not self.running: break
                    self.frame_processed.emit(frames_buffer[i], prediction.depth[i], global_exts[i])
                
                self.processed_frame_count += (len(frames_buffer) - emit_start)

                # Store data for next alignment
                self.prev_batch_data = {
                    'depth': prediction.depth,
                    'intrinsics': prediction.intrinsics,
                    'extrinsics': prediction.extrinsics,
                    'conf': prediction.conf
                }
                
                # Prepare next buffer: Keep 'overlap' frames
                frames_buffer = frames_buffer[-self.overlap:]

        # Process remaining frames (if any new frames are left)
        if self.running and len(frames_buffer) > self.overlap:
            with torch.no_grad():
                prediction = self.model.inference(
                    frames_buffer,
                    process_res=self.process_res,
                    ref_view_strategy=self.strategy
                )
            
            if self.prev_batch_data is not None:
                # Align final chunk
                num_avail_overlap = min(len(prediction.depth), len(self.prev_batch_data['depth']), self.overlap)
                if num_avail_overlap > 0:
                    cur_depth = prediction.depth[:num_avail_overlap]
                    cur_intr = prediction.intrinsics[:num_avail_overlap]
                    cur_extr = prediction.extrinsics[:num_avail_overlap]
                    cur_conf = prediction.conf[:num_avail_overlap]
                    
                    prev_depth = self.prev_batch_data['depth'][-num_avail_overlap:]
                    prev_intr = self.prev_batch_data['intrinsics'][-num_avail_overlap:]
                    prev_extr = self.prev_batch_data['extrinsics'][-num_avail_overlap:]
                    prev_conf = self.prev_batch_data['conf'][-num_avail_overlap:]
                    
                    pcd_cur = depth_to_point_cloud_optimized_torch(cur_depth, cur_intr, cur_extr)
                    pcd_prev = depth_to_point_cloud_optimized_torch(prev_depth, prev_intr, prev_extr)
                    
                    weights = (cur_conf.reshape(-1) * prev_conf.reshape(-1))
                    if hasattr(weights, 'cpu'): weights = weights.cpu()
                    if hasattr(weights, 'numpy'): weights = weights.numpy()
                    
                    s_src = pcd_cur.reshape(-1, 3)
                    if hasattr(s_src, 'cpu'): s_src = s_src.cpu()
                    if hasattr(s_src, 'numpy'): s_src = s_src.numpy()
                    
                    s_tgt = pcd_prev.reshape(-1, 3)
                    if hasattr(s_tgt, 'cpu'): s_tgt = s_tgt.cpu()
                    if hasattr(s_tgt, 'numpy'): s_tgt = s_tgt.numpy()

                    s_rel, R_rel, t_rel = robust_weighted_estimate_sim3_torch(
                        s_src, s_tgt, weights, align_method="sim3"
                    )
                    
                    self.cum_R = self.cum_R @ R_rel
                    self.cum_s = self.cum_s * s_rel
                    self.cum_t = self.cum_s * (self.cum_R @ t_rel) + self.cum_t

            global_exts = self.apply_cum_sim3_to_ext(prediction.extrinsics)
            for i in range(self.overlap, len(frames_buffer)):
                if not self.running: break
                self.frame_processed.emit(frames_buffer[i], prediction.depth[i], global_exts[i])

        cap.release()
        self.finished.emit()

    def stop(self):
        self.running = False

    def optimize_trajectory(self):
        """
        Post-processing: Detect loops and optimize the whole trajectory.
        """
        try:
            if not self.prediction_paths:
                print("No chunks to optimize.")
                return

            print("Starting Global Optimization...")
            
            # 1. Prepare keyframe images for LoopDetector
            img_temp_dir = os.path.join(self.temp_dir, "images")
            os.makedirs(img_temp_dir, exist_ok=True)
            
            # Since we don't store all RGB frames in RAM, we might need to re-read video
            # or extract them during initial processing. 
            # To be efficient, let's assume we extract them here (limited to keyframes).
            cap = cv2.VideoCapture(self.video_path)
            
            # For simplicity, extract 1 frame per chunk (the middle frame)
            kf_paths = []
            for i, (start, end) in enumerate(self.chunk_indices):
                mid_idx = (start + end) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
                ret, frame = cap.read()
                if ret:
                    p = os.path.join(img_temp_dir, f"chunk_{i:04d}.jpg")
                    cv2.imwrite(p, frame)
                    kf_paths.append(p)
            cap.release()

            if not kf_paths:
                print("Failed to extract keyframes for loop detection.")
                return

            # 2. Loop Detection
            config = {
                "Weights": {"SALAD": os.path.join(da3_streaming_path, "weights", "dino_salad.ckpt")},
                "Loop": {
                    "SALAD": {
                        "image_size": [336, 336],
                        "batch_size": 32,
                        "similarity_threshold": 0.85,
                        "top_k": 5,
                        "use_nms": True,
                        "nms_threshold": 5 # Smaller threshold for chunks
                    },
                    "SIM3_Optimizer": {
                        "max_iterations": 30,
                        "lambda_init": 1e-6
                    }
                }
            }
            
            detector = LoopDetector(img_temp_dir, config=config)
            loop_closures = detector.run() # List of (idx1, idx2, score)
            
            if not loop_closures:
                print("No loop closures detected.")
                # Even if no loops, we could technically run optimization, 
                # but it won't change sequential poses.
                # However, let's send the current poses back as "optimized" to show we finished.
                self.emit_final_poses()
                return

            print(f"Found {len(loop_closures)} loop closures.")

            # 3. Estimate Sim3 for loop constraints
            loop_constraints = []
            for i1, i2, score in loop_closures:
                # i1, i2 are indices of CHUNKS
                # We need to find the relative Sim3 between chunk i1 and chunk i2
                # In this simplified version, we'll align the overlap regions if possible, 
                # but since they might be far apart, we'll align the middle part of chunks.
                
                chunk_a = np.load(self.prediction_paths[i1], allow_pickle=True).item()
                chunk_b = np.load(self.prediction_paths[i2], allow_pickle=True).item()
                
                # Align chunk b to chunk a
                # pcd_a = depth_to_point_cloud_optimized_torch(chunk_a.depth, chunk_a.intrinsics, chunk_a.extrinsics)
                # pcd_b = depth_to_point_cloud_optimized_torch(chunk_b.depth, chunk_b.intrinsics, chunk_b.extrinsics)
                
                # For robustness, just use the middle frame point cloud from each
                mid_a = len(chunk_a.depth) // 2
                mid_b = len(chunk_b.depth) // 2
                
                pcd_a = depth_to_point_cloud_optimized_torch(
                    chunk_a.depth[mid_a:mid_a+1], 
                    chunk_a.intrinsics[mid_a:mid_a+1], 
                    chunk_a.extrinsics[mid_a:mid_a+1]
                ).reshape(-1, 3)
                
                pcd_b = depth_to_point_cloud_optimized_torch(
                    chunk_b.depth[mid_b:mid_b+1], 
                    chunk_b.intrinsics[mid_b:mid_b+1], 
                    chunk_b.extrinsics[mid_b:mid_b+1]
                ).reshape(-1, 3)
                
                conf_a = chunk_a.conf[mid_a:mid_a+1].reshape(-1)
                conf_b = chunk_b.conf[mid_b:mid_b+1].reshape(-1)
                
                weights = (conf_a * conf_b)
                
                s_rel, R_rel, t_rel = robust_weighted_estimate_sim3_torch(
                    pcd_b.cpu().numpy(), 
                    pcd_a.cpu().numpy(), 
                    weights.cpu().numpy(),
                    align_method="sim3"
                )
                
                loop_constraints.append((i1, i2, (s_rel, R_rel, t_rel)))

            # 4. Global Optimization
            optimizer = Sim3LoopOptimizer(config)
            optimized_sim3 = optimizer.optimize(self.sim3_list, loop_constraints)
            
            # 5. Apply optimized transforms and emit
            self.emit_final_poses(optimized_sim3)

        except Exception as e:
            print(f"Optimization error: {e}")
            import traceback
            traceback.print_exc()

    def emit_final_poses(self, optimized_sim3_list=None):
        """
        Calculate global poses using (optionally optimized) Sim3 list and emit.
        """
        all_optimized_exts = []
        
        # If no optimized list, use identity/existing sequential accumulated ones.
        # But it's easier to just re-accumulate from sim3_list.
        
        # Identity for first chunk
        T_cum = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        
        # Process first chunk
        chunk0 = np.load(self.prediction_paths[0], allow_pickle=True).item()
        
        # First chunk gets NO transform (or identity)
        for i in range(len(chunk0.extrinsics)):
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = chunk0.extrinsics[i]
            all_optimized_exts.append(w2c.flatten().tolist())
            
        # Subsequent chunks
        current_sim3 = optimized_sim3_list if optimized_sim3_list else self.sim3_list
        accum_sim3 = accumulate_sim3_transforms(current_sim3)
        
        for chunk_idx in range(1, len(self.prediction_paths)):
            s, R, t = accum_sim3[chunk_idx - 1]
            chunk_data = np.load(self.prediction_paths[chunk_idx], allow_pickle=True).item()
            
            S = np.eye(4, dtype=np.float32)
            S[:3, :3] = s * R
            S[:3, 3] = t
            
            for i in range(len(chunk_data.extrinsics)):
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :] = chunk_data.extrinsics[i]
                c2w = np.linalg.inv(w2c)
                
                transformed_c2w = S @ c2w
                transformed_c2w[:3, :3] /= s
                
                transformed_w2c = np.linalg.inv(transformed_c2w)
                all_optimized_exts.append(transformed_w2c.flatten().tolist())

        self.optimization_finished.emit(all_optimized_exts)
