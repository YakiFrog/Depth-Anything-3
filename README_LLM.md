# LLM Context: Depth Anything 3 (DA3)

This file provide a high-level technical overview of the DA3 codebase for LLMs and AI agents.

## Project Purpose
Depth Anything 3 (DA3) is a foundation model for visual geometry, predicting consistent depth, camera poses, and 3D Gaussians from single or multiple views.

## Core Structure
- `src/depth_anything_3/`: Primary package.
    - `api.py`: **Main Entry Point.** `DepthAnything3` class handles model loading (`from_pretrained`) and `inference`.
    - `model/`: Architecture definitions.
        - `da3.py`: Unified model class (`DepthAnything3Net`).
        - `dinov2/`: DINOv2 backbone.
        - `dualdpt.py`: Depth/Pose decoder.
        - `gs_adapter.py`: Gaussian Splatting decoder.
    - `cli.py`: Command-line interface logic using `typer`.
    - `configs/`: YAML configurations for different model variants.
- `da3_streaming/`: Logic for processing long video sequences using a sliding-window approach.
- `custom_gui/`: (Added) Interactive desktop application.
    - `app.py`: PySide6-based main window.
    - `depth_processor.py`: QThread for asynchronous video processing.
    - `viewer/`: FastAPI-based 3D point cloud visualizer.

## Key API Usage
```python
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
prediction = model.inference(images_list_or_path, process_res=756)
# prediction contains: .depth, .extrinsics, .intrinsics, .processed_images
```

## Dependencies
- Backend: `torch`, `torchvision`, `xformers`, `huggingface_hub`, `omegaconf`.
- 3D/Processing: `pycolmap`, `trimesh`, `open3d`, `numpy`, `opencv-python`.
- GUI/Viewer: `PySide6`, `fastapi`, `uvicorn`, `websockets`.

## Coding Patterns
- Configurations are handled via `omegaconf` and a custom object creation registry (`cfg.py`).
- Inference uses a sliding window or batching depending on the model/series.
- `gsplat` is used for the Gaussian head.
