#__init__.py
import os
import sys
import torch
import logging
import platform
import folder_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComfyUI-IF_Trellis')


# Add parent directory to Python path to find trellis package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both current and parent dir to handle different installation scenarios
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add trellis package path
trellis_path = os.path.join(current_dir, "trellis")
if os.path.exists(trellis_path) and trellis_path not in sys.path:
    sys.path.insert(0, trellis_path)
    logger.info(f"Added trellis path to sys.path: {trellis_path}")

# Verify trellis package is importable
try:
    import trellis
    logger.info("Trellis package imported successfully")
except ImportError as e:
    logger.error(f"Failed to import trellis package: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

# Register model paths with ComfyUI
try:
    folder_paths.add_model_folder_path("trellis", os.path.join(folder_paths.models_dir, "trellis"))
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.models_dir, "checkpoints"))
except Exception as e:
    logger.error(f"Error registering model paths: {e}")

try:
    from IF_TrellisCheckpointLoader import IF_TrellisCheckpointLoader
    from IF_Trellis import IF_TrellisImageTo3D, IF_TrellisImageTo3D_v2, IF_TrellisRenderVideo, IF_TrellisRenderMultiview, IF_TrellisRenderGLB
    NODE_CLASS_MAPPINGS = {
        "IF_TrellisCheckpointLoader": IF_TrellisCheckpointLoader,
        "IF_TrellisImageTo3D": IF_TrellisImageTo3D,
        "IF_TrellisImageTo3D_v2": IF_TrellisImageTo3D_v2,
        "IF_TrellisRenderVideo": IF_TrellisRenderVideo,
        "IF_TrellisRenderMultiview": IF_TrellisRenderMultiview,
        "IF_TrellisRenderGLB": IF_TrellisRenderGLB
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "IF_TrellisCheckpointLoader": "Trellis Model Loader 💾",
        "IF_TrellisImageTo3D": "Trellis Image to 3D 🖼️➡️🎲",
        "IF_TrellisImageTo3D_v2": "Trellis Image to 3D v2 🖼️➡️🎲",
        "IF_TrellisRenderVideo": "Trellis Render Video 🎲➡️🎥",
        "IF_TrellisRenderMultiview": "Trellis Render Multiview 🎲➡️🖼️",
        "IF_TrellisRenderGLB": "Trellis Render GLB 🎲➡️📦"
    }

except Exception as e:
    logger.error(f"Error importing node classes: {e}")
    raise

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']