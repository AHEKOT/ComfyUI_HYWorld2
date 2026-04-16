"""HY-World 2.0 ComfyUI nodes — WorldMirror V1 & V2 3D reconstruction."""

import os
import sys

# Ensure repo root is in sys.path so `import hyworld2` and `from src...` work
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    import traceback
    print(f"❌ [HYWorld2] Failed to load nodes: {e}")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
