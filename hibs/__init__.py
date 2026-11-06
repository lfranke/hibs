import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I3DGS = os.path.join(ROOT, "submodules", "i3dgs")

for _path in (I3DGS, os.path.join(I3DGS, "submodules", "Depth-Anything-V2")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
