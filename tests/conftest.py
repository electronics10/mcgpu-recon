"""Test setup. The tests use a small dense matrix as the system operator, so
they need neither a GPU nor parallelproj. If parallelproj / mcgpu_pet_wrapper
are not installed (e.g. on a laptop), minimal stand-ins are registered so that
the package can be imported; the real modules are used when available."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import parallelproj  # noqa: F401
except ImportError:
    sys.modules["parallelproj"] = types.ModuleType("parallelproj")

try:
    import mcgpu_pet_wrapper  # noqa: F401
except ImportError:
    pkg = types.ModuleType("mcgpu_pet_wrapper")
    pkg.lors = types.ModuleType("mcgpu_pet_wrapper.lors")
    cfg = types.ModuleType("mcgpu_pet_wrapper.config")
    cfg.voxel_space_shape_zyx = cfg.grid_size_mm = lambda *a, **k: None
    pkg.config = cfg
    sys.modules["mcgpu_pet_wrapper"] = pkg
    sys.modules["mcgpu_pet_wrapper.lors"] = pkg.lors
    sys.modules["mcgpu_pet_wrapper.config"] = cfg
