
from ..device import is_npu_available
from ..import_utils import is_nvtx_available
from .performance import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from .profile import DistProfilerExtension, ProfilerConfig

if is_nvtx_available():
    from .nvtx_profile import NsightSystemsProfiler as DistProfiler
    from .nvtx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
elif is_npu_available:
    from .mstx_profile import NPUProfiler as DistProfiler
    from .mstx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
else:
    from .performance import marked_timer
    from .profile import DistProfiler, mark_annotate, mark_end_range, mark_start_range

__all__ = [
    "GPUMemoryLogger",
    "log_gpu_memory_usage",
    "mark_start_range",
    "mark_end_range",
    "mark_annotate",
    "DistProfiler",
    "DistProfilerExtension",
    "ProfilerConfig",
    "simple_timer",
    "marked_timer",
]
