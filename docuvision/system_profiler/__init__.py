"""System profiling: detect CPU/GPU/RAM/VRAM and installed libs, then recommend a tier."""
from docuvision.system_profiler.profiler import profile_system, SystemCapabilityReport

__all__ = ["profile_system", "SystemCapabilityReport"]
