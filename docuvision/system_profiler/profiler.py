"""
docuvision.system_profiler.profiler
===================================

Detects system capabilities and produces a SystemCapabilityReport with a
heuristic-based recommended pipeline tier.

Tier mapping (heuristic, not a hard contract):

    Tier 0 — CPU only, <= 4 GB RAM       → tesseract + CRAFT (light)
    Tier 1 — CPU,  8 GB RAM              → easyocr
    Tier 2 — CPU + small GPU (<4 GB VRAM)→ paddleocr
    Tier 3 — GPU  6-12 GB VRAM           → doctr
    Tier 4 — GPU  12 GB+ VRAM            → trocr / donut

The profiler NEVER imports heavy libs; it only checks availability.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from docuvision.utils.lazy_import import is_available, optional_import
from docuvision.utils.logging import get_logger

log = get_logger("profiler")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class SystemCapabilityReport:
    os: str                                   # "Linux", "Darwin", "Windows"
    os_version: str
    python_version: str

    cpu_model: str
    cpu_count: int
    cpu_features: List[str]                   # ["avx", "avx2", "sse4_2", ...]
    cpu_score: float                          # heuristic 0..100

    ram_total_gb: float
    ram_available_gb: float

    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    cuda_version: Optional[str] = None
    tensorrt_available: bool = False
    vram_total_gb: float = 0.0
    vram_available_gb: float = 0.0
    gpu_score: float = 0.0

    installed_libs: Dict[str, bool] = field(default_factory=dict)
    tesseract_binary: Optional[str] = None

    tier: int = 0
    recommended_pipeline: str = "tesseract"
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "=== DocuVision System Capability Report ===",
            f"OS              : {self.os} {self.os_version}",
            f"Python          : {self.python_version}",
            f"CPU             : {self.cpu_model} ({self.cpu_count} cores)",
            f"CPU features    : {', '.join(self.cpu_features) or 'n/a'}",
            f"CPU score       : {self.cpu_score:.1f}",
            f"RAM             : {self.ram_available_gb:.1f}/{self.ram_total_gb:.1f} GB available",
            f"GPU             : {self.gpu_name or 'none'} "
            f"({'available' if self.gpu_available else 'unavailable'})",
            f"CUDA            : {self.cuda_version or 'n/a'}",
            f"TensorRT        : {'yes' if self.tensorrt_available else 'no'}",
            f"VRAM            : {self.vram_available_gb:.1f}/{self.vram_total_gb:.1f} GB available",
            f"GPU score       : {self.gpu_score:.1f}",
            f"Installed libs  : " + ", ".join(
                f"{k}={'Y' if v else 'N'}" for k, v in self.installed_libs.items()
            ),
            f"Tesseract binary: {self.tesseract_binary or 'not found'}",
            f"Tier            : {self.tier}",
            f"Recommended     : {self.recommended_pipeline}",
        ]
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _detect_cpu() -> Dict[str, object]:
    info: Dict[str, object] = {
        "model": platform.processor() or platform.machine() or "unknown",
        "count": os.cpu_count() or 1,
        "features": [],
    }

    # CPU features — try /proc/cpuinfo (Linux), sysctl (macOS), or cpuinfo lib
    features: List[str] = []
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("flags") or line.lower().startswith("features"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            features = parts[1].strip().split()
                            break
                    if line.lower().startswith("model name") and info["model"] in ("unknown", ""):
                        info["model"] = line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).decode("utf-8", errors="ignore")
            features = out.lower().split()
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).decode("utf-8", errors="ignore").strip()
            if brand:
                info["model"] = brand
        except Exception:
            pass

    # Normalize
    feat_set = {f.lower() for f in features}
    interesting = ["avx", "avx2", "avx512f", "sse4_1", "sse4_2", "fma", "neon"]
    info["features"] = [f for f in interesting if f in feat_set or f.replace("_", ".") in feat_set]
    return info


def _detect_ram() -> Dict[str, float]:
    psutil = optional_import("psutil")
    if psutil is None:
        return {"total_gb": 0.0, "available_gb": 0.0}
    vm = psutil.virtual_memory()
    return {
        "total_gb": vm.total / (1024 ** 3),
        "available_gb": vm.available / (1024 ** 3),
    }


def _detect_gpu_torch() -> Optional[Dict[str, object]]:
    torch = optional_import("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            # Check MPS for Apple Silicon
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return {
                    "available": True,
                    "name": "Apple MPS",
                    "count": 1,
                    "cuda_version": None,
                    "vram_total_gb": 0.0,
                    "vram_available_gb": 0.0,
                    "backend": "mps",
                }
            return None
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        total = props.total_memory / (1024 ** 3)
        try:
            free, _ = torch.cuda.mem_get_info(dev)
            free_gb = free / (1024 ** 3)
        except Exception:
            free_gb = total  # best-effort
        return {
            "available": True,
            "name": props.name,
            "count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "vram_total_gb": total,
            "vram_available_gb": free_gb,
            "backend": "cuda",
        }
    except Exception as e:
        log.debug("torch GPU detection failed: %s", e)
        return None


def _detect_gpu_nvidia_smi() -> Optional[Dict[str, object]]:
    """Fallback GPU detection via nvidia-smi, when torch isn't installed."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", errors="ignore").strip()
        lines = [l for l in out.splitlines() if l.strip()]
        if not lines:
            return None
        first = [c.strip() for c in lines[0].split(",")]
        name = first[0]
        total_mb = float(first[1])
        free_mb = float(first[2])
        return {
            "available": True,
            "name": name,
            "count": len(lines),
            "cuda_version": None,  # unknown without torch/cuda libs
            "vram_total_gb": total_mb / 1024.0,
            "vram_available_gb": free_mb / 1024.0,
            "backend": "cuda",
        }
    except Exception as e:
        log.debug("nvidia-smi detection failed: %s", e)
        return None


def _detect_tensorrt() -> bool:
    return is_available("tensorrt")


def _detect_libs() -> Dict[str, bool]:
    libs = [
        "torch",
        "torchvision",
        "onnxruntime",
        "tensorflow",
        "paddleocr",
        "paddle",
        "easyocr",
        "pytesseract",
        "doctr",       # python-doctr
        "transformers",
        "ultralytics",
        "hdbscan",
        "sklearn",     # scikit-learn
    ]
    return {lib: is_available(lib) for lib in libs}


def _detect_tesseract_binary() -> Optional[str]:
    # Env var takes precedence (used by pytesseract.pytesseract.tesseract_cmd)
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.exists(env):
        return env
    return shutil.which("tesseract")


# ---------------------------------------------------------------------------
# Scoring + tier decision
# ---------------------------------------------------------------------------
def _score_cpu(cpu: Dict[str, object], ram_gb: float) -> float:
    # Up to 60 points for core count, 20 for AVX2, 10 for AVX, 10 for RAM>=8GB
    cores = int(cpu["count"])  # type: ignore[index]
    score = min(60.0, cores * 6.0)
    feats = {f.lower() for f in cpu["features"]}  # type: ignore[index]
    if "avx2" in feats or "avx512f" in feats:
        score += 20
    elif "avx" in feats:
        score += 10
    if ram_gb >= 16:
        score += 20
    elif ram_gb >= 8:
        score += 10
    elif ram_gb >= 4:
        score += 5
    return min(100.0, score)


def _score_gpu(gpu: Optional[Dict[str, object]]) -> float:
    if not gpu or not gpu.get("available"):
        return 0.0
    vram = float(gpu.get("vram_total_gb", 0.0))
    # Simple piecewise: 0..4 GB -> 20..40, 4..8 -> 40..70, 8..16 -> 70..90, 16+ -> 95+
    if vram <= 0:
        return 20.0  # MPS or unknown
    if vram < 4:
        score = 20 + (vram / 4.0) * 20
    elif vram < 8:
        score = 40 + ((vram - 4) / 4.0) * 30
    elif vram < 16:
        score = 70 + ((vram - 8) / 8.0) * 20
    else:
        score = min(100.0, 90 + (vram - 16) * 0.5)
    return float(score)


def _decide_tier(
    cpu_score: float,
    gpu_score: float,
    ram_gb: float,
    vram_gb: float,
    libs: Dict[str, bool],
    tesseract_bin: Optional[str],
) -> tuple[int, str, List[str]]:
    notes: List[str] = []

    has_tesseract = libs.get("pytesseract") and tesseract_bin is not None
    has_easy = libs.get("easyocr", False)
    has_paddle = libs.get("paddleocr", False)
    has_doctr = libs.get("doctr", False)
    has_trocr = libs.get("transformers", False) and libs.get("torch", False)

    # Tier 4: big GPU + TrOCR/Donut possible
    if gpu_score >= 85 and vram_gb >= 12 and has_trocr:
        return 4, "trocr", notes
    # Tier 3: mid/high GPU, docTR
    if gpu_score >= 60 and vram_gb >= 6 and has_doctr:
        return 3, "doctr", notes
    # Tier 2: small GPU or strong CPU, PaddleOCR
    if has_paddle and (gpu_score >= 20 or cpu_score >= 60):
        return 2, "paddleocr", notes
    # Tier 1: 8GB+ RAM with EasyOCR
    if has_easy and ram_gb >= 6:
        return 1, "easyocr", notes
    # Tier 0 fallback
    if has_tesseract:
        return 0, "tesseract", notes
    # Absolutely nothing installed
    notes.append(
        "No OCR engine installed. Install one of: pytesseract (+tesseract), "
        "easyocr, paddleocr, python-doctr, or transformers+torch."
    )
    return 0, "none", notes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def profile_system(refresh: bool = False) -> SystemCapabilityReport:
    """Detect system capabilities and return a SystemCapabilityReport.

    Results are cached; pass `refresh=True` to re-run detection.
    (Note: due to lru_cache semantics, `refresh=True` creates a separate
    cache entry; the most common case — no args — stays cached.)
    """
    if refresh:
        profile_system.cache_clear()

    # OS
    os_name = platform.system()
    os_version = platform.release()

    # CPU
    cpu = _detect_cpu()

    # RAM
    ram = _detect_ram()

    # GPU — prefer torch, fall back to nvidia-smi
    gpu = _detect_gpu_torch() or _detect_gpu_nvidia_smi()

    # TensorRT
    trt = _detect_tensorrt()

    # Libs
    libs = _detect_libs()
    tesseract_bin = _detect_tesseract_binary()

    # Scores + tier
    cpu_score = _score_cpu(cpu, ram["total_gb"])
    gpu_score = _score_gpu(gpu)
    tier, recommended, notes = _decide_tier(
        cpu_score, gpu_score,
        ram["total_gb"], (gpu or {}).get("vram_total_gb", 0.0),  # type: ignore[arg-type]
        libs, tesseract_bin,
    )

    report = SystemCapabilityReport(
        os=os_name,
        os_version=os_version,
        python_version=platform.python_version(),
        cpu_model=str(cpu["model"]),
        cpu_count=int(cpu["count"]),  # type: ignore[arg-type]
        cpu_features=list(cpu["features"]),  # type: ignore[arg-type]
        cpu_score=cpu_score,
        ram_total_gb=ram["total_gb"],
        ram_available_gb=ram["available_gb"],
        gpu_available=bool(gpu and gpu.get("available")),
        gpu_name=(gpu or {}).get("name"),  # type: ignore[arg-type]
        gpu_count=int((gpu or {}).get("count", 0) or 0),  # type: ignore[arg-type]
        cuda_version=(gpu or {}).get("cuda_version"),  # type: ignore[arg-type]
        tensorrt_available=trt,
        vram_total_gb=float((gpu or {}).get("vram_total_gb", 0.0) or 0.0),  # type: ignore[arg-type]
        vram_available_gb=float((gpu or {}).get("vram_available_gb", 0.0) or 0.0),  # type: ignore[arg-type]
        gpu_score=gpu_score,
        installed_libs=libs,
        tesseract_binary=tesseract_bin,
        tier=tier,
        recommended_pipeline=recommended,
        notes=notes,
    )
    log.debug("Profiled system: tier=%d pipeline=%s", tier, recommended)
    return report
