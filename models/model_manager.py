import os
import urllib.request
from typing import Dict, List, Any, Optional, Callable
from .base_upscaler import BaseUpscaler
from .fast_upscaler import (
    FastLanczosUpscaler, FastNEDIUpscaler, GuidedEdgeUpscaler, 
    RAISRUpscaler, xBRZRuleUpscaler, VectorContourUpscaler
)

# Neural model classes are imported lazily inside load_model() to avoid
# importing torch/timm/einops at startup, which adds ~2-3s of latency.
# This allows the GUI window to appear near-instantly.

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "realesrgan_x4_photo": {
        "name": "Real-ESRGAN Photo (x4)",
        "description": "High-fidelity Super-Resolution optimized for real-world photos.",
        "type": "neural_esrgan",
        "scale": 4,
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    },
    "remacri_x4": {
        "name": "Remacri Details (x4)",
        "description": "Community favorite model for lifelike facial textures, skin, and fabrics.",
        "type": "neural_esrgan",
        "scale": 4,
        "filename": "remacri_x4.pth",
        "url": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    },
    "bsrgan_x4": {
        "name": "BSRGAN Restorer (x4)",
        "description": "Specialized for restoring heavily degraded, blurry, or noisy old photos.",
        "type": "neural_esrgan",
        "scale": 4,
        "filename": "bsrgan_x4.pth",
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
    },
    "dat_x4": {
        "name": "DAT Transformer (x4)",
        "description": "SOTA ICCV 2023 Vision Transformer combining spatial & channel attention.",
        "type": "neural_dat",
        "scale": 4,
        "filename": "dat_x4.pth",
        "url": "https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x4.pth"
    },
    "realesrgan_x4_anime": {
        "name": "Real-ESRGAN Anime (x4)",
        "description": "Specialized neural network for sharp lines and digital illustrations.",
        "type": "neural_esrgan",
        "scale": 4,
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    },
    "realesrgan_x2_general": {
        "name": "Real-ESRGAN Fast (x2)",
        "description": "2x Super-Resolution for fast quality upscaling (supports Double Upscale to 4x).",
        "type": "neural_esrgan",
        "scale": 2,
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    },
    "swinir_x4_classical": {
        "name": "SwinIR Classical (x4)",
        "description": "SOTA Vision Transformer model for ultra-sharp photo restoration and fine details.",
        "type": "neural_swinir",
        "scale": 4,
        "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
    },
    "swinir_x4_real": {
        "name": "Real-SwinIR Photo (x4)",
        "description": "Vision Transformer model specialized for removing JPEG compression and noise.",
        "type": "neural_swinir",
        "scale": 4,
        "filename": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    },
    "fast_lanczos": {
        "name": "Fast Lanczos4 Baseline",
        "description": "Classic mathematical Lanczos-4 resampling with unsharp mask.",
        "type": "fast_lanczos",
        "scale": 4,
        "filename": None,
        "url": None
    },
    "fast_nedi": {
        "name": "Fast NEDI (Edge-Directed)",
        "description": "Directional edge covariance interpolation. Sharp diagonal lines without aliasing.",
        "type": "fast_nedi",
        "scale": 4,
        "filename": None,
        "url": None
    },
    "guided_edge": {
        "name": "Guided Edge Filter",
        "description": "Guided Filter edge-preserving enhancement. Crisp details without halo/ringing.",
        "type": "guided_edge",
        "scale": 4,
        "filename": None,
        "url": None
    },
    "raisr_patch": {
        "name": "RAISR Patch Regression",
        "description": "Example-based patch covariance regression (Google RAISR). Fast adaptive edge refinement.",
        "type": "raisr_patch",
        "scale": 4,
        "filename": None,
        "url": None
    },
    "xbrz_pattern": {
        "name": "xBRZ Pattern Engine",
        "description": "Rule-based pixel pattern engine (xBRZ/ScaleNx). Smooth anti-aliased curves for pixel art & 2D icons.",
        "type": "xbrz_pattern",
        "scale": 4,
        "filename": None,
        "url": None
    },
    "vector_contour": {
        "name": "Vector Contour Engine",
        "description": "Vectorization & polygon curve tracing. Smooth infinite-scale vector contours for logos & typography.",
        "type": "vector_contour",
        "scale": 4,
        "filename": None,
        "url": None
    }
}


def is_model_downloaded(model_id: str, weights_dir: str = "weights") -> bool:
    if model_id not in MODEL_REGISTRY:
        return True
    info = MODEL_REGISTRY[model_id]
    filename = info.get("filename")
    if not filename:
        return True
    path = os.path.join(weights_dir, filename)
    return os.path.exists(path) and os.path.getsize(path) > 0


def get_available_models(weights_dir: str = "weights") -> List[Dict[str, Any]]:
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        models.append({
            "id": model_id,
            "name": info["name"],
            "description": info["description"],
            "type": info["type"],
            "default_scale": info["scale"],
            "is_downloaded": is_model_downloaded(model_id, weights_dir)
        })
    return models


def get_vram_gb() -> float:
    """Detects available CUDA GPU VRAM in Gigabytes."""
    try:
        import torch
        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_bytes / (1024.0 ** 3)
    except Exception:
        pass
    return 0.0

def get_adaptive_tile_size() -> int:
    """
    Dynamically computes optimal tile size based on detected GPU VRAM.
    VRAM < 4GB    : 256px  (Low VRAM)
    VRAM 4GB-8GB  : 512px  (Medium VRAM)
    VRAM 8GB-12GB : 768px  (High VRAM)
    VRAM >= 12GB  : 1024px (Ultra VRAM)
    """
    vram = get_vram_gb()
    if vram <= 0:
        print("[VRAM Optimizer] CPU mode or unknown GPU. Setting safe tile size: 256px.")
        return 256
    elif vram < 4.0:
        print(f"[VRAM Optimizer] Detected {vram:.2f} GB VRAM (Low Tier). Setting adaptive tile size: 256px.")
        return 256
    elif vram < 8.0:
        print(f"[VRAM Optimizer] Detected {vram:.2f} GB VRAM (Medium Tier). Setting adaptive tile size: 512px.")
        return 512
    elif vram < 12.0:
        print(f"[VRAM Optimizer] Detected {vram:.2f} GB VRAM (High Tier). Setting adaptive tile size: 768px.")
        return 768
    else:
        print(f"[VRAM Optimizer] Detected {vram:.2f} GB VRAM (Ultra Tier). Setting adaptive tile size: 1024px.")
        return 1024


class ModelManager:
    """
    Handles weight fetching, model instantiation, and hardware device selection.
    """
    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)
        self._loaded_models: List[BaseUpscaler] = []

    def unload_all_models(self):
        """
        Unloads all tracked upscaler models, frees PyTorch CUDA VRAM, and runs garbage collection.
        """
        for model in self._loaded_models:
            try:
                model.unload_model()
            except Exception as e:
                print(f"[ModelManager] Error unloading model: {e}")
        self._loaded_models.clear()

        import sys
        if 'torch' in sys.modules:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        import gc
        gc.collect()

    def download_model_weight(
        self, 
        model_id: str, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        if model_id not in MODEL_REGISTRY:
            return True
        info = MODEL_REGISTRY[model_id]
        filename = info.get("filename")
        url = info.get("url")

        if not filename or not url:
            return True

        model_path = os.path.join(self.weights_dir, filename)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            return True

        print(f"[ModelManager] Downloading {filename}...")
        if progress_callback:
            progress_callback(0, 100, f"Downloading model weight {filename}...")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req) as resp:
                total_size = int(resp.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 1024 * 64

                with open(model_path, 'wb') as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            pct = int((downloaded / total_size) * 100)
                            dl_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            msg = f"Downloading {filename}: {pct}% ({dl_mb:.1f}/{total_mb:.1f} MB)"
                            progress_callback(pct, 100, msg)

            print(f"[ModelManager] Downloaded {filename} successfully.")
            return True
        except Exception as e:
            print(f"[ModelManager] Could not download model weights ({e}).")
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except Exception:
                    pass
            return False

    def load_model(
        self, 
        model_id: str, 
        scale: int = 4, 
        tile_size: int = 256,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> BaseUpscaler:
        if tile_size < 0 or tile_size == -1:
            tile_size = get_adaptive_tile_size()

        if model_id not in MODEL_REGISTRY:
            model_id = "guided_edge"

        info = MODEL_REGISTRY[model_id]
        mtype = info["type"]
        upscaler = None

        if mtype == "fast_lanczos":
            upscaler = FastLanczosUpscaler(scale=scale, tile_size=0)
        elif mtype == "fast_nedi":
            upscaler = FastNEDIUpscaler(scale=scale, tile_size=0)
        elif mtype == "guided_edge":
            upscaler = GuidedEdgeUpscaler(scale=scale, tile_size=0)
        elif mtype == "raisr_patch":
            upscaler = RAISRUpscaler(scale=scale, tile_size=0)
        elif mtype == "xbrz_pattern":
            upscaler = xBRZRuleUpscaler(scale=scale, tile_size=0)
        elif mtype == "vector_contour":
            upscaler = VectorContourUpscaler(scale=scale, tile_size=0)
        
        if upscaler is None:
            filename = info["filename"]
            model_path = os.path.join(self.weights_dir, filename) if filename else None

            # Download weights if missing with live progress reporting
            if filename and not (os.path.exists(model_path) and os.path.getsize(model_path) > 0) and info["url"]:
                print(f"[ModelManager] Model weight missing. Downloading {filename}...")
                if progress_callback:
                    progress_callback(0, 100, f"Downloading model weight {filename}...")

                try:
                    req = urllib.request.Request(info["url"], headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                    with urllib.request.urlopen(req) as resp:
                        total_size = int(resp.headers.get('content-length', 0))
                        downloaded = 0
                        chunk_size = 1024 * 64

                        with open(model_path, 'wb') as f:
                            while True:
                                chunk = resp.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                                downloaded += len(chunk)

                                if progress_callback and total_size > 0:
                                    pct = int((downloaded / total_size) * 100)
                                    dl_mb = downloaded / (1024 * 1024)
                                    total_mb = total_size / (1024 * 1024)
                                    msg = f"Downloading {filename}: {pct}% ({dl_mb:.1f}/{total_mb:.1f} MB)"
                                    progress_callback(pct, 100, msg)

                    print(f"[ModelManager] Downloaded {filename} successfully.")
                except Exception as e:
                    if model_path and os.path.exists(model_path):
                        try:
                            os.remove(model_path)
                        except Exception:
                            pass
                    raise RuntimeError(f"Could not download model weights for {filename}: {e}")

            if mtype == "neural_dat":
                from .dat import DATUpscaler
                upscaler = DATUpscaler(
                    model_path=model_path,
                    scale=scale,
                    tile_size=tile_size
                )
            elif mtype == "neural_swinir":
                from .swin_ir import SwinIRUpscaler
                upscaler = SwinIRUpscaler(
                    model_path=model_path,
                    scale=scale,
                    tile_size=tile_size
                )
            elif mtype == "neural_esrgan":
                from .real_esrgan import RealESRGANUpscaler
                upscaler = RealESRGANUpscaler(
                    model_path=model_path, 
                    scale=scale, 
                    tile_size=tile_size
                )

        if upscaler is None:
            raise RuntimeError(f"Model '{model_id}' (type: '{mtype}') could not be initialized.")

        self._loaded_models.append(upscaler)
        return upscaler
