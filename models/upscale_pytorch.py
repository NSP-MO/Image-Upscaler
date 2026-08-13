import sys
import os

# Register Windows PyTorch DLL directory before importing torch to prevent WinError 126
if sys.platform == "win32":
    py_dir = os.path.dirname(sys.executable)
    torch_lib_dirs = [
        os.path.join(py_dir, "Lib", "site-packages", "torch", "lib"),
        os.path.join(py_dir, "Lib", "site-packages", "torch"),
        os.path.join(py_dir, "DLLs")
    ]
    for d in torch_lib_dirs:
        if os.path.exists(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

import argparse
from PIL import Image
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model_manager import ModelManager, get_adaptive_tile_size

def main():
    parser = argparse.ArgumentParser(description="PyTorch Upscale Bridge")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID")
    parser.add_argument("--input", type=str, required=True, help="Input Image Path")
    parser.add_argument("--output", type=str, required=True, help="Output Image Path")
    parser.add_argument("--scale", type=int, default=4, help="Scale Factor")
    parser.add_argument("--tile_size", type=int, default=-1, help="Tile Size (-1 = Auto VRAM Adaptive)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    print(f"Active Hardware: {device_name} (CUDA={cuda_available})")

    resolved_tile = args.tile_size
    if resolved_tile < 0 or resolved_tile == -1:
        resolved_tile = get_adaptive_tile_size()
        print(f"Auto VRAM Adaptive Tile Size resolved: {resolved_tile}px", flush=True)
    else:
        print(f"User Tile Size selected: {resolved_tile}px", flush=True)

    print(f"Loading PyTorch model {args.model_id} (scale={args.scale}x, tile={resolved_tile}px)...", flush=True)

    # Instantiate ModelManager and load native PyTorch upscaler
    manager = ModelManager(weights_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights")))
    upscaler = manager.load_model(args.model_id, scale=args.scale, tile_size=resolved_tile)

    print(f"Opening image {args.input}...", flush=True)
    img = Image.open(args.input)

    def progress_callback(current, total, msg):
        print(f"[PROGRESS] {current}/{total} - [{device_name}] {msg}", flush=True)

    print(f"Executing neural upscaling pass on [{device_name}]...", flush=True)
    upscaled_img = upscaler.upscale_image(img, progress_callback=progress_callback)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    upscaled_img.save(args.output)
    print(f"SUCCESS: Saved upscaled output to {args.output} ({upscaled_img.size[0]}x{upscaled_img.size[1]}px)", flush=True)

if __name__ == "__main__":
    main()
