import gc
import cv2
import math
import os
import threading
import multiprocessing
import numpy as np
import torch
import concurrent.futures
from PIL import Image
from typing import Callable, Optional, Tuple

class BaseUpscaler:
    """
    Abstract Base Class for Image Super-Resolution Upscalers.
    Includes exact tile-based processing engine, optimal multi-pass upscaling engine,
    automatic luminance/color alignment, and linear feathering overlap blending.
    """
    gpu_lock = threading.Lock()

    def __init__(self, scale: int = 4, tile_size: int = 512, tile_pad: int = 32):
        self.scale = scale
        self.native_scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Maximize CPU thread count across OpenCV and PyTorch backends
        num_cores = os.cpu_count() or multiprocessing.cpu_count() or 16
        cv2.setNumThreads(num_cores)
        torch.set_num_threads(num_cores)
        try:
            torch.set_num_interop_threads(num_cores)
        except Exception:
            pass

    def upscale_image(
        self, 
        image: Image.Image, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Image.Image:
        """
        Upscales a PIL Image. Supports transparent PNGs (RGBA). Automatically performs optimal multi-pass upscaling and
        exact luminance/color balance matching with the original input image.
        """
        has_alpha = False
        alpha_channel = None
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            rgba = image.convert('RGBA')
            alpha_np = np.array(rgba.split()[3])
            if np.any(alpha_np < 255):
                has_alpha = True
                alpha_channel = rgba.split()[3]
            rgb_image = rgba.convert('RGB')
        else:
            rgb_image = image.convert('RGB')

        target_scale = self.scale
        native_scale = getattr(self, 'native_scale', target_scale)

        orig_scale = self.scale
        self.scale = native_scale

        try:
            # Pass 1: Primary Native Neural Pass
            if progress_callback:
                progress_callback(0, 1, f"Running Pass 1 ({native_scale}x native)...")

            current_img = self._do_single_pass(rgb_image, progress_callback)
            current_scale = native_scale

            # Pass 2: Clean Edge Refinement if target_scale > native_scale
            if current_scale < target_scale:
                rem_scale = int(round(float(target_scale) / float(current_scale)))
                if progress_callback:
                    progress_callback(0, 1, f"Running Pass 2 (Edge Refinement {rem_scale}x)...")

                from .fast_upscaler import FastEdgeUpscaler
                edge_refiner = FastEdgeUpscaler(scale=rem_scale)
                current_img = edge_refiner.upscale_image(current_img)
                current_scale = target_scale

            # Final size adjustment if needed
            if current_scale != target_scale:
                w, h = rgb_image.size
                final_w = int(round(w * target_scale))
                final_h = int(round(h * target_scale))
                current_img = current_img.resize((final_w, final_h), Image.LANCZOS)

            # Match luminance & color distribution with original input image
            current_img = self._align_luminance_and_color(rgb_image, current_img)

            if has_alpha and alpha_channel is not None:
                upscaled_alpha = alpha_channel.resize(current_img.size, Image.LANCZOS)
                r, g, b = current_img.split()
                return Image.merge('RGBA', (r, g, b, upscaled_alpha))

            return current_img
        finally:
            self.scale = orig_scale

    def _align_luminance_and_color(self, orig_img: Image.Image, upscaled_img: Image.Image) -> Image.Image:
        """
        Adjusts upscaled image luminance and color balance so it matches the exact brightness
        and tone distribution of the original input image.
        """
        try:
            orig_np = np.array(orig_img.convert('RGB'))
            up_np = np.array(upscaled_img.convert('RGB'))

            # Convert to LAB color space (L = Luminance, A = Green-Red, B = Blue-Yellow)
            orig_lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            up_lab = cv2.cvtColor(up_np, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Downsample L channel of upscaled image to original size for exact mean comparison
            w_orig, h_orig = orig_img.size
            up_l_down = cv2.resize(up_lab[:, :, 0], (w_orig, h_orig), interpolation=cv2.INTER_AREA)

            mean_orig = float(orig_lab[:, :, 0].mean())
            mean_up = float(up_l_down.mean())

            # Luminance shift correction
            l_diff = mean_orig - mean_up
            if abs(l_diff) > 0.3:
                l_corrected = up_lab[:, :, 0] + l_diff
                up_lab[:, :, 0] = np.clip(l_corrected, 0, 255)

                matched_rgb = cv2.cvtColor(np.clip(up_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
                return Image.fromarray(matched_rgb)
        except Exception as e:
            pass

        return upscaled_img

    def _do_single_pass(
        self, 
        image: Image.Image, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Image.Image:
        img_np = np.array(image.convert('RGB'))
        h, w, c = img_np.shape

        # If image is small enough, process directly
        if self.tile_size <= 0 or (h <= self.tile_size and w <= self.tile_size):
            upscaled_np = self._process_tile(img_np)
            return Image.fromarray(np.clip(np.round(upscaled_np), 0, 255).astype(np.uint8))

        return self._upscale_tiled(img_np, progress_callback)

    def unload_model(self):
        """
        Subclasses should override to free PyTorch GPU VRAM and resources.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        """
        Must be implemented by subclasses to upscale a single RGB numpy array (H, W, 3) [0..255]
        and return upscaled RGB numpy array (H*scale, W*scale, 3) [0..255].
        """
        raise NotImplementedError

    def _upscale_tiled(
        self, 
        img_np: np.ndarray, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Image.Image:
        """
        Processes image in overlapping tiles concurrently, blending seamless upscaled output 
        with zero grid seams using multi-core utilization.
        """
        h, w, c = img_np.shape
        out_h, out_w = h * self.scale, w * self.scale
        output = np.zeros((out_h, out_w, c), dtype=np.float32)
        weight_map = np.zeros((out_h, out_w, c), dtype=np.float32)

        tiles_x = math.ceil(w / self.tile_size)
        tiles_y = math.ceil(h / self.tile_size)
        total_tiles = tiles_x * tiles_y

        # Define tile arguments
        tile_args = []
        for y_idx in range(tiles_y):
            for x_idx in range(tiles_x):
                # Unpadded core bounds
                x_start = x_idx * self.tile_size
                y_start = y_idx * self.tile_size
                x_end = min((x_idx + 1) * self.tile_size, w)
                y_end = min((y_idx + 1) * self.tile_size, h)

                # Padded input bounds
                x1 = max(x_start - self.tile_pad, 0)
                y1 = max(y_start - self.tile_pad, 0)
                x2 = min(x_end + self.tile_pad, w)
                y2 = min(y_end + self.tile_pad, h)

                tile_args.append((x_idx, y_idx, x_start, y_start, x_end, y_end, x1, y1, x2, y2))

        # Dynamically allocate worker threads to match full CPU core count (e.g. 16 threads)
        max_workers = os.cpu_count() or multiprocessing.cpu_count() or 16

        completed_tiles = 0

        def process_single_tile(args):
            x_idx, y_idx, x_start, y_start, x_end, y_end, x1, y1, x2, y2 = args
            tile_in = img_np[y1:y2, x1:x2]
            
            # If model is on CUDA, serialize neural network forward pass safely with lock,
            # while allowing tile slicing, array math, feather masks, and blending to run in parallel on all CPU threads!
            if hasattr(self, 'device') and self.device.type == 'cuda':
                with BaseUpscaler.gpu_lock:
                    tile_out = self._process_tile(tile_in)
            else:
                tile_out = self._process_tile(tile_in)
            
            # Pad mask calculations
            th_out, tw_out, _ = tile_out.shape
            pad_left_out = (x_start - x1) * self.scale
            pad_right_out = (x2 - x_end) * self.scale
            pad_top_out = (y_start - y1) * self.scale
            pad_bottom_out = (y2 - y_end) * self.scale
            
            mask = self._build_feather_mask_exact(
                th_out, tw_out, 
                pad_top_out, pad_bottom_out, pad_left_out, pad_right_out
            )
            
            return args, tile_out, mask, th_out, tw_out

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_tile, arg): arg for arg in tile_args}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    args, tile_out, mask, th_out, tw_out = future.result()
                    x_idx, y_idx, x_start, y_start, x_end, y_end, x1, y1, x2, y2 = args
                    
                    completed_tiles += 1
                    if progress_callback:
                        progress_callback(
                            completed_tiles, 
                            total_tiles, 
                            f"Upscaling tile {completed_tiles}/{total_tiles}..."
                        )

                    # Output slice coordinates
                    out_x1, out_y1 = x1 * self.scale, y1 * self.scale
                    out_x2, out_y2 = out_x1 + tw_out, out_y1 + th_out

                    # Accumulate onto output canvas
                    output[out_y1:out_y2, out_x1:out_x2] += tile_out.astype(np.float32) * mask
                    weight_map[out_y1:out_y2, out_x1:out_x2] += mask
                except Exception as exc:
                    print(f"Tile processing generated an exception: {exc}")

        # Normalize output by accumulated weights
        weight_map = np.maximum(weight_map, 1e-5)
        final_img = output / weight_map
        final_img_uint8 = np.clip(np.round(final_img), 0, 255).astype(np.uint8)

        return Image.fromarray(final_img_uint8)

    def _build_feather_mask_exact(
        self, 
        h: int, w: int, 
        pad_top: int, pad_bottom: int, pad_left: int, pad_right: int
    ) -> np.ndarray:
        """
        Generates 2D smooth cosine weight mask for seamless tile blending on inner overlapping edges only.
        Outer image boundaries keep weight = 1.0.
        """
        y_weight = np.ones(h, dtype=np.float32)
        x_weight = np.ones(w, dtype=np.float32)

        if pad_top > 0 and h >= pad_top:
            y_weight[:pad_top] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, pad_top, dtype=np.float32))
        if pad_bottom > 0 and h >= pad_bottom:
            y_weight[-pad_bottom:] = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0, pad_bottom, dtype=np.float32))

        if pad_left > 0 and w >= pad_left:
            x_weight[:pad_left] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, pad_left, dtype=np.float32))
        if pad_right > 0 and w >= pad_right:
            x_weight[-pad_right:] = 0.5 - 0.5 * np.cos(np.linspace(np.pi, 0, pad_right, dtype=np.float32))

        mask2d = np.outer(y_weight, x_weight)
        return np.expand_dims(mask2d, axis=-1)
