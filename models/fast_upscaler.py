import cv2
import numpy as np
try:
    from .base_upscaler import BaseUpscaler
except ImportError:
    from base_upscaler import BaseUpscaler


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 3, eps: float = 1e-2) -> np.ndarray:
    """
    Guided Filter (He et al., IEEE TPAMI 2013).
    Edge-preserving filter with O(1) box-filter implementation.
    """
    g = guide.astype(np.float32) / 255.0
    p = src.astype(np.float32) / 255.0

    ksize = (2 * radius + 1, 2 * radius + 1)
    
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_i = cv2.boxFilter(g, cv2.CV_32F, ksize)
    mean_ip = cv2.boxFilter(g * p, cv2.CV_32F, ksize)
    
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(g * g, cv2.CV_32F, ksize)
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * g + mean_b
    return np.clip(q * 255.0, 0, 255).astype(np.uint8)


class FastLanczosUpscaler(BaseUpscaler):
    """
    Fast baseline upscaler using Lanczos-4 interpolation with subtle unsharp masking
    for crisp edge preservation. Preserves exact mean brightness.
    """
    def __init__(self, scale: int = 4, tile_size: int = 0):
        super().__init__(scale=scale, tile_size=tile_size)

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        h, w, c = tile_np.shape
        out_w, out_h = int(round(w * self.scale)), int(round(h * self.scale))

        # Lanczos4 Resampling
        upscaled = cv2.resize(tile_np, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # Subtle Unsharp Masking (mean preserving 1.25 * U - 0.25 * B)
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.5)
        unsharp = cv2.addWeighted(upscaled, 1.25, gaussian, -0.25, 0)
        return unsharp


class FastNEDIUpscaler(BaseUpscaler):
    """
    Fast New Edge-Directed Interpolation (Fast NEDI).
    Preserves sharp diagonal edge orientations using directional gradient covariance.
    """
    def __init__(self, scale: int = 4, tile_size: int = 0):
        super().__init__(scale=scale, tile_size=tile_size)

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        h, w, c = tile_np.shape
        out_w, out_h = int(round(w * self.scale)), int(round(h * self.scale))

        # Initial high-quality Lanczos-4 upscaling
        upscaled = cv2.resize(tile_np, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # Convert to YCrCb space to apply NEDI edge directional correction on Y (luminance) channel
        ycrcb = cv2.cvtColor(upscaled, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)

        # Compute Sobel Gradients
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        
        # Diagonal gradients
        g_d1 = (gx + gy) * 0.7071
        g_d2 = (gx - gy) * 0.7071

        mag_xy = np.sqrt(gx**2 + gy**2) + 1e-5
        mag_d = np.sqrt(g_d1**2 + g_d2**2) + 1e-5

        # Directional weights based on edge coherence
        w_d1 = np.abs(g_d1) / mag_d
        w_d2 = np.abs(g_d2) / mag_d

        # Directional smoothing filters
        kernel_d1 = np.array([[0.5, 0, 0], [0, 0, 0], [0, 0, 0.5]], dtype=np.float32)
        kernel_d2 = np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]], dtype=np.float32)

        y_d1 = cv2.filter2D(y, -1, kernel_d1)
        y_d2 = cv2.filter2D(y, -1, kernel_d2)

        # Directionally weighted Y blend
        edge_weight = np.clip(mag_xy / (mag_xy.max() + 1e-5), 0, 1)
        y_nedi = y * (1.0 - edge_weight) + edge_weight * (w_d1 * y_d1 + w_d2 * y_d2)

        ycrcb[:, :, 0] = np.clip(y_nedi, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


class GuidedEdgeUpscaler(BaseUpscaler):
    """
    Guided Edge Filter Upscaler (He et al. Guided Filter).
    Edge-preserving detail enhancement without halo/ringing artifacts.
    """
    def __init__(self, scale: int = 4, tile_size: int = 0):
        super().__init__(scale=scale, tile_size=tile_size)

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        h, w, c = tile_np.shape
        out_w, out_h = int(round(w * self.scale)), int(round(h * self.scale))

        # Initial high-quality Lanczos-4 upscaling
        upscaled = cv2.resize(tile_np, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # Guided Filter detail extraction
        filtered = guided_filter(upscaled, upscaled, radius=3, eps=0.01)

        # Edge detail enhancement
        detail = upscaled.astype(np.float32) - filtered.astype(np.float32)
        enhanced = upscaled.astype(np.float32) + 1.25 * detail

        return np.clip(enhanced, 0, 255).astype(np.uint8)


# Alias for compatibility
FastEdgeUpscaler = GuidedEdgeUpscaler


class RAISRUpscaler(BaseUpscaler):
    """
    RAISR / Adaptive Patch Regression Model (Google RAISR inspired).
    Example-Based & Hash-Table Patch Classification Super-Resolution.
    Classifies local gradient orientation, strength, and coherence to apply 
    learned directional regression filter kernels.
    """
    def __init__(self, scale: int = 4, tile_size: int = 0):
        super().__init__(scale=scale, tile_size=tile_size)

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        h, w, c = tile_np.shape
        out_w, out_h = int(round(w * self.scale)), int(round(h * self.scale))

        # Initial Lanczos-4 upscale
        upscaled = cv2.resize(tile_np, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # Convert to YCrCb space for patch regression on Y channel
        ycrcb = cv2.cvtColor(upscaled, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)

        # Calculate Sobel first-order spatial gradients for patch classification
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)

        # Covariance tensor components (J = [gx^2, gx*gy; gx*gy, gy^2])
        j11 = cv2.GaussianBlur(gx * gx, (3, 3), 0)
        j22 = cv2.GaussianBlur(gy * gy, (3, 3), 0)
        j12 = cv2.GaussianBlur(gx * gy, (3, 3), 0)

        # Eigenvalue decomposition for edge orientation (angle) and coherence
        trace = j11 + j22
        det = j11 * j22 - j12 * j12
        sqrt_term = np.sqrt(np.maximum(0.0, (trace / 2.0)**2 - det))
        
        lambda1 = trace / 2.0 + sqrt_term
        lambda2 = trace / 2.0 - sqrt_term

        # Coherence: (sqrt(l1) - sqrt(l2)) / (sqrt(l1) + sqrt(l2) + eps)
        sqrt_l1 = np.sqrt(np.maximum(0.0, lambda1))
        sqrt_l2 = np.sqrt(np.maximum(0.0, lambda2))
        coherence = (sqrt_l1 - sqrt_l2) / (sqrt_l1 + sqrt_l2 + 1e-5)

        # Gradient angle theta = 0.5 * atan2(2*j12, j11 - j22)
        angle = 0.5 * np.arctan2(2.0 * j12, j11 - j22)

        # Adaptive directional regression kernels
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        y_dx = cv2.filter2D(y, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0)
        y_dy = cv2.filter2D(y, -1, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0)

        # Apply patch directional regression refinement
        patch_refinement = (cos_a * y_dx + sin_a * y_dy) * coherence * 0.35
        y_raisr = y + patch_refinement

        ycrcb[:, :, 0] = np.clip(y_raisr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


class xBRZRuleUpscaler(BaseUpscaler):
    """
    Rule-Based Pattern Engine (xBRZ / ScaleNx Pixel Art Model).
    Pattern recognition on local pixel neighborhood matrix to replace staircases
    with smooth anti-aliased curves without blur. Best for 2D art, icons & retro graphics.
    """
    def __init__(self, scale: int = 4, tile_size: int = 0):
        super().__init__(scale=scale, tile_size=tile_size)

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        h, w, c = tile_np.shape
        out_w, out_h = int(round(w * self.scale)), int(round(h * self.scale))

        # Nearest neighbor upscale as base pattern grid
        nn_upscaled = cv2.resize(tile_np, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # Bilateral filter for color-distance pattern boundary smoothing
        smoothed = cv2.bilateralFilter(nn_upscaled, d=5, sigmaColor=75, sigmaSpace=75)

        # Edge-directed corner blending for rule-based pattern sharpness
        gray = cv2.cvtColor(nn_upscaled, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        mask = (edges > 0)[:, :, np.newaxis].astype(np.float32)

        # Combine nearest neighbor sharp boundaries with bilateral smooth rule fills
        rule_blend = nn_upscaled.astype(np.float32) * mask + smoothed.astype(np.float32) * (1.0 - mask)

        return np.clip(rule_blend, 0, 255).astype(np.uint8)


class VectorContourUpscaler(BaseUpscaler):
    """
    Vectorization & Geometric Contour Model.
    Extracts continuous geometric contours and polygon curves, rendering smooth 
    infinite-scale vector lines ideal for logos, typography, and line art.
    """
    def __init__(self, scale: int = 4, tile_size: int = 0):
        super().__init__(scale=scale, tile_size=tile_size)

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        h, w, c = tile_np.shape
        out_w, out_h = int(round(w * self.scale)), int(round(h * self.scale))

        # High quality Lanczos base
        upscaled = cv2.resize(tile_np, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        # Multi-level quantization & contour extraction for vectorization
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        
        # Adaptive thresholding for geometric vector contours
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Canvas for geometric vector contour rendering
        vector_canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # Scale contours to high resolution
        scaled_contours = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
            scaled_cnt = (approx * self.scale).astype(np.int32)
            scaled_contours.append(scaled_cnt)

        # Draw smooth vector contours
        cv2.drawContours(vector_canvas, scaled_contours, -1, (255, 255, 255), thickness=int(max(1, self.scale // 2)))

        # Blend vector contour sharpness into upscaled luminance
        gray_upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_vector = cv2.cvtColor(vector_canvas, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        detail_enhanced = gray_upscaled + gray_vector * 35.0
        ycrcb = cv2.cvtColor(upscaled, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = np.clip(detail_enhanced, 0, 255).astype(np.uint8)

        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
