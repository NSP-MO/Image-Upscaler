import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from .base_upscaler import BaseUpscaler


def pixel_unshuffle(x, scale=2):
    """
    Pixel Unshuffle operation: reorganizes spatial elements into channel dimension.
    Input: (B, C, H, W) -> Output: (B, C * scale * scale, H / scale, W / scale)
    Uses PyTorch native F.pixel_unshuffle for 100% exact channel alignment.
    """
    return F.pixel_unshuffle(x, scale)


class ResidualDenseBlock_5C(nn.Module):
    """Residual Dense Block (RDB) with 5 convolutions."""
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB)."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Official Real-ESRGAN Architecture with Residual in Residual Dense Blocks.
    Supports pixel unshuffle (in_nc=12 for x2plus) with automatic odd dimension padding,
    variable blocks (nb=23 or nb=6), and scale factors (2x, 4x, 8x).
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.in_nc = in_nc
        self.is_unshuffle = (in_nc == 12)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling layers
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = 0
        pad_w = 0

        # Auto-pad odd spatial dimensions to even numbers for pixel unshuffle
        if self.is_unshuffle:
            pad_h = (2 - (h % 2)) % 2
            pad_w = (2 - (w % 2)) % 2
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        if self.is_unshuffle:
            fea = pixel_unshuffle(x, scale=2)
            fea = self.conv_first(fea)
        else:
            fea = self.conv_first(x)

        body_fea = self.conv_body(self.body(fea))
        fea = fea + body_fea

        if self.scale == 2 and not self.is_unshuffle:
            fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale == 4 or (self.scale == 2 and self.is_unshuffle):
            fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale == 8:
            fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.conv_up3(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        # Crop output back to target scale size if padded
        if pad_h > 0 or pad_w > 0:
            target_h = h * self.scale
            target_w = w * self.scale
            out = out[:, :, :target_h, :target_w]

        return out


class RealESRGANUpscaler(BaseUpscaler):
    """
    PyTorch Neural Upscaler running Real-ESRGAN / RRDBNet architecture with CUDA GPU acceleration.
    Automatically detects model native scale (2x for x2plus, 4x for x4plus) and block counts.
    Supports Remacri, BSRGAN, and standard Real-ESRGAN weight keys.
    """
    def __init__(self, model_path: str = None, scale: int = 4, tile_size: int = 512, tile_pad: int = 32):
        super().__init__(scale=scale, tile_size=tile_size, tile_pad=tile_pad)
        self.model_path = model_path
        self.net = None
        self._init_model()

    def _init_model(self):
        nb = 23
        in_nc = 3
        state_dict = None

        if self.model_path and os.path.exists(self.model_path):
            try:
                raw_dict = torch.load(self.model_path, map_location=self.device)
                if 'params_ema' in raw_dict:
                    state_dict = raw_dict['params_ema']
                elif 'params' in raw_dict:
                    state_dict = raw_dict['params']
                else:
                    state_dict = raw_dict

                # Normalize key names from legacy ESRGAN / BSRGAN / Remacri formats
                new_dict = {}
                for k, v in state_dict.items():
                    new_k = k
                    if new_k.startswith('model.0.'):
                        new_k = new_k.replace('model.0.', 'conv_first.')
                    elif new_k.startswith('model.1.sub.23.'):
                        new_k = new_k.replace('model.1.sub.23.', 'conv_body.')
                    elif new_k.startswith('model.1.sub.'):
                        new_k = new_k.replace('model.1.sub.', 'body.')
                    elif new_k.startswith('model.3.'):
                        new_k = new_k.replace('model.3.', 'conv_up1.')
                    elif new_k.startswith('model.6.'):
                        new_k = new_k.replace('model.6.', 'conv_up2.')
                    elif new_k.startswith('model.8.'):
                        new_k = new_k.replace('model.8.', 'conv_hr.')
                    elif new_k.startswith('model.10.'):
                        new_k = new_k.replace('model.10.', 'conv_last.')

                    new_k = new_k.replace('RRDB_trunk.', 'body.')
                    new_k = new_k.replace('trunk_conv.', 'conv_body.')
                    new_k = new_k.replace('upconv1.', 'conv_up1.')
                    new_k = new_k.replace('upconv2.', 'conv_up2.')
                    new_k = new_k.replace('HRconv.', 'conv_hr.')
                    new_k = new_k.replace('RDB1', 'rdb1').replace('RDB2', 'rdb2').replace('RDB3', 'rdb3')
                    new_k = new_k.replace('.conv1.0.', '.conv1.').replace('.conv2.0.', '.conv2.')
                    new_k = new_k.replace('.conv3.0.', '.conv3.').replace('.conv4.0.', '.conv4.').replace('.conv5.0.', '.conv5.')
                    new_dict[new_k] = v
                state_dict = new_dict

                # Detect in_nc (3 for x4plus, 12 for x2plus)
                if 'conv_first.weight' in state_dict:
                    in_nc = state_dict['conv_first.weight'].shape[1]

                # Detect num blocks nb dynamically from weight keys
                block_indices = set()
                for k in state_dict.keys():
                    if k.startswith('body.'):
                        parts = k.split('.')
                        block_indices.add(int(parts[1]))
                if block_indices:
                    nb = len(block_indices)
            except Exception as e:
                print(f"[RealESRGAN] Error loading weight file: {e}")
                state_dict = None

        # Set native_scale (2x for x2plus, 4x for x4plus)
        self.native_scale = 2 if in_nc == 12 else 4

        # Build net using its native_scale for single pass tile processing
        self.net = RRDBNet(in_nc=in_nc, out_nc=3, nf=64, nb=nb, gc=32, scale=self.native_scale)
        self.net.to(self.device)

        if state_dict:
            try:
                self.net.load_state_dict(state_dict, strict=True)
                print(f"[RealESRGAN] Successfully loaded 100% of weights (in_nc={in_nc}, native_scale={self.native_scale}x, nb={nb}) from {self.model_path}")
            except Exception as e:
                print(f"[RealESRGAN] Warning loading state dict: {e}")

        self.net.eval()

    def unload_model(self):
        """
        Unloads PyTorch neural network from GPU, empties CUDA memory cache, and triggers garbage collection.
        """
        if hasattr(self, 'net') and self.net is not None:
            try:
                self.net.cpu()
            except Exception:
                pass
            del self.net
            self.net = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        print("[RealESRGAN] VRAM and GPU memory successfully freed.")

    def _process_tile(self, tile_np: np.ndarray) -> np.ndarray:
        if self.net is None:
            self._init_model()

        # Convert RGB numpy HWC [0..255] to Torch CHW float [0..1]
        img_tensor = torch.from_numpy(tile_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.net(img_tensor)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        # Convert back to HWC numpy [0..255]
        out_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        return np.clip(out_np, 0, 255).astype(np.uint8)
