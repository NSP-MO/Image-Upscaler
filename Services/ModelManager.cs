using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using ImageUpscaler.Models;

namespace ImageUpscaler.Services
{
    public class ModelManager
    {
        private readonly string _weightsDir;
        private static readonly HttpClient _httpClient = new HttpClient();
        private readonly List<BaseUpscaler> _loadedUpscalers = new();
        private readonly object _lock = new object();

        public static readonly Dictionary<string, ModelInfo> ModelRegistry = new()
        {
            ["realesrgan_x4_photo"] = new ModelInfo
            {
                Id = "realesrgan_x4_photo",
                Name = "Real-ESRGAN Photo (x4)",
                Description = "High-fidelity Super-Resolution optimized for real-world photos.",
                Type = UpscalerType.NeuralEsrgan,
                DefaultScale = 4,
                Filename = "RealESRGAN_x4plus.pth",
                Url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            },
            ["remacri_x4"] = new ModelInfo
            {
                Id = "remacri_x4",
                Name = "Remacri Details (x4)",
                Description = "Community favorite model for lifelike facial textures, skin, and fabrics.",
                Type = UpscalerType.NeuralEsrgan,
                DefaultScale = 4,
                Filename = "remacri_x4.pth",
                Url = "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
            },
            ["bsrgan_x4"] = new ModelInfo
            {
                Id = "bsrgan_x4",
                Name = "BSRGAN Restorer (x4)",
                Description = "Specialized for restoring heavily degraded, blurry, or noisy old photos.",
                Type = UpscalerType.NeuralEsrgan,
                DefaultScale = 4,
                Filename = "bsrgan_x4.pth",
                Url = "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
            },
            ["dat_x4"] = new ModelInfo
            {
                Id = "dat_x4",
                Name = "DAT Transformer (x4)",
                Description = "SOTA ICCV 2023 Vision Transformer combining spatial & channel attention.",
                Type = UpscalerType.NeuralDat,
                DefaultScale = 4,
                Filename = "dat_x4.pth",
                Url = "https://huggingface.co/w-e-w/DAT/resolve/main/experiments/pretrained_models/DAT/DAT_x4.pth"
            },
            ["realesrgan_x4_anime"] = new ModelInfo
            {
                Id = "realesrgan_x4_anime",
                Name = "Real-ESRGAN Anime (x4)",
                Description = "Specialized neural network for sharp lines and digital illustrations.",
                Type = UpscalerType.NeuralEsrgan,
                DefaultScale = 4,
                Filename = "RealESRGAN_x4plus_anime_6B.pth",
                Url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            },
            ["realesrgan_x2_general"] = new ModelInfo
            {
                Id = "realesrgan_x2_general",
                Name = "Real-ESRGAN Fast (x2)",
                Description = "2x Super-Resolution for fast quality upscaling.",
                Type = UpscalerType.NeuralEsrgan,
                DefaultScale = 2,
                Filename = "RealESRGAN_x2plus.pth",
                Url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            },
            ["swinir_x4_classical"] = new ModelInfo
            {
                Id = "swinir_x4_classical",
                Name = "SwinIR Classical (x4)",
                Description = "SOTA Vision Transformer model for ultra-sharp photo restoration.",
                Type = UpscalerType.NeuralSwinir,
                DefaultScale = 4,
                Filename = "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
                Url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
            },
            ["swinir_x4_real"] = new ModelInfo
            {
                Id = "swinir_x4_real",
                Name = "Real-SwinIR Photo (x4)",
                Description = "Vision Transformer model specialized for removing JPEG compression and noise.",
                Type = UpscalerType.NeuralSwinir,
                DefaultScale = 4,
                Filename = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
                Url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
            },
            ["fast_lanczos"] = new ModelInfo
            {
                Id = "fast_lanczos",
                Name = "Fast Lanczos4 Baseline",
                Description = "Classic mathematical Lanczos-4 resampling with neutral sharpness.",
                Type = UpscalerType.FastLanczos,
                DefaultScale = 4
            },
            ["fast_nedi"] = new ModelInfo
            {
                Id = "fast_nedi",
                Name = "Fast NEDI (Edge-Directed)",
                Description = "Directional edge covariance interpolation. Sharp diagonal lines without aliasing.",
                Type = UpscalerType.FastNedi,
                DefaultScale = 4
            },
            ["guided_edge"] = new ModelInfo
            {
                Id = "guided_edge",
                Name = "Guided Edge Filter",
                Description = "Guided Filter edge-preserving enhancement. Crisp details without halo/ringing.",
                Type = UpscalerType.GuidedEdge,
                DefaultScale = 4
            },
            ["raisr_patch"] = new ModelInfo
            {
                Id = "raisr_patch",
                Name = "RAISR Patch Regression",
                Description = "Example-based patch covariance regression (Google RAISR). Fast adaptive edge refinement.",
                Type = UpscalerType.RaisrPatch,
                DefaultScale = 4
            },
            ["xbrz_pattern"] = new ModelInfo
            {
                Id = "xbrz_pattern",
                Name = "xBRZ Pattern Engine",
                Description = "Rule-based pixel pattern engine (xBRZ/ScaleNx). Smooth anti-aliased curves for 2D icons.",
                Type = UpscalerType.XbrzPattern,
                DefaultScale = 4
            },
            ["vector_contour"] = new ModelInfo
            {
                Id = "vector_contour",
                Name = "Vector Contour Engine",
                Description = "Vectorization & polygon curve tracing. Smooth infinite-scale vector contours for logos.",
                Type = UpscalerType.VectorContour,
                DefaultScale = 4
            }
        };

        public ModelManager(string weightsDir = "weights")
        {
            _weightsDir = weightsDir;
            if (!Directory.Exists(_weightsDir))
            {
                Directory.CreateDirectory(_weightsDir);
            }
        }

        public bool IsModelDownloaded(string modelId)
        {
            if (!ModelRegistry.TryGetValue(modelId, out var info)) return true;
            if (string.IsNullOrEmpty(info.Filename)) return true;

            string filePath = Path.Combine(_weightsDir, info.Filename);
            return File.Exists(filePath) && new FileInfo(filePath).Length > 0;
        }

        public List<ModelInfo> GetAvailableModels()
        {
            var list = new List<ModelInfo>();
            foreach (var kvp in ModelRegistry)
            {
                var info = kvp.Value;
                info.IsDownloaded = IsModelDownloaded(kvp.Key);
                list.Add(info);
            }
            return list;
        }

        public async Task<bool> DownloadModelWeightAsync(
            string modelId,
            Action<int, int, string>? progressCallback = null)
        {
            if (!ModelRegistry.TryGetValue(modelId, out var info)) return true;
            if (string.IsNullOrEmpty(info.Filename) || string.IsNullOrEmpty(info.Url)) return true;

            string filePath = Path.Combine(_weightsDir, info.Filename);
            if (File.Exists(filePath) && new FileInfo(filePath).Length > 0) return true;

            progressCallback?.Invoke(0, 100, $"Downloading model weight {info.Filename}...");

            try
            {
                using var response = await _httpClient.GetAsync(info.Url, HttpCompletionOption.ResponseHeadersRead);
                response.EnsureSuccessStatusCode();

                long? totalBytes = response.Content.Headers.ContentLength;
                using var contentStream = await response.Content.ReadAsStreamAsync();
                using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

                byte[] buffer = new byte[81920];
                long totalRead = 0;
                int read;

                while ((read = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                {
                    await fileStream.WriteAsync(buffer, 0, read);
                    totalRead += read;

                    if (totalBytes.HasValue && totalBytes.Value > 0)
                    {
                        int pct = (int)((double)totalRead / totalBytes.Value * 100);
                        double dlMb = totalRead / (1024.0 * 1024.0);
                        double totalMb = totalBytes.Value / (1024.0 * 1024.0);
                        progressCallback?.Invoke(pct, 100, $"Downloading {info.Filename}: {pct}% ({dlMb:F1}/{totalMb:F1} MB)");
                    }
                }

                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Download failed: {ex.Message}");
                if (File.Exists(filePath))
                {
                    try { File.Delete(filePath); } catch { }
                }
                return false;
            }
        }

        public BaseUpscaler LoadModel(
            string modelId,
            int scale = 4,
            int tileSize = 256,
            Action<int, int, string>? progressCallback = null)
        {
            BaseUpscaler upscaler;
            if (!ModelRegistry.TryGetValue(modelId, out var info))
            {
                upscaler = new GuidedEdgeUpscaler(scale, 0, 0);
            }
            else
            {
                switch (info.Type)
                {
                    case UpscalerType.FastLanczos:
                        upscaler = new FastLanczosUpscaler(scale, 0, 0);
                        break;
                    case UpscalerType.FastNedi:
                        upscaler = new FastNediUpscaler(scale, 0, 0);
                        break;
                    case UpscalerType.GuidedEdge:
                        upscaler = new GuidedEdgeUpscaler(scale, 0, 0);
                        break;
                    case UpscalerType.RaisrPatch:
                        upscaler = new RaisrUpscaler(scale, 0, 0);
                        break;
                    case UpscalerType.XbrzPattern:
                        upscaler = new XbrzUpscaler(scale, 0, 0);
                        break;
                    case UpscalerType.VectorContour:
                        upscaler = new VectorContourUpscaler(scale, 0, 0);
                        break;
                    case UpscalerType.NeuralEsrgan:
                    case UpscalerType.NeuralSwinir:
                    case UpscalerType.NeuralDat:
                        upscaler = new PytorchUpscaler(info.Id, scale, tileSize, 16);
                        break;
                    default:
                        upscaler = new GuidedEdgeUpscaler(scale, 0, 0);
                        break;
                }
            }

            lock (_lock)
            {
                _loadedUpscalers.Add(upscaler);
            }
            return upscaler;
        }

        public void UnloadAllModels()
        {
            lock (_lock)
            {
                foreach (var upscaler in _loadedUpscalers)
                {
                    try
                    {
                        upscaler.UnloadModel();
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Failed to unload model: {ex.Message}");
                    }
                }
                _loadedUpscalers.Clear();
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }
}
