using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using ImageUpscaler.Models;

namespace ImageUpscaler.Services
{
    public class BatchProcessor
    {
        private readonly ModelManager _modelManager;

        public BatchProcessor(ModelManager modelManager)
        {
            _modelManager = modelManager;
        }

        public async Task ProcessFolderAsync(
            string inputFolder,
            string outputFolder,
            string modelId,
            int scale,
            int tileSize,
            Action<int, int, string>? progressCallback,
            CancellationToken cancellationToken = default)
        {
            if (!Directory.Exists(inputFolder)) return;
            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }

            string[] validExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
            var files = Directory.GetFiles(inputFolder)
                .Where(f => validExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                .ToArray();

            if (files.Length == 0)
            {
                progressCallback?.Invoke(100, 100, "No supported image files found in input directory.");
                return;
            }

            var upscaler = _modelManager.LoadModel(modelId, scale, tileSize, progressCallback);

            try
            {
                for (int i = 0; i < files.Length; i++)
                {
                    if (cancellationToken.IsCancellationRequested) break;

                    string file = files[i];
                    string filename = Path.GetFileName(file);
                    string outPath = Path.Combine(outputFolder, $"{Path.GetFileNameWithoutExtension(file)}_upscaled{Path.GetExtension(file)}");

                    int filePct = (int)((double)i / files.Length * 100);
                    progressCallback?.Invoke(filePct, 100, $"Batch ({i + 1}/{files.Length}): Upscaling {filename}...");

                    await Task.Run(() =>
                    {
                        using var src = ImageUtils.LoadImage(file);
                        using var upscaled = upscaler.UpscaleImage(src);
                        upscaled.Save(outPath);
                    }, cancellationToken);
                }

                progressCallback?.Invoke(100, 100, $"Batch complete! Processed {files.Length} images.");
            }
            finally
            {
                upscaler.UnloadModel();
            }
        }
    }
}
