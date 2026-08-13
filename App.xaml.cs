using System;
using System.Linq;
using System.Windows;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using ImageUpscaler.Models;

namespace ImageUpscaler
{
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            if (e.Args.Contains("--test"))
            {
                string modelId = e.Args.Length > 1 ? e.Args[1] : "realesrgan_x4_photo";
                Console.WriteLine($"[App] Testing PytorchUpscaler with model '{modelId}' in WPF environment...");
                try
                {
                    using var img = new Image<Rgba32>(64, 64);
                    var upscaler = new PytorchUpscaler(modelId, scale: 4, tileSize: -1);
                    using var outImg = upscaler.UpscaleImage(img, (pct, total, msg) =>
                    {
                        Console.WriteLine($"[WPF App Test Progress] {pct}% - {msg}");
                    });
                    outImg.Save($"wpf_test_out_{modelId}.png");
                    Console.WriteLine($"[App] SUCCESS for '{modelId}'! Output size: {outImg.Width}x{outImg.Height}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[App] FAILED for '{modelId}': {ex}");
                }
                Shutdown(0);
                return;
            }

            base.OnStartup(e);
        }
    }
}
