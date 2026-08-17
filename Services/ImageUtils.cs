using System;
using System.IO;
using System.Windows.Media.Imaging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageUpscaler.Services
{
    public static class ImageUtils
    {
        public static BitmapImage ImageSharpToBitmapImage(Image<Rgb24> image)
        {
            using var ms = new MemoryStream();
            image.SaveAsPng(ms);
            ms.Position = 0;

            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = ms;
            bitmap.EndInit();
            bitmap.Freeze();

            return bitmap;
        }

        public static BitmapImage ImageSharpToBitmapImage(Image<Rgba32> image)
        {
            using var ms = new MemoryStream();
            image.SaveAsPng(ms);
            ms.Position = 0;

            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = ms;
            bitmap.EndInit();
            bitmap.Freeze();

            return bitmap;
        }

        public static Image<Rgba32> LoadImage(string filePath)
        {
            return Image.Load<Rgba32>(filePath);
        }

        public static BitmapImage LoadBitmapImage(string filePath)
        {
            using var ms = new MemoryStream(File.ReadAllBytes(filePath));
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = ms;
            bitmap.EndInit();
            bitmap.Freeze();
            return bitmap;
        }

        public static BitmapImage? CreateThumbnail(string filePath, int maxWidth, int maxHeight)
        {
            try
            {
                using var image = Image.Load<Rgba32>(filePath);
                image.Mutate(x => x.Resize(new ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size(maxWidth, maxHeight),
                    Mode = ResizeMode.Max
                }));
                return ImageSharpToBitmapImage(image);
            }
            catch
            {
                return null;
            }
        }
    }
}
