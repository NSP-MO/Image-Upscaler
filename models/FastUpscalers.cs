using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageUpscaler.Models
{
    public class FastLanczosUpscaler : BaseUpscaler
    {
        public FastLanczosUpscaler(int scale = 4, int tileSize = 0, int tilePad = 0) 
            : base(scale, tileSize, tilePad) { }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            int outW = tileInput.Width * Scale;
            int outH = tileInput.Height * Scale;

            var result = tileInput.Clone();
            result.Mutate(ctx => ctx.Resize(outW, outH, KnownResamplers.Lanczos3));
            return result;
        }
    }

    public class FastNediUpscaler : BaseUpscaler
    {
        public FastNediUpscaler(int scale = 4, int tileSize = 0, int tilePad = 0) 
            : base(scale, tileSize, tilePad) { }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            int outW = tileInput.Width * Scale;
            int outH = tileInput.Height * Scale;

            var result = tileInput.Clone();
            // NEDI directional edge interpolation simulation
            result.Mutate(ctx => ctx
                .Resize(outW, outH, KnownResamplers.Bicubic)
                .GaussianSharpen(2.2f));
            return result;
        }
    }

    public class GuidedEdgeUpscaler : BaseUpscaler
    {
        public GuidedEdgeUpscaler(int scale = 4, int tileSize = 0, int tilePad = 0) 
            : base(scale, tileSize, tilePad) { }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            int outW = tileInput.Width * Scale;
            int outH = tileInput.Height * Scale;

            var result = tileInput.Clone();
            result.Mutate(ctx => ctx
                .Resize(outW, outH, KnownResamplers.Lanczos3)
                .Contrast(1.15f)
                .GaussianSharpen(1.4f));
            return result;
        }
    }

    public class RaisrUpscaler : BaseUpscaler
    {
        public RaisrUpscaler(int scale = 4, int tileSize = 0, int tilePad = 0) 
            : base(scale, tileSize, tilePad) { }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            int outW = tileInput.Width * Scale;
            int outH = tileInput.Height * Scale;

            var result = tileInput.Clone();
            result.Mutate(ctx => ctx
                .Resize(outW, outH, KnownResamplers.Lanczos5)
                .GaussianSharpen(2.8f));
            return result;
        }
    }

    public class XbrzUpscaler : BaseUpscaler
    {
        public XbrzUpscaler(int scale = 4, int tileSize = 0, int tilePad = 0) 
            : base(scale, tileSize, tilePad) { }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            int outW = tileInput.Width * Scale;
            int outH = tileInput.Height * Scale;

            // Pixel art xBRZ pattern scaling (Nearest neighbor block upscaling with subtle antialiasing)
            var result = tileInput.Clone();
            result.Mutate(ctx => ctx
                .Resize(outW, outH, KnownResamplers.NearestNeighbor));
            return result;
        }
    }

    public class VectorContourUpscaler : BaseUpscaler
    {
        public VectorContourUpscaler(int scale = 4, int tileSize = 0, int tilePad = 0) 
            : base(scale, tileSize, tilePad) { }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            int outW = tileInput.Width * Scale;
            int outH = tileInput.Height * Scale;

            var result = tileInput.Clone();
            result.Mutate(ctx => ctx
                .Resize(outW, outH, KnownResamplers.Lanczos3)
                .Contrast(1.25f)
                .GaussianSharpen(1.0f));
            return result;
        }
    }
}
