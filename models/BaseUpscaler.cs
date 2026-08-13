using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageUpscaler.Models
{
    public abstract class BaseUpscaler
    {
        public int Scale { get; set; } = 4;
        public int NativeScale { get; set; } = 4;
        public int TileSize { get; set; } = 512;
        public int TilePad { get; set; } = 32;

        protected BaseUpscaler(int scale = 4, int tileSize = 512, int tilePad = 32)
        {
            Scale = scale;
            NativeScale = scale;
            TileSize = tileSize < 0 ? 512 : tileSize;
            TilePad = tilePad;
        }

        public virtual void UnloadModel()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        public virtual Image<Rgb24> UpscaleImage(
            Image<Rgb24> srcImage,
            Action<int, int, string>? progressCallback = null)
        {
            int targetScale = Scale;
            int nativeScale = NativeScale;

            progressCallback?.Invoke(0, 100, $"Running Native Pass ({nativeScale}x)...");

            var currentImg = DoSinglePass(srcImage, progressCallback);
            int currentScale = nativeScale;

            if (currentScale < targetScale)
            {
                int remScale = (int)Math.Round((double)targetScale / currentScale);
                progressCallback?.Invoke(50, 100, $"Running Edge Refinement Pass ({remScale}x)...");

                var edgeRefiner = new FastEdgeUpscaler(remScale);
                currentImg = edgeRefiner.UpscaleImage(currentImg);
                currentScale = targetScale;
            }

            if (currentScale != targetScale)
            {
                int finalW = (int)Math.Round((double)srcImage.Width * targetScale);
                int finalH = (int)Math.Round((double)srcImage.Height * targetScale);
                currentImg.Mutate(x => x.Resize(finalW, finalH, KnownResamplers.Lanczos3));
            }

            currentImg = AlignLuminanceAndColor(srcImage, currentImg);
            progressCallback?.Invoke(100, 100, "Upscaling complete.");

            return currentImg;
        }

        protected virtual Image<Rgb24> DoSinglePass(
            Image<Rgb24> image,
            Action<int, int, string>? progressCallback = null)
        {
            int h = image.Height;
            int w = image.Width;

            if (TileSize <= 0 || (h <= TileSize && w <= TileSize))
            {
                return ProcessTile(image);
            }

            return UpscaleTiled(image, progressCallback);
        }

        protected abstract Image<Rgb24> ProcessTile(Image<Rgb24> tileInput);

        protected Image<Rgb24> UpscaleTiled(
            Image<Rgb24> img,
            Action<int, int, string>? progressCallback = null)
        {
            int w = img.Width;
            int h = img.Height;
            int outW = w * Scale;
            int outH = h * Scale;

            float[,,] outputBuffer = new float[outH, outW, 3];
            float[,,] weightBuffer = new float[outH, outW, 3];

            int tilesX = (int)Math.Ceiling((double)w / TileSize);
            int tilesY = (int)Math.Ceiling((double)h / TileSize);
            int totalTiles = tilesX * tilesY;

            var tileArgs = new ConcurrentQueue<(int xIdx, int yIdx, int xStart, int yStart, int xEnd, int yEnd, int x1, int y1, int x2, int y2)>();

            for (int yIdx = 0; yIdx < tilesY; yIdx++)
            {
                for (int xIdx = 0; xIdx < tilesX; xIdx++)
                {
                    int xStart = xIdx * TileSize;
                    int yStart = yIdx * TileSize;
                    int xEnd = Math.Min((xIdx + 1) * TileSize, w);
                    int yEnd = Math.Min((yIdx + 1) * TileSize, h);

                    int x1 = Math.Max(xStart - TilePad, 0);
                    int y1 = Math.Max(yStart - TilePad, 0);
                    int x2 = Math.Min(xEnd + TilePad, w);
                    int y2 = Math.Min(yEnd + TilePad, h);

                    tileArgs.Enqueue((xIdx, yIdx, xStart, yStart, xEnd, yEnd, x1, y1, x2, y2));
                }
            }

            int completedTiles = 0;
            object lockObj = new object();

            Parallel.ForEach(tileArgs, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, args =>
            {
                int tw = args.x2 - args.x1;
                int th = args.y2 - args.y1;

                var tileIn = new Image<Rgb24>(tw, th);
                for (int y = 0; y < th; y++)
                {
                    for (int x = 0; x < tw; x++)
                    {
                        tileIn[x, y] = img[args.x1 + x, args.y1 + y];
                    }
                }

                using var tileOut = ProcessTile(tileIn);
                tileIn.Dispose();

                int thOut = tileOut.Height;
                int twOut = tileOut.Width;

                int padLeftOut = (args.xStart - args.x1) * Scale;
                int padRightOut = (args.x2 - args.xEnd) * Scale;
                int padTopOut = (args.yStart - args.y1) * Scale;
                int padBottomOut = (args.y2 - args.yEnd) * Scale;

                float[,] mask = BuildFeatherMask(thOut, twOut, padTopOut, padBottomOut, padLeftOut, padRightOut);

                int outX1 = args.x1 * Scale;
                int outY1 = args.y1 * Scale;

                lock (lockObj)
                {
                    for (int y = 0; y < thOut; y++)
                    {
                        int targetY = outY1 + y;
                        if (targetY >= outH) continue;

                        for (int x = 0; x < twOut; x++)
                        {
                            int targetX = outX1 + x;
                            if (targetX >= outW) continue;

                            float wVal = mask[y, x];
                            var px = tileOut[x, y];

                            outputBuffer[targetY, targetX, 0] += px.R * wVal;
                            outputBuffer[targetY, targetX, 1] += px.G * wVal;
                            outputBuffer[targetY, targetX, 2] += px.B * wVal;

                            weightBuffer[targetY, targetX, 0] += wVal;
                            weightBuffer[targetY, targetX, 1] += wVal;
                            weightBuffer[targetY, targetX, 2] += wVal;
                        }
                    }

                    completedTiles++;
                    int pct = (int)((double)completedTiles / totalTiles * 100);
                    progressCallback?.Invoke(pct, 100, $"Upscaling tile {completedTiles}/{totalTiles}...");
                }
            });

            var result = new Image<Rgb24>(outW, outH);
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    float wR = Math.Max(weightBuffer[y, x, 0], 1e-5f);
                    float wG = Math.Max(weightBuffer[y, x, 1], 1e-5f);
                    float wB = Math.Max(weightBuffer[y, x, 2], 1e-5f);

                    byte r = (byte)Math.Clamp((int)Math.Round(outputBuffer[y, x, 0] / wR), 0, 255);
                    byte g = (byte)Math.Clamp((int)Math.Round(outputBuffer[y, x, 1] / wG), 0, 255);
                    byte b = (byte)Math.Clamp((int)Math.Round(outputBuffer[y, x, 2] / wB), 0, 255);

                    result[x, y] = new Rgb24(r, g, b);
                }
            }

            return result;
        }

        protected float[,] BuildFeatherMask(int h, int w, int padTop, int padBottom, int padLeft, int padRight)
        {
            float[] yWeight = new float[h];
            float[] xWeight = new float[w];

            for (int i = 0; i < h; i++) yWeight[i] = 1.0f;
            for (int i = 0; i < w; i++) xWeight[i] = 1.0f;

            if (padTop > 0 && h >= padTop)
            {
                for (int i = 0; i < padTop; i++)
                {
                    yWeight[i] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * i / padTop));
                }
            }

            if (padBottom > 0 && h >= padBottom)
            {
                for (int i = 0; i < padBottom; i++)
                {
                    int idx = h - padBottom + i;
                    yWeight[idx] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * (padBottom - i) / padBottom));
                }
            }

            if (padLeft > 0 && w >= padLeft)
            {
                for (int i = 0; i < padLeft; i++)
                {
                    xWeight[i] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * i / padLeft));
                }
            }

            if (padRight > 0 && w >= padRight)
            {
                for (int i = 0; i < padRight; i++)
                {
                    int idx = w - padRight + i;
                    xWeight[idx] = (float)(0.5 - 0.5 * Math.Cos(Math.PI * (padRight - i) / padRight));
                }
            }

            float[,] mask = new float[h, w];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    mask[y, x] = yWeight[y] * xWeight[x];
                }
            }

            return mask;
        }

        protected Image<Rgb24> AlignLuminanceAndColor(Image<Rgb24> orig, Image<Rgb24> upscaled)
        {
            try
            {
                double origLSum = 0;
                long count = orig.Width * orig.Height;
                for (int y = 0; y < orig.Height; y++)
                {
                    for (int x = 0; x < orig.Width; x++)
                    {
                        var p = orig[x, y];
                        origLSum += (0.299 * p.R + 0.587 * p.G + 0.114 * p.B);
                    }
                }
                double origMeanL = origLSum / count;

                double upLSum = 0;
                long upCount = upscaled.Width * upscaled.Height;
                for (int y = 0; y < upscaled.Height; y++)
                {
                    for (int x = 0; x < upscaled.Width; x++)
                    {
                        var p = upscaled[x, y];
                        upLSum += (0.299 * p.R + 0.587 * p.G + 0.114 * p.B);
                    }
                }
                double upMeanL = upLSum / upCount;

                double diff = origMeanL - upMeanL;
                if (Math.Abs(diff) > 5.0)
                {
                    upscaled.Mutate(ctx =>
                    {
                        ctx.Brightness((float)(1.0 + (diff / 255.0)));
                    });
                }
            }
            catch
            {
                // Fallback silently if luminance adjustment fails
            }

            return upscaled;
        }
    }

    public class FastEdgeUpscaler : BaseUpscaler
    {
        public FastEdgeUpscaler(int scale = 2) : base(scale, 0, 0) { }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            int newW = tileInput.Width * Scale;
            int newH = tileInput.Height * Scale;

            var copy = tileInput.Clone();
            copy.Mutate(x => x.Resize(newW, newH, KnownResamplers.Lanczos3).GaussianSharpen(1.2f));
            return copy;
        }
    }
}
