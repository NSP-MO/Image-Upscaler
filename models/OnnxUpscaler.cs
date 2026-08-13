using System;
using System.Collections.Generic;
using System.IO;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageUpscaler.Models
{
    public class OnnxUpscaler : BaseUpscaler
    {
        private readonly string? _modelPath;
        private InferenceSession? _session;

        public OnnxUpscaler(string? modelPath, int scale = 4, int tileSize = 256, int tilePad = 16)
            : base(scale, tileSize, tilePad)
        {
            _modelPath = modelPath;
            InitSession();
        }

        public override void UnloadModel()
        {
            if (_session != null)
            {
                try
                {
                    _session.Dispose();
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error disposing ONNX InferenceSession: {ex.Message}");
                }
                _session = null;
            }

            base.UnloadModel();
        }

        private void InitSession()
        {
            if (!string.IsNullOrEmpty(_modelPath) && File.Exists(_modelPath))
            {
                try
                {
                    var options = new SessionOptions();
                    options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    _session = new InferenceSession(_modelPath, options);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Failed to load ONNX model: {ex.Message}");
                    _session = null;
                }
            }
        }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            if (_session != null)
            {
                try
                {
                    return RunInference(tileInput);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"ONNX Inference error: {ex.Message}");
                }
            }

            // Fallback high-fidelity processing if ONNX model is missing or failed
            int outW = tileInput.Width * Scale;
            int outH = tileInput.Height * Scale;
            var fallback = tileInput.Clone();
            fallback.Mutate(ctx => ctx.Resize(outW, outH, KnownResamplers.Lanczos3).GaussianSharpen(1.4f));
            return fallback;
        }

        private Image<Rgb24> RunInference(Image<Rgb24> tileInput)
        {
            int h = tileInput.Height;
            int w = tileInput.Width;

            var inputTensor = new DenseTensor<float>(new[] { 1, 3, h, w });
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    var px = tileInput[x, y];
                    inputTensor[0, 0, y, x] = px.R / 255.0f;
                    inputTensor[0, 1, y, x] = px.G / 255.0f;
                    inputTensor[0, 2, y, x] = px.B / 255.0f;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _session!.Run(inputs);
            using var outputValue = results[0];
            var outputTensor = outputValue.AsTensor<float>();

            int outH = outputTensor.Dimensions[2];
            int outW = outputTensor.Dimensions[3];

            var result = new Image<Rgb24>(outW, outH);
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    byte r = (byte)Math.Clamp((int)Math.Round(outputTensor[0, 0, y, x] * 255.0f), 0, 255);
                    byte g = (byte)Math.Clamp((int)Math.Round(outputTensor[0, 1, y, x] * 255.0f), 0, 255);
                    byte b = (byte)Math.Clamp((int)Math.Round(outputTensor[0, 2, y, x] * 255.0f), 0, 255);

                    result[x, y] = new Rgb24(r, g, b);
                }
            }

            return result;
        }
    }
}
