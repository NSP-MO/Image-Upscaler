using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ImageUpscaler.Models
{
    public class PytorchUpscaler : BaseUpscaler
    {
        private readonly string _modelId;
        private Process? _currentProcess;
        private readonly object _processLock = new object();

        public PytorchUpscaler(string modelId, int scale = 4, int tileSize = 512, int tilePad = 32)
            : base(scale, tileSize, tilePad)
        {
            _modelId = modelId;
        }

        public override void UnloadModel()
        {
            lock (_processLock)
            {
                if (_currentProcess != null)
                {
                    try
                    {
                        if (!_currentProcess.HasExited)
                        {
                            _currentProcess.Kill(true); // Terminate process and child process tree
                        }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Error killing PyTorch process: {ex.Message}");
                    }
                    try
                    {
                        _currentProcess.Dispose();
                    }
                    catch { }
                    _currentProcess = null;
                }
            }

            base.UnloadModel();
        }

        protected override Image<Rgb24> ProcessTile(Image<Rgb24> tileInput)
        {
            return tileInput.Clone();
        }

        private static string ResolvePythonExecutable()
        {
            string[] candidatePaths = new[]
            {
                @"C:\Python\Python313\python.exe",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Python\Python313\python.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Python\Python312\python.exe"),
                "python.exe"
            };

            foreach (var path in candidatePaths)
            {
                if (File.Exists(path))
                {
                    return path;
                }
            }

            return "python";
        }

        private static (string scriptPath, string projectRoot) ResolveBridgeScriptAndRoot()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string currentDir = Directory.GetCurrentDirectory();

            string[] searchRoots = new[]
            {
                baseDir,
                Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..")),
                Path.GetFullPath(Path.Combine(baseDir, "..", "..")),
                currentDir
            };

            foreach (var root in searchRoots)
            {
                string candidate = Path.Combine(root, "exec", "upscale_pytorch.py");
                if (File.Exists(candidate))
                {
                    return (candidate, root);
                }
            }

            // Fallback default
            return (Path.Combine(baseDir, "exec", "upscale_pytorch.py"), baseDir);
        }

        public override Image<Rgb24> UpscaleRgb(Image<Rgb24> srcImage, Action<int, int, string>? progressCallback = null)
        {
            string tempInput = Path.Combine(Path.GetTempPath(), $"image_in_{Guid.NewGuid():N}.png");
            string tempOutput = Path.Combine(Path.GetTempPath(), $"image_out_{Guid.NewGuid():N}.png");

            try
            {
                progressCallback?.Invoke(0, 100, $"Preparing PyTorch Neural Model ({_modelId})...");
                srcImage.Save(tempInput);

                var (bridgeScript, projectRoot) = ResolveBridgeScriptAndRoot();

                if (!File.Exists(bridgeScript))
                {
                    throw new FileNotFoundException($"PyTorch bridge script not found at {bridgeScript}");
                }

                string pythonExe = ResolvePythonExecutable();

                var psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{bridgeScript}\" --model_id \"{_modelId}\" --input \"{tempInput}\" --output \"{tempOutput}\" --scale {Scale} --tile_size {TileSize}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = projectRoot
                };

                psi.EnvironmentVariables["PYTHONPATH"] = projectRoot;

                var errBuilder = new StringBuilder();
                lock (_processLock)
                {
                    _currentProcess = new Process { StartInfo = psi };
                }

                _currentProcess.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrWhiteSpace(e.Data))
                    {
                        errBuilder.AppendLine(e.Data);
                    }
                };

                _currentProcess.Start();
                _currentProcess.BeginErrorReadLine();

                string? line;
                while ((line = _currentProcess.StandardOutput.ReadLine()) != null)
                {
                    if (line.StartsWith("[PROGRESS]"))
                    {
                        string parts = line.Substring("[PROGRESS]".Length).Trim();
                        progressCallback?.Invoke(50, 100, parts);
                    }
                    else if (!string.IsNullOrWhiteSpace(line))
                    {
                        progressCallback?.Invoke(25, 100, line);
                    }
                }

                _currentProcess.WaitForExit();

                if (File.Exists(tempOutput))
                {
                    var result = Image.Load<Rgb24>(tempOutput);
                    progressCallback?.Invoke(100, 100, "PyTorch Neural Inference Complete.");
                    return result;
                }
                else
                {
                    string err = errBuilder.ToString();
                    System.Diagnostics.Debug.WriteLine($"PyTorch Bridge execution failed (exit code {_currentProcess.ExitCode}): {err}");
                    throw new InvalidOperationException($"PyTorch Bridge failed: {err}");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"PyTorch execution error: {ex.Message}");
                progressCallback?.Invoke(50, 100, $"PyTorch Error: {ex.Message}");
            }
            finally
            {
                lock (_processLock)
                {
                    if (_currentProcess != null)
                    {
                        try
                        {
                            if (!_currentProcess.HasExited)
                            {
                                _currentProcess.Kill(true);
                            }
                        }
                        catch { }
                        _currentProcess.Dispose();
                        _currentProcess = null;
                    }
                }

                if (File.Exists(tempInput)) try { File.Delete(tempInput); } catch { }
                if (File.Exists(tempOutput)) try { File.Delete(tempOutput); } catch { }
            }

            // Fallback to Fast Edge upscaler if PyTorch process fails
            progressCallback?.Invoke(50, 100, "Falling back to Edge Refinement...");
            var fallback = new GuidedEdgeUpscaler(Scale, TileSize, TilePad);
            return fallback.UpscaleRgb(srcImage, progressCallback);
        }
    }
}
