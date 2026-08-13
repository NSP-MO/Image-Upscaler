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

        private static bool IsValidPythonExecutable(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;

            // Reject WindowsApps App Execution Alias stub
            if (path.Contains(@"Microsoft\WindowsApps", StringComparison.OrdinalIgnoreCase)) return false;

            try
            {
                if (File.Exists(path))
                {
                    var info = new FileInfo(path);
                    // Real python.exe is > 50KB, Microsoft Store stub is 0 bytes or tiny wrapper
                    if (info.Length > 50000) return true;
                }
            }
            catch { }

            return false;
        }

        private static string ResolvePythonExecutable()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

            string[] candidatePaths = new[]
            {
                // Portable python inside app directory
                Path.Combine(baseDir, "python_runtime", "python.exe"),
                Path.Combine(baseDir, "python", "python.exe"),

                // Standard system installs
                @"C:\Python313\python.exe",
                @"C:\Python312\python.exe",
                @"C:\Python311\python.exe",
                @"C:\Python310\python.exe",
                @"C:\Python\Python313\python.exe",
                @"C:\Python\Python312\python.exe",
                @"C:\Python\Python311\python.exe",
                Path.Combine(localAppData, @"Programs\Python\Python313\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python312\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python311\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python310\python.exe")
            };

            foreach (var path in candidatePaths)
            {
                if (IsValidPythonExecutable(path))
                {
                    return path;
                }
            }

            // Search PATH environment variable for non-WindowsApps python.exe
            string? pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                string[] pathDirs = pathEnv.Split(Path.PathSeparator);
                foreach (var dir in pathDirs)
                {
                    if (dir.Contains(@"Microsoft\WindowsApps", StringComparison.OrdinalIgnoreCase)) continue;

                    string fullPath = Path.Combine(dir.Trim(), "python.exe");
                    if (IsValidPythonExecutable(fullPath))
                    {
                        return fullPath;
                    }
                }
            }

            throw new InvalidOperationException(
                "Python 3.10+ runtime was not detected on this system (found Microsoft Store alias stub instead of installed Python).\n\n" +
                "Please run ImageUpscaler-Setup.msi with 'Python & PyTorch AI Engine' checked to install Python, or install Python 3.10+ manually.");
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
                string candidate = Path.Combine(root, "models", "upscale_pytorch.py");
                if (File.Exists(candidate))
                {
                    return (candidate, root);
                }
            }

            // Fallback default
            return (Path.Combine(baseDir, "models", "upscale_pytorch.py"), baseDir);
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

                string pythonExe = ImageUpscaler.Services.PythonBootstrapper.EnsurePythonEnvironmentAsync(progressCallback).GetAwaiter().GetResult();

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
                    else if (line.StartsWith("[PyTorch Bridge] Active Hardware:"))
                    {
                        string hwInfo = line.Substring("[PyTorch Bridge]".Length).Trim();
                        progressCallback?.Invoke(10, 100, hwInfo);
                    }
                    else if (!string.IsNullOrWhiteSpace(line))
                    {
                        progressCallback?.Invoke(25, 100, line);
                    }
                }

                _currentProcess.WaitForExit();

                if (_currentProcess.ExitCode == 0 && File.Exists(tempOutput))
                {
                    var result = Image.Load<Rgb24>(tempOutput);
                    progressCallback?.Invoke(100, 100, "PyTorch Neural Inference Complete.");
                    return result;
                }
                else
                {
                    string err = errBuilder.ToString();
                    System.Diagnostics.Debug.WriteLine($"PyTorch Bridge execution failed (exit code {_currentProcess.ExitCode}): {err}");
                    throw new InvalidOperationException($"PyTorch execution failed: {err}");
                }
            }
            catch (System.ComponentModel.Win32Exception winEx)
            {
                throw new InvalidOperationException(
                    $"Python runtime executable ('{winEx.Message}') was not found on this system.\n\n" +
                    $"To run PyTorch Neural Models ({_modelId}), please install Python 3.10+ (with PyTorch) or place a portable Python runtime in 'python_runtime/'.\n\n" +
                    $"Alternatively, you can select any of the Built-in C# Native Upscalers (Fast Lanczos4, Fast NEDI, Guided Edge, RAISR, xBRZ, Vector Contour) which run 100% natively without Python!", winEx);
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
        }
    }
}
