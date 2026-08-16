using System;
using System.IO;
using System.Net.Http;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ImageUpscaler.Services
{
    public static class PythonBootstrapper
    {
        private static bool IsValidPythonExecutable(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;

            // Filter out WindowsApps Microsoft Store stub executables
            if (path.Contains(@"Microsoft\WindowsApps", StringComparison.OrdinalIgnoreCase)) return false;

            try
            {
                if (File.Exists(path))
                {
                    var info = new FileInfo(path);
                    if (info.Length > 50000) return true;
                }
            }
            catch { }

            return false;
        }

        public static bool IsVisualCppRedistributableInstalled()
        {
            string system32 = Environment.GetFolderPath(Environment.SpecialFolder.System);
            string vcomp140 = Path.Combine(system32, "vcomp140.dll");
            string vcruntime140 = Path.Combine(system32, "vcruntime140.dll");
            return File.Exists(vcomp140) && File.Exists(vcruntime140);
        }

        public static string? ResolvePythonExecutable()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

            string[] candidatePaths = new[]
            {
                Path.Combine(baseDir, "python_runtime", "python.exe"),
                Path.Combine(baseDir, "python", "python.exe"),
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

            return null;
        }

        public static bool ArePythonModulesInstalled(string pythonPath)
        {
            if (!IsValidPythonExecutable(pythonPath)) return false;

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = "-c \"import torch, torchvision, timm, PIL, cv2, numpy\"",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden
                };

                using (var proc = Process.Start(psi))
                {
                    if (proc != null)
                    {
                        proc.WaitForExit(5000);
                        return proc.ExitCode == 0;
                    }
                }
            }
            catch { }

            return false;
        }

        public static (bool hasMissing, string description) GetMissingDependenciesDescription()
        {
            bool missingVc = !IsVisualCppRedistributableInstalled();
            string? resolvedPython = ResolvePythonExecutable();
            bool missingPython = resolvedPython == null;
            bool missingModules = !missingPython && !ArePythonModulesInstalled(resolvedPython!);

            if (!missingVc && !missingPython && !missingModules)
            {
                return (false, string.Empty);
            }

            var sb = new StringBuilder();
            sb.AppendLine("The selected Neural Model requires the following runtime dependencies to be installed on your system:");
            sb.AppendLine();

            long totalDownloadMb = 0;
            long totalDiskMb = 0;

            if (missingVc)
            {
                sb.AppendLine("• Microsoft Visual C++ 2015-2022 Redistributable x64");
                sb.AppendLine("  - Purpose: Required for PyTorch C++ native DLLs (vcomp140.dll)");
                sb.AppendLine("  - Download Size: ~14 MB | Required Disk Space: ~35 MB");
                sb.AppendLine();
                totalDownloadMb += 14;
                totalDiskMb += 35;
            }

            if (missingPython)
            {
                sb.AppendLine("• Python 3.11 64-bit Runtime & PyTorch Neural Engine");
                sb.AppendLine("  - Purpose: Required for Real-ESRGAN, SwinIR, and DAT Neural Models");
                sb.AppendLine("  - Download Size: ~204 MB | Required Disk Space: ~750 MB");
                sb.AppendLine();
                totalDownloadMb += 204;
                totalDiskMb += 750;
            }
            else if (missingModules)
            {
                sb.AppendLine("• Python Packages & Neural Modules (torch, torchvision, timm, pillow, opencv-python, numpy)");
                sb.AppendLine("  - Purpose: Python runtime detected, but required Neural Model packages are missing.");
                sb.AppendLine("  - Download Size: ~180 MB | Required Disk Space: ~650 MB");
                sb.AppendLine();
                totalDownloadMb += 180;
                totalDiskMb += 650;
            }

            sb.AppendLine("------------------------------------------------------------------");
            sb.AppendLine($"TOTAL ESTIMATED DOWNLOAD SIZE : ~{totalDownloadMb} MB");
            sb.AppendLine($"TOTAL ESTIMATED DISK SPACE   : ~{totalDiskMb} MB ({(totalDiskMb / 1024.0):F1} GB)");
            sb.AppendLine("------------------------------------------------------------------");
            sb.AppendLine();
            sb.AppendLine("Would you like to grant consent to download and install these required dependencies now?");
            sb.AppendLine();
            sb.AppendLine("• Click 'Yes' to grant consent and start automated setup.");
            sb.AppendLine("• Click 'No' to cancel and switch to built-in native C# upscalers.");

            return (true, sb.ToString());
        }

        private static async Task EnsureVisualCppRedistributableAsync(Action<int, int, string>? progressCallback, CancellationToken cancellationToken)
        {
            if (IsVisualCppRedistributableInstalled())
            {
                return;
            }

            progressCallback?.Invoke(2, 100, "Installing Microsoft Visual C++ 2015-2022 Redistributable x64...");

            string vcRedistPath = Path.Combine(Path.GetTempPath(), "vc_redist.x64.exe");
            string vcUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe";

            try
            {
                using (var client = new HttpClient())
                using (var response = await client.GetAsync(vcUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                {
                    response.EnsureSuccessStatusCode();
                    using (var content = await response.Content.ReadAsStreamAsync(cancellationToken))
                    using (var fileStream = new FileStream(vcRedistPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true))
                    {
                        await content.CopyToAsync(fileStream, cancellationToken);
                    }
                }

                var psi = new ProcessStartInfo
                {
                    FileName = vcRedistPath,
                    Arguments = "/quiet /norestart",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden
                };

                using (var proc = Process.Start(psi))
                {
                    if (proc != null)
                    {
                        int vcPct = 2;
                        while (!proc.HasExited)
                        {
                            await Task.Delay(300, cancellationToken);
                            if (vcPct < 4)
                            {
                                vcPct++;
                                progressCallback?.Invoke(vcPct, 100, $"Installing Microsoft Visual C++ Redistributable ({vcPct}%)...");
                            }
                        }
                        await proc.WaitForExitAsync(cancellationToken);
                    }
                }
            }
            catch { }
            finally
            {
                if (File.Exists(vcRedistPath)) try { File.Delete(vcRedistPath); } catch { }
            }
        }

        public static async Task<string> EnsurePythonEnvironmentAsync(Action<int, int, string>? progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Ensure Visual C++ Redistributable DLLs are present for PyTorch
            await EnsureVisualCppRedistributableAsync(progressCallback, cancellationToken);

            string? existingPython = ResolvePythonExecutable();
            bool hasPython = !string.IsNullOrEmpty(existingPython);
            bool modulesInstalled = hasPython && ArePythonModulesInstalled(existingPython!);

            if (hasPython && modulesInstalled)
            {
                return existingPython!;
            }

            string targetPython = existingPython ?? string.Empty;

            if (!hasPython)
            {
                cancellationToken.ThrowIfCancellationRequested();
                progressCallback?.Invoke(5, 100, "Python runtime missing. Preparing automated download...");

                string tempInstallerPath = Path.Combine(Path.GetTempPath(), "python-3.11.9-amd64.exe");
                string pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe";

                using (var httpClient = new HttpClient())
                {
                    httpClient.Timeout = TimeSpan.FromMinutes(10);
                    using (var response = await httpClient.GetAsync(pythonUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                    {
                        response.EnsureSuccessStatusCode();
                        long totalBytes = response.Content.Headers.ContentLength ?? 58000000;

                        using (var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken))
                        using (var fileStream = new FileStream(tempInstallerPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true))
                        {
                            var buffer = new byte[8192];
                            long totalRead = 0;
                            int bytesRead;

                            while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
                            {
                                cancellationToken.ThrowIfCancellationRequested();
                                await fileStream.WriteAsync(buffer, 0, bytesRead, cancellationToken);
                                totalRead += bytesRead;

                                double mbRead = Math.Round((double)totalRead / (1024.0 * 1024.0), 1);
                                double mbTotal = Math.Round((double)totalBytes / (1024.0 * 1024.0), 1);
                                int pct = (int)((double)totalRead / totalBytes * 40.0) + 10;

                                progressCallback?.Invoke(pct, 100, $"Downloading Python 3.11 Runtime: {mbRead} MB / {mbTotal} MB ({pct}%)");
                            }
                        }
                    }
                }

                cancellationToken.ThrowIfCancellationRequested();
                progressCallback?.Invoke(55, 100, "Installing Python 3.11 Runtime (55%)...");

                var psi = new ProcessStartInfo
                {
                    FileName = tempInstallerPath,
                    Arguments = "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 SimpleInstall=1",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden
                };

                using (var proc = Process.Start(psi))
                {
                    if (proc != null)
                    {
                        int pyPct = 55;
                        while (!proc.HasExited)
                        {
                            await Task.Delay(400, cancellationToken);
                            if (pyPct < 64)
                            {
                                pyPct++;
                                progressCallback?.Invoke(pyPct, 100, $"Installing Python 3.11 Runtime ({pyPct}%)...");
                            }
                        }
                        await proc.WaitForExitAsync(cancellationToken);
                    }
                }

                if (File.Exists(tempInstallerPath))
                {
                    try { File.Delete(tempInstallerPath); } catch { }
                }

                cancellationToken.ThrowIfCancellationRequested();
                string? installedPython = ResolvePythonExecutable();
                if (string.IsNullOrEmpty(installedPython))
                {
                    throw new InvalidOperationException("Automated Python 3.11 installation finished, but python executable could not be resolved.");
                }
                targetPython = installedPython;
            }

            cancellationToken.ThrowIfCancellationRequested();
            progressCallback?.Invoke(65, 100, "Configuring PyTorch & Neural Packages (65%)...");

            var pipPsi = new ProcessStartInfo
            {
                FileName = targetPython,
                Arguments = "-m pip install --upgrade pip",
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };

            using (var pipProc = Process.Start(pipPsi))
            {
                if (pipProc != null) await pipProc.WaitForExitAsync(cancellationToken);
            }

            cancellationToken.ThrowIfCancellationRequested();
            string reqFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "requirements.txt");
            string pipArgs = File.Exists(reqFile)
                ? $"-m pip install -r \"{reqFile}\""
                : "-m pip install torch torchvision timm pillow opencv-python numpy";

            var reqPipPsi = new ProcessStartInfo
            {
                FileName = targetPython,
                Arguments = pipArgs,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };

            using (var reqProc = Process.Start(reqPipPsi))
            {
                if (reqProc != null)
                {
                    int currentPipPct = 68;
                    reqProc.OutputDataReceived += (s, e) =>
                    {
                        if (!string.IsNullOrWhiteSpace(e.Data))
                        {
                            string data = e.Data.Trim();
                            if (data.Contains("Downloading"))
                            {
                                if (currentPipPct < 88) currentPipPct += 3;
                                progressCallback?.Invoke(currentPipPct, 100, $"Downloading Neural Package ({currentPipPct}%): {data}");
                            }
                            else if (data.Contains("Installing collected packages") || data.Contains("Installing"))
                            {
                                currentPipPct = 92;
                                progressCallback?.Invoke(92, 100, "Installing Neural Engine Packages (92%)...");
                            }
                            else if (data.Contains("Successfully installed"))
                            {
                                currentPipPct = 98;
                                progressCallback?.Invoke(98, 100, "Finalizing Neural Engine Setup (98%)...");
                            }
                        }
                    };
                    reqProc.BeginOutputReadLine();

                    while (!reqProc.HasExited)
                    {
                        await Task.Delay(500, cancellationToken);
                        if (currentPipPct < 90)
                        {
                            currentPipPct++;
                            progressCallback?.Invoke(currentPipPct, 100, $"Setting up PyTorch Neural Engine ({currentPipPct}%)...");
                        }
                    }

                    await reqProc.WaitForExitAsync(cancellationToken);
                }
            }

            progressCallback?.Invoke(100, 100, "Python & PyTorch Neural Engine setup complete!");
            return targetPython;
        }
    }
}
