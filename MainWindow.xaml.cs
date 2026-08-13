using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using ImageUpscaler.Models;
using ImageUpscaler.Services;
using ImageUpscaler.UI.Dialogs;

namespace ImageUpscaler
{
    public partial class MainWindow : Window
    {
        private readonly ModelManager _modelManager;
        private string? _currentPath;
        private bool _isFolderSelected = false;
        private Image<Rgba32>? _currentOriginalImage;
        private Image<Rgba32>? _currentUpscaledImage;
        private BitmapImage? _originalBitmap;
        private BitmapImage? _upscaledBitmap;

        public MainWindow()
        {
            InitializeComponent();
            _modelManager = new ModelManager("weights");

            // Wire UI Events
            DropZoneControl.FileSelected += OnPathSelected;
            CompareCanvasControl.FileSelected += OnPathSelected;
            CompareCanvasControl.BatchRequested += OnBatchRequested;
            SidebarControl.SelectFileRequested += OnPromptSelectFileOrFolder;
            SidebarControl.UpscaleRequested += OnUpscaleRequested;
            SidebarControl.SaveRequested += OnSaveRequested;
            SidebarControl.BatchRequested += OnBatchRequested;

            // Load models list into sidebar
            var models = _modelManager.GetAvailableModels();
            SidebarControl.PopulateModels(models);
        }

        private void OnPromptSelectFileOrFolder()
        {
            var dlg = new OpenFileDialog
            {
                Filter = "Image Files (*.png;*.jpg;*.jpeg;*.webp;*.bmp)|*.png;*.jpg;*.jpeg;*.webp;*.bmp|All Files (*.*)|*.*",
                Title = "Select Image File to Upscale"
            };

            if (dlg.ShowDialog() == true)
            {
                OnPathSelected(dlg.FileName);
            }
        }

        private void OnPathSelected(string path)
        {
            if (Directory.Exists(path))
            {
                // Folder loaded
                _currentPath = path;
                _isFolderSelected = true;
                _currentOriginalImage?.Dispose();
                _currentOriginalImage = null;
                _originalBitmap = null;
                _currentUpscaledImage?.Dispose();
                _currentUpscaledImage = null;

                string[] validExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
                var files = Directory.GetFiles(path)
                    .Where(f => validExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                    .ToArray();

                CompareCanvasControl.SetFolderView(path, files.Length);
                DropZoneControl.Visibility = Visibility.Collapsed;
                CompareCanvasControl.Visibility = Visibility.Visible;

                SidebarControl.SetSaveButtonEnabled(false);
                StatusTextBlock.Text = $"Loaded folder: {Path.GetFileName(path)} ({files.Length} images). Click Batch Folder Upscale to process.";
            }
            else if (File.Exists(path))
            {
                // Single image loaded
                try
                {
                    _currentPath = path;
                    _isFolderSelected = false;
                    _currentOriginalImage?.Dispose();
                    _currentOriginalImage = ImageUtils.LoadImage(path);
                    _originalBitmap = ImageUtils.LoadBitmapImage(path);

                    _currentUpscaledImage?.Dispose();
                    _currentUpscaledImage = null;
                    _upscaledBitmap = null;

                    // Show ONLY single loaded image preview initially (NO clip geometry!)
                    CompareCanvasControl.SetSingleImage(_originalBitmap);
                    DropZoneControl.Visibility = Visibility.Collapsed;
                    CompareCanvasControl.Visibility = Visibility.Visible;

                    SidebarControl.SetSaveButtonEnabled(false);
                    StatusTextBlock.Text = $"Loaded image: {Path.GetFileName(path)} ({_currentOriginalImage.Width}x{_currentOriginalImage.Height}px). Ready to upscale.";
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Could not load image: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private async void OnUpscaleRequested()
        {
            if (_isFolderSelected && !string.IsNullOrEmpty(_currentPath))
            {
                OnBatchRequested();
                return;
            }

            if (_currentOriginalImage == null)
            {
                MessageBox.Show("Please drag and drop or select an image first.", "No Image Selected", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var model = SidebarControl.SelectedModel;
            if (model == null) return;

            int scale = SidebarControl.SelectedScale;
            int tileSize = SidebarControl.SelectedTileSize;

            SidebarControl.SetUpscaleButtonEnabled(false);
            MainProgressBar.Visibility = Visibility.Visible;
            MainProgressBar.Value = 0;

            StatusTextBlock.Text = $"Preparing {model.Name}...";

            // Check & Download model weights if needed
            if (!model.IsDownloaded && !string.IsNullOrEmpty(model.Url))
            {
                bool downloaded = await _modelManager.DownloadModelWeightAsync(model.Id, (pct, total, msg) =>
                {
                    Dispatcher.Invoke(() =>
                    {
                        MainProgressBar.Value = pct;
                        StatusTextBlock.Text = msg;
                    });
                });

                if (!downloaded)
                {
                    MessageBox.Show($"Could not download model weights for '{model.Name}'. Please check your network connection.", "Model Weights Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    StatusTextBlock.Text = $"Failed to download model weights for {model.Name}.";
                    return;
                }
            }

            // Run Upscaling in Background Task
            try
            {
                await Task.Run(() =>
                {
                    var upscaler = _modelManager.LoadModel(model.Id, scale, tileSize, (pct, total, msg) =>
                    {
                        Dispatcher.Invoke(() =>
                        {
                            MainProgressBar.Value = pct;
                            StatusTextBlock.Text = msg;
                        });
                    });

                    try
                    {
                        _currentUpscaledImage?.Dispose();
                        _currentUpscaledImage = upscaler.UpscaleImage(_currentOriginalImage, (pct, total, msg) =>
                        {
                            Dispatcher.Invoke(() =>
                            {
                                MainProgressBar.Value = pct;
                                StatusTextBlock.Text = msg;
                            });
                        });
                    }
                    finally
                    {
                        upscaler.UnloadModel();
                    }
                });

                if (_currentUpscaledImage != null && _originalBitmap != null)
                {
                    _upscaledBitmap = ImageUtils.ImageSharpToBitmapImage(_currentUpscaledImage);
                    // NOW show the comparison view (Before vs After) with split slider
                    CompareCanvasControl.SetComparisonImages(_originalBitmap, _upscaledBitmap);
                    SidebarControl.SetSaveButtonEnabled(true);
                    StatusTextBlock.Text = $"Upscaling complete! Model: {model.Name} | Resolution: {_currentUpscaledImage.Width}x{_currentUpscaledImage.Height}px ({scale}x)";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Upscaling error: {ex.Message}", "Processing Error", MessageBoxButton.OK, MessageBoxImage.Error);
                StatusTextBlock.Text = "Upscaling failed.";
            }
            finally
            {
                SidebarControl.SetUpscaleButtonEnabled(true);
                MainProgressBar.Visibility = Visibility.Collapsed;
            }
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            base.OnClosing(e);

            try
            {
                StatusTextBlock.Text = "Closing software and stopping processes...";
                _modelManager.UnloadAllModels();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error during MainWindow close cleanup: {ex.Message}");
            }
        }

        private void OnSaveRequested()
        {
            if (_currentUpscaledImage == null) return;

            var dlg = new SaveFileDialog
            {
                Filter = "PNG Image (*.png)|*.png|JPEG Image (*.jpg)|*.jpg|WebP Image (*.webp)|*.webp",
                Title = "Save Upscaled Image",
                FileName = _currentPath != null && File.Exists(_currentPath) ? $"{Path.GetFileNameWithoutExtension(_currentPath)}_upscaled.png" : "upscaled.png"
            };

            if (dlg.ShowDialog() == true)
            {
                try
                {
                    _currentUpscaledImage.Save(dlg.FileName);
                    MessageBox.Show($"Saved image successfully to:\n{dlg.FileName}", "Saved Successfully", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Could not save image: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void OnBatchRequested()
        {
            var model = SidebarControl.SelectedModel ?? _modelManager.GetAvailableModels()[0];
            int scale = SidebarControl.SelectedScale;
            int tileSize = SidebarControl.SelectedTileSize;

            var dlg = new BatchDialog(_modelManager, model, scale, tileSize)
            {
                Owner = this
            };

            if (_isFolderSelected && !string.IsNullOrEmpty(_currentPath) && Directory.Exists(_currentPath))
            {
                dlg.InputFolderBox.Text = _currentPath;
                dlg.OutputFolderBox.Text = Path.Combine(_currentPath, "Upscaled_Output");
            }

            dlg.ShowDialog();
        }
    }
}