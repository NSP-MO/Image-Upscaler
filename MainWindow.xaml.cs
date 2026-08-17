using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
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
        private readonly ObservableCollection<UpscaleItem> _images = new();
        private UpscaleItem? _selectedItem;

        // Interactive Zoom & Pan State
        private double _currentZoom = 1.0;
        private System.Windows.Point _lastMousePos;
        private bool _isPanning = false;

        // Split Curtain Slider State
        private double _splitRatio = 0.5;
        private bool _isDraggingSlider = false;
        private string? _lastExportedFilePath;

        public MainWindow()
        {
            InitializeComponent();
            _modelManager = new ModelManager("weights");

            if (ImagesListBox != null)
            {
                ImagesListBox.ItemsSource = _images;
            }

            // Populate Models Dropdown
            var availableModels = _modelManager.GetAvailableModels();
            if (ModelSelectionComboBox != null)
            {
                ModelSelectionComboBox.ItemsSource = availableModels;
                if (availableModels.Count > 0)
                {
                    ModelSelectionComboBox.SelectedIndex = 0;
                }
            }

            UpdateImageCount();
            UpdateWorkspaceState();
        }

        #region File Import & Drag-and-Drop

        private void OnImportImagesClick(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "Image Files (*.png;*.jpg;*.jpeg;*.webp;*.bmp)|*.png;*.jpg;*.jpeg;*.webp;*.bmp|All Files (*.*)|*.*",
                Title = "Select Image Files to Upscale",
                Multiselect = true
            };

            if (dlg.ShowDialog() == true)
            {
                LoadImageFiles(dlg.FileNames);
            }
        }

        private void Window_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[]? files = (string[]?)e.Data.GetData(DataFormats.FileDrop);
                if (files != null && files.Length > 0)
                {
                    string[] validExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
                    var imageFiles = files
                        .Where(f => File.Exists(f) && validExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                        .ToArray();

                    if (imageFiles.Length > 0)
                    {
                        LoadImageFiles(imageFiles);
                    }
                    else
                    {
                        // Check if a directory was dropped
                        var dirs = files.Where(Directory.Exists).ToArray();
                        if (dirs.Length > 0)
                        {
                            var folderImages = Directory.GetFiles(dirs[0])
                                .Where(f => validExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                                .ToArray();
                            if (folderImages.Length > 0)
                            {
                                LoadImageFiles(folderImages);
                            }
                        }
                    }
                }
            }
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.V && (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control)
            {
                OnPasteClipboardClick(sender, e);
                e.Handled = true;
            }
            else if (e.Key == Key.Delete)
            {
                OnDeleteSelectedClick(sender, e);
                e.Handled = true;
            }
        }

        private void OnPasteClipboardClick(object sender, RoutedEventArgs e)
        {
            if (Clipboard.ContainsImage())
            {
                try
                {
                    var bitmapSource = Clipboard.GetImage();
                    if (bitmapSource != null)
                    {
                        string tempDir = Path.Combine(Path.GetTempPath(), "ImageUpscaler");
                        Directory.CreateDirectory(tempDir);
                        string tempFile = Path.Combine(tempDir, $"clipboard_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                        using (var fileStream = new FileStream(tempFile, FileMode.Create))
                        {
                            var encoder = new PngBitmapEncoder();
                            encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
                            encoder.Save(fileStream);
                        }

                        LoadImageFiles(new[] { tempFile });
                        if (StatusTextBlock != null)
                        {
                            StatusTextBlock.Text = "Image pasted from clipboard.";
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to paste image from clipboard: {ex.Message}", "Clipboard Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                }
            }
            else
            {
                MessageBox.Show("No image found in clipboard.", "Clipboard Empty", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        private void LoadImageFiles(string[] paths)
        {
            // Reset Split View toggle to false on new image import so image displays full
            if (SplitViewToggle != null)
            {
                SplitViewToggle.IsChecked = false;
            }

            foreach (var path in paths)
            {
                if (!File.Exists(path)) continue;

                // Avoid adding duplicate paths already in the list
                if (_images.Any(img => string.Equals(img.FilePath, path, StringComparison.OrdinalIgnoreCase)))
                    continue;

                try
                {
                    var originalImage = ImageUtils.LoadImage(path);
                    var originalBitmap = ImageUtils.LoadBitmapImage(path);
                    var thumbnailBitmap = ImageUtils.CreateThumbnail(path, 180, 120);

                    var item = new UpscaleItem
                    {
                        FilePath = path,
                        FileName = Path.GetFileName(path),
                        OriginalDimensions = $"{originalImage.Width} x {originalImage.Height} px",
                        OriginalImage = originalImage,
                        OriginalBitmap = originalBitmap,
                        ThumbnailBitmap = thumbnailBitmap ?? originalBitmap,
                        Status = "Ready"
                    };

                    _images.Add(item);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error loading image {path}: {ex.Message}");
                }
            }

            UpdateImageCount();

            if (_images.Count > 0 && _selectedItem == null && ImagesListBox != null)
            {
                ImagesListBox.SelectedIndex = 0;
            }

            UpdateWorkspaceState();
        }

        private void UpdateImageCount()
        {
            if (ImageCountTextBlock != null)
            {
                ImageCountTextBlock.Text = $"({_images.Count})";
            }
        }

        #endregion

        #region Left Queue Controls

        private void OnImageSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ImagesListBox == null) return;
            _selectedItem = ImagesListBox.SelectedItem as UpscaleItem;

            // Automatically set split view to true if item is already upscaled, otherwise full view
            if (SplitViewToggle != null)
            {
                SplitViewToggle.IsChecked = _selectedItem?.IsUpscaled == true;
            }

            UpdateWorkspaceState();
            ResetZoomAndPan();
        }

        private void OnMoveUpClick(object sender, RoutedEventArgs e)
        {
            if (ImagesListBox == null) return;
            int index = ImagesListBox.SelectedIndex;
            if (index > 0)
            {
                var item = _images[index];
                _images.RemoveAt(index);
                _images.Insert(index - 1, item);
                ImagesListBox.SelectedIndex = index - 1;
            }
        }

        private void OnMoveDownClick(object sender, RoutedEventArgs e)
        {
            if (ImagesListBox == null) return;
            int index = ImagesListBox.SelectedIndex;
            if (index >= 0 && index < _images.Count - 1)
            {
                var item = _images[index];
                _images.RemoveAt(index);
                _images.Insert(index + 1, item);
                ImagesListBox.SelectedIndex = index + 1;
            }
        }

        private void OnDeleteSelectedClick(object sender, RoutedEventArgs e)
        {
            if (ImagesListBox == null) return;
            int index = ImagesListBox.SelectedIndex;
            if (index >= 0)
            {
                var item = _images[index];
                item.DisposeImages();
                _images.RemoveAt(index);
                UpdateImageCount();

                if (_images.Count > 0)
                {
                    ImagesListBox.SelectedIndex = Math.Min(index, _images.Count - 1);
                }
                else
                {
                    _selectedItem = null;
                    UpdateWorkspaceState();
                }
            }
        }

        private void OnClearAllClick(object sender, RoutedEventArgs e)
        {
            if (_images.Count == 0) return;

            foreach (var item in _images)
            {
                item.DisposeImages();
            }
            _images.Clear();
            _selectedItem = null;
            UpdateImageCount();
            UpdateWorkspaceState();
        }

        #endregion

        #region Workspace & Comparison Display

        private void UpdateWorkspaceState()
        {
            if (EmptyStateBorder == null || ActiveWorkspaceGrid == null || StatusTextBlock == null || DetailsTextBlock == null)
                return;

            if (_selectedItem == null)
            {
                EmptyStateBorder.Visibility = Visibility.Visible;
                ActiveWorkspaceGrid.Visibility = Visibility.Collapsed;
                if (ToolbarSaveButton != null) ToolbarSaveButton.IsEnabled = false;
                if (SaveActionButton != null) SaveActionButton.IsEnabled = false;
                if (UpscaleActionButton != null) UpscaleActionButton.IsEnabled = false;
                if (SingleViewBadge != null) SingleViewBadge.Visibility = Visibility.Collapsed;
                StatusTextBlock.Text = "Ready. Import images or drag and drop to start.";
                DetailsTextBlock.Text = "No image selected";
                return;
            }

            EmptyStateBorder.Visibility = Visibility.Collapsed;
            ActiveWorkspaceGrid.Visibility = Visibility.Visible;
            if (UpscaleActionButton != null) UpscaleActionButton.IsEnabled = true;

            bool isSplitView = SplitViewToggle?.IsChecked == true;
            bool hasUpscaled = _selectedItem.IsUpscaled && _selectedItem.UpscaledBitmap != null;

            if (ToolbarSaveButton != null) ToolbarSaveButton.IsEnabled = hasUpscaled;
            if (SaveActionButton != null) SaveActionButton.IsEnabled = hasUpscaled;

            if (hasUpscaled)
            {
                if (UpscaledImageDisplay != null)
                {
                    UpscaledImageDisplay.Source = _selectedItem.UpscaledBitmap;
                    UpscaledImageDisplay.Visibility = Visibility.Visible;
                }

                if (isSplitView)
                {
                    if (OriginalImageDisplay != null)
                    {
                        OriginalImageDisplay.Source = _selectedItem.OriginalBitmap;
                        OriginalImageDisplay.Visibility = Visibility.Visible;
                    }
                    if (ComparisonOverlayCanvas != null) ComparisonOverlayCanvas.Visibility = Visibility.Visible;
                    if (SingleViewBadge != null) SingleViewBadge.Visibility = Visibility.Collapsed;
                    UpdateSplitCurtain();
                }
                else
                {
                    if (OriginalImageDisplay != null)
                    {
                        OriginalImageDisplay.Visibility = Visibility.Collapsed;
                        OriginalImageDisplay.Clip = null;
                    }
                    if (ComparisonOverlayCanvas != null) ComparisonOverlayCanvas.Visibility = Visibility.Collapsed;
                    if (SingleViewBadge != null)
                    {
                        SingleViewBadge.Visibility = Visibility.Visible;
                        if (SingleViewBadgeText != null)
                        {
                            SingleViewBadgeText.Text = $"Upscaled ({_selectedItem.ModelName} - {_selectedItem.Scale}x)";
                        }
                    }
                }

                StatusTextBlock.Text = $"Viewing upscaled: {_selectedItem.FileName} | {_selectedItem.UpscaledDimensions}";
                DetailsTextBlock.Text = $"Model: {_selectedItem.ModelName} | Scale: {_selectedItem.Scale}x | Resolution: {_selectedItem.UpscaledDimensions}";
            }
            else
            {
                // Only original image is available: always full view
                if (OriginalImageDisplay != null)
                {
                    OriginalImageDisplay.Source = _selectedItem.OriginalBitmap;
                    OriginalImageDisplay.Visibility = Visibility.Visible;
                    OriginalImageDisplay.Clip = null;
                }
                if (UpscaledImageDisplay != null) UpscaledImageDisplay.Visibility = Visibility.Collapsed;
                if (ComparisonOverlayCanvas != null) ComparisonOverlayCanvas.Visibility = Visibility.Collapsed;

                if (SingleViewBadge != null)
                {
                    SingleViewBadge.Visibility = Visibility.Visible;
                    if (SingleViewBadgeText != null)
                    {
                        SingleViewBadgeText.Text = "Original Image - Full Preview";
                    }
                }

                StatusTextBlock.Text = $"Loaded image: {_selectedItem.FileName} ({_selectedItem.OriginalDimensions}). Ready to upscale.";
                DetailsTextBlock.Text = $"Resolution: {_selectedItem.OriginalDimensions}";
            }
        }

        private void OnSplitViewToggled(object sender, RoutedEventArgs e)
        {
            if (EmptyStateBorder == null || ActiveWorkspaceGrid == null) return;
            UpdateWorkspaceState();
        }

        private void UpdateSplitCurtain()
        {
            if (ActiveWorkspaceGrid == null || DividerLine == null || SliderHandle == null || OriginalImageDisplay == null) return;
            if (ActiveWorkspaceGrid.ActualWidth <= 0 || ActiveWorkspaceGrid.ActualHeight <= 0) return;

            double width = ActiveWorkspaceGrid.ActualWidth;
            double height = ActiveWorkspaceGrid.ActualHeight;
            double splitX = width * _splitRatio;

            // 1. Update Divider line and Slider handle in screen canvas coordinates
            DividerLine.X1 = splitX;
            DividerLine.X2 = splitX;
            DividerLine.Y1 = 0;
            DividerLine.Y2 = height;

            Canvas.SetLeft(SliderHandle, splitX - 18);
            Canvas.SetTop(SliderHandle, (height / 2) - 18);

            // 2. Compute exact local clip coordinate on OriginalImageDisplay matching screen divider position
            try
            {
                if (OriginalImageDisplay.IsVisible && VisualTreeHelper.GetParent(OriginalImageDisplay) != null)
                {
                    var transform = ActiveWorkspaceGrid.TransformToVisual(OriginalImageDisplay);
                    System.Windows.Point localSplitPoint = transform.Transform(new System.Windows.Point(splitX, 0));
                    double localSplitX = Math.Max(0, localSplitPoint.X);
                    double clipHeight = Math.Max(OriginalImageDisplay.ActualHeight, 20000);

                    OriginalImageDisplay.Clip = new RectangleGeometry(new Rect(0, 0, localSplitX, clipHeight));
                }
                else
                {
                    OriginalImageDisplay.Clip = null;
                }
            }
            catch
            {
                OriginalImageDisplay.Clip = null;
            }
        }

        private void OnCanvasContainerSizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (SplitViewToggle?.IsChecked == true && _selectedItem?.IsUpscaled == true)
            {
                UpdateSplitCurtain();
            }
        }

        #endregion

        #region Zoom, Pan, and Split Drag Handlers

        private void ApplyZoom(double targetZoom, System.Windows.Point focalPoint)
        {
            double newZoom = Math.Clamp(targetZoom, 0.2, 10.0);
            if (Math.Abs(newZoom - _currentZoom) < 0.0001) return;

            double scaleRatio = newZoom / _currentZoom;
            _currentZoom = newZoom;

            if (CanvasScaleTransform != null)
            {
                CanvasScaleTransform.ScaleX = _currentZoom;
                CanvasScaleTransform.ScaleY = _currentZoom;
            }

            if (CanvasTranslateTransform != null)
            {
                CanvasTranslateTransform.X = focalPoint.X - (focalPoint.X - CanvasTranslateTransform.X) * scaleRatio;
                CanvasTranslateTransform.Y = focalPoint.Y - (focalPoint.Y - CanvasTranslateTransform.Y) * scaleRatio;
            }

            if (ZoomPercentText != null)
            {
                ZoomPercentText.Text = $"{Math.Round(_currentZoom * 100)}%";
            }

            if (SplitViewToggle?.IsChecked == true && _selectedItem?.IsUpscaled == true)
            {
                UpdateSplitCurtain();
            }
        }

        private void OnCanvasMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (ActiveWorkspaceGrid == null) return;

            var mousePos = e.GetPosition(ActiveWorkspaceGrid);
            double zoomFactor = e.Delta > 0 ? 1.18 : 0.85;
            ApplyZoom(_currentZoom * zoomFactor, mousePos);

            e.Handled = true;
        }

        private void OnCanvasMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed && !_isDraggingSlider && ActiveWorkspaceGrid != null)
            {
                _isPanning = true;
                _lastMousePos = e.GetPosition(ActiveWorkspaceGrid);
                ActiveWorkspaceGrid.CaptureMouse();
            }
        }

        private void OnCanvasMouseMove(object sender, MouseEventArgs e)
        {
            if (ActiveWorkspaceGrid == null) return;

            if (_isDraggingSlider)
            {
                var pos = e.GetPosition(ActiveWorkspaceGrid);
                _splitRatio = Math.Clamp(pos.X / ActiveWorkspaceGrid.ActualWidth, 0.02, 0.98);
                UpdateSplitCurtain();
            }
            else if (_isPanning && CanvasTranslateTransform != null)
            {
                var currentPos = e.GetPosition(ActiveWorkspaceGrid);
                var delta = currentPos - _lastMousePos;

                CanvasTranslateTransform.X += delta.X;
                CanvasTranslateTransform.Y += delta.Y;

                _lastMousePos = currentPos;

                if (SplitViewToggle?.IsChecked == true && _selectedItem?.IsUpscaled == true)
                {
                    UpdateSplitCurtain();
                }
            }
        }

        private void OnCanvasMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (_isDraggingSlider && SliderHandle != null)
            {
                _isDraggingSlider = false;
                SliderHandle.ReleaseMouseCapture();
            }

            if (_isPanning && ActiveWorkspaceGrid != null)
            {
                _isPanning = false;
                ActiveWorkspaceGrid.ReleaseMouseCapture();
            }
        }

        private void OnHandleMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed && SliderHandle != null)
            {
                _isDraggingSlider = true;
                SliderHandle.CaptureMouse();
                e.Handled = true;
            }
        }

        private void OnHandleMouseMove(object sender, MouseEventArgs e)
        {
            if (_isDraggingSlider && ActiveWorkspaceGrid != null)
            {
                var pos = e.GetPosition(ActiveWorkspaceGrid);
                _splitRatio = Math.Clamp(pos.X / ActiveWorkspaceGrid.ActualWidth, 0.02, 0.98);
                UpdateSplitCurtain();
                e.Handled = true;
            }
        }

        private void OnHandleMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (_isDraggingSlider && SliderHandle != null)
            {
                _isDraggingSlider = false;
                SliderHandle.ReleaseMouseCapture();
                e.Handled = true;
            }
        }

        private void OnZoomInClick(object sender, RoutedEventArgs e)
        {
            if (ActiveWorkspaceGrid == null) return;
            var centerPoint = new System.Windows.Point(ActiveWorkspaceGrid.ActualWidth / 2.0, ActiveWorkspaceGrid.ActualHeight / 2.0);
            ApplyZoom(_currentZoom * 1.25, centerPoint);
        }

        private void OnZoomOutClick(object sender, RoutedEventArgs e)
        {
            if (ActiveWorkspaceGrid == null) return;
            var centerPoint = new System.Windows.Point(ActiveWorkspaceGrid.ActualWidth / 2.0, ActiveWorkspaceGrid.ActualHeight / 2.0);
            ApplyZoom(_currentZoom / 1.25, centerPoint);
        }

        private void OnResetViewClick(object sender, RoutedEventArgs e)
        {
            ResetZoomAndPan();
        }

        private void ResetZoomAndPan()
        {
            _currentZoom = 1.0;
            if (CanvasScaleTransform != null)
            {
                CanvasScaleTransform.ScaleX = 1.0;
                CanvasScaleTransform.ScaleY = 1.0;
            }
            if (CanvasTranslateTransform != null)
            {
                CanvasTranslateTransform.X = 0;
                CanvasTranslateTransform.Y = 0;
            }
            if (ZoomPercentText != null)
            {
                ZoomPercentText.Text = "100%";
            }
            if (SplitViewToggle?.IsChecked == true && _selectedItem?.IsUpscaled == true)
            {
                UpdateSplitCurtain();
            }
        }

        #endregion

        #region Model Selection & Configuration

        private void OnModelSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ModelSelectionComboBox?.SelectedItem is ModelInfo selectedModel)
            {
                if (ModelDescriptionTextBlock != null)
                {
                    ModelDescriptionTextBlock.Text = selectedModel.Description;
                }

                // Adjust default scale selection
                if (selectedModel.DefaultScale == 2 && Scale2xRadio != null)
                {
                    Scale2xRadio.IsChecked = true;
                }
                else if (selectedModel.DefaultScale == 8 && Scale8xRadio != null)
                {
                    Scale8xRadio.IsChecked = true;
                }
                else if (Scale4xRadio != null)
                {
                    Scale4xRadio.IsChecked = true;
                }
            }
        }

        private string GetSelectedModelId()
        {
            if (ModelSelectionComboBox?.SelectedItem is ModelInfo selectedModel)
            {
                return selectedModel.Id;
            }
            return "realesrgan_x4_photo";
        }

        private string GetSelectedModelDisplayName()
        {
            if (ModelSelectionComboBox?.SelectedItem is ModelInfo selectedModel)
            {
                return selectedModel.Name;
            }
            return "Real-ESRGAN Photo (x4)";
        }

        private int GetSelectedScale()
        {
            if (Scale2xRadio?.IsChecked == true) return 2;
            if (Scale8xRadio?.IsChecked == true) return 8;
            return 4;
        }

        private int GetSelectedTileSize()
        {
            if (TileSizeComboBox?.SelectedItem is ComboBoxItem item && int.TryParse(item.Tag?.ToString(), out int val))
            {
                return val;
            }
            return -1;
        }

        #endregion

        #region Upscale Processing

        private async void OnUpscaleActionClick(object sender, RoutedEventArgs e)
        {
            if (_selectedItem == null || _selectedItem.OriginalImage == null)
            {
                MessageBox.Show("Please select an image to upscale first.", "No Image Selected", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var itemToProcess = _selectedItem;
            string modelId = GetSelectedModelId();
            string modelName = GetSelectedModelDisplayName();
            int scale = GetSelectedScale();
            int tileSize = GetSelectedTileSize();

            // 1. Immediately activate responsive processing modal
            if (UpscaleActionButton != null) UpscaleActionButton.IsEnabled = false;
            if (OverlayTitleText != null) OverlayTitleText.Text = $"Processing: {modelName}";
            if (OverlayStatusText != null) OverlayStatusText.Text = "Initializing super-resolution engine...";
            if (OverlayPercentText != null) OverlayPercentText.Text = "Starting...";
            if (OverlayProgressBar != null)
            {
                OverlayProgressBar.IsIndeterminate = true;
                OverlayProgressBar.Value = 0;
            }
            if (StatusTextBlock != null) StatusTextBlock.Text = $"Initializing {modelName}...";
            if (LoadingOverlay != null) LoadingOverlay.Visibility = Visibility.Visible;

            // Allow the UI thread to render the modal immediately
            await Task.Yield();

            var modelInfo = _modelManager.GetModelById(modelId);
            if (modelInfo == null)
            {
                if (LoadingOverlay != null) LoadingOverlay.Visibility = Visibility.Collapsed;
                if (UpscaleActionButton != null) UpscaleActionButton.IsEnabled = true;
                MessageBox.Show($"Model configuration for '{modelId}' was not found.", "Model Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // Check if Neural model requires Python runtime dependencies
            bool isNeural = modelInfo.Type == UpscalerType.NeuralEsrgan ||
                            modelInfo.Type == UpscalerType.NeuralSwinir ||
                            modelInfo.Type == UpscalerType.NeuralDat ||
                            modelId.Contains("realesrgan") ||
                            modelId.Contains("swinir") ||
                            modelId.Contains("dat");

            if (isNeural)
            {
                var (hasMissing, consentMessage) = PythonBootstrapper.GetMissingDependenciesDescription();
                if (hasMissing)
                {
                    if (LoadingOverlay != null) LoadingOverlay.Visibility = Visibility.Collapsed;
                    var choice = MessageBox.Show(
                        consentMessage,
                        "Runtime Dependencies Setup Consent",
                        MessageBoxButton.YesNo,
                        MessageBoxImage.Question);

                    if (choice != MessageBoxResult.Yes)
                    {
                        if (StatusTextBlock != null) StatusTextBlock.Text = "Runtime dependency installation cancelled by user.";
                        if (UpscaleActionButton != null) UpscaleActionButton.IsEnabled = true;
                        return;
                    }
                    if (LoadingOverlay != null) LoadingOverlay.Visibility = Visibility.Visible;
                }
            }

            // Check & Download model weights if needed
            if (!modelInfo.IsDownloaded && !string.IsNullOrEmpty(modelInfo.Url))
            {
                if (OverlayStatusText != null) OverlayStatusText.Text = $"Downloading model weights ({modelName})...";
                bool downloaded = await _modelManager.DownloadModelWeightAsync(modelInfo.Id, (pct, total, msg) =>
                {
                    Dispatcher.Invoke(() =>
                    {
                        if (OverlayProgressBar != null)
                        {
                            OverlayProgressBar.IsIndeterminate = pct <= 0;
                            if (pct > 0) OverlayProgressBar.Value = pct;
                        }
                        if (OverlayPercentText != null && pct > 0) OverlayPercentText.Text = $"{pct}%";
                        if (OverlayStatusText != null) OverlayStatusText.Text = msg;
                        if (StatusTextBlock != null) StatusTextBlock.Text = msg;
                    });
                });

                if (!downloaded)
                {
                    if (LoadingOverlay != null) LoadingOverlay.Visibility = Visibility.Collapsed;
                    if (UpscaleActionButton != null) UpscaleActionButton.IsEnabled = true;
                    MessageBox.Show($"Could not download model weights for '{modelName}'. Please verify network connection.", "Model Weights Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    if (StatusTextBlock != null) StatusTextBlock.Text = $"Failed to download model weights for {modelName}.";
                    return;
                }
            }

            // Run Upscaling in Background
            try
            {
                Image<Rgba32>? resultImage = null;

                await Task.Run(() =>
                {
                    var upscaler = _modelManager.LoadModel(modelId, scale, tileSize, (pct, total, msg) =>
                    {
                        Dispatcher.Invoke(() =>
                        {
                            if (OverlayProgressBar != null)
                            {
                                OverlayProgressBar.IsIndeterminate = pct <= 0;
                                if (pct > 0) OverlayProgressBar.Value = pct;
                            }
                            if (OverlayPercentText != null && pct > 0) OverlayPercentText.Text = $"{pct}%";
                            if (OverlayStatusText != null) OverlayStatusText.Text = msg;
                            if (StatusTextBlock != null) StatusTextBlock.Text = msg;
                        });
                    });

                    try
                    {
                        resultImage = upscaler.UpscaleImage(itemToProcess.OriginalImage, (pct, total, msg) =>
                        {
                            Dispatcher.Invoke(() =>
                            {
                                if (OverlayProgressBar != null)
                                {
                                    OverlayProgressBar.IsIndeterminate = pct <= 0;
                                    if (pct > 0) OverlayProgressBar.Value = pct;
                                }
                                if (OverlayPercentText != null && pct > 0) OverlayPercentText.Text = $"{pct}%";
                                if (OverlayStatusText != null) OverlayStatusText.Text = msg;
                                if (StatusTextBlock != null) StatusTextBlock.Text = msg;
                            });
                        });
                    }
                    finally
                    {
                        upscaler.UnloadModel();
                    }
                });

                if (resultImage != null)
                {
                    itemToProcess.UpscaledImage?.Dispose();
                    itemToProcess.UpscaledImage = resultImage;
                    itemToProcess.UpscaledBitmap = ImageUtils.ImageSharpToBitmapImage(resultImage);
                    itemToProcess.IsUpscaled = true;
                    itemToProcess.ModelName = modelName;
                    itemToProcess.Scale = scale;
                    itemToProcess.UpscaledDimensions = $"{resultImage.Width} x {resultImage.Height} px";
                    itemToProcess.Status = "Upscaled";

                    // Default to split view comparison after upscaling
                    if (SplitViewToggle != null)
                    {
                        SplitViewToggle.IsChecked = true;
                    }

                    UpdateWorkspaceState();
                    if (StatusTextBlock != null) StatusTextBlock.Text = $"Upscaling complete! Model: {modelName} | Output: {resultImage.Width}x{resultImage.Height}px ({scale}x)";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Upscaling error: {ex.Message}", "Processing Error", MessageBoxButton.OK, MessageBoxImage.Error);
                if (StatusTextBlock != null) StatusTextBlock.Text = "Upscaling failed.";
            }
            finally
            {
                if (LoadingOverlay != null) LoadingOverlay.Visibility = Visibility.Collapsed;
                if (UpscaleActionButton != null) UpscaleActionButton.IsEnabled = true;
            }
        }

        #endregion

        #region Export & Batch

        private void OnSaveCurrentClick(object sender, RoutedEventArgs e)
        {
            if (_selectedItem == null || _selectedItem.UpscaledImage == null) return;

            var dlg = new SaveFileDialog
            {
                Filter = "PNG Image (*.png)|*.png|JPEG Image (*.jpg)|*.jpg|WebP Image (*.webp)|*.webp",
                Title = "Save Upscaled Image",
                FileName = !string.IsNullOrEmpty(_selectedItem.FilePath)
                    ? $"{Path.GetFileNameWithoutExtension(_selectedItem.FilePath)}_upscaled.png"
                    : "upscaled.png"
            };

            if (dlg.ShowDialog() == true)
            {
                try
                {
                    _selectedItem.UpscaledImage.Save(dlg.FileName);
                    _lastExportedFilePath = dlg.FileName;

                    ShowSuccessModal("Image Saved Successfully", $"Saved enhanced image to:\n{dlg.FileName}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Could not save image: {ex.Message}", "Save Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void OnBatchFolderClick(object sender, RoutedEventArgs e)
        {
            string modelId = GetSelectedModelId();
            var model = _modelManager.GetModelById(modelId) ?? _modelManager.GetAvailableModels()[0];
            int scale = GetSelectedScale();
            int tileSize = GetSelectedTileSize();

            var dlg = new BatchDialog(_modelManager, model, scale, tileSize)
            {
                Owner = this
            };

            dlg.ShowDialog();
        }

        private void ShowSuccessModal(string title, string message)
        {
            if (SuccessModalTitle != null) SuccessModalTitle.Text = title;
            if (SuccessModalMessage != null) SuccessModalMessage.Text = message;
            if (SuccessModalDialog != null) SuccessModalDialog.Visibility = Visibility.Visible;
        }

        private void OnCloseSuccessModalClick(object sender, RoutedEventArgs e)
        {
            if (SuccessModalDialog != null) SuccessModalDialog.Visibility = Visibility.Collapsed;
        }

        private void OnOpenExportedFolderClick(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(_lastExportedFilePath) && File.Exists(_lastExportedFilePath))
            {
                string? folder = Path.GetDirectoryName(_lastExportedFilePath);
                if (!string.IsNullOrEmpty(folder) && Directory.Exists(folder))
                {
                    Process.Start(new ProcessStartInfo("explorer.exe", $"/select,\"{_lastExportedFilePath}\"") { UseShellExecute = true });
                }
            }
        }

        private void OnOpenExportedFileClick(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(_lastExportedFilePath) && File.Exists(_lastExportedFilePath))
            {
                Process.Start(new ProcessStartInfo(_lastExportedFilePath) { UseShellExecute = true });
            }
        }

        #endregion

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            base.OnClosing(e);

            try
            {
                foreach (var item in _images)
                {
                    item.DisposeImages();
                }
                _modelManager.UnloadAllModels();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error during MainWindow closing cleanup: {ex.Message}");
            }
        }
    }
}