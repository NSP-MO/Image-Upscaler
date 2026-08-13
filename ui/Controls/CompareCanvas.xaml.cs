using System;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImageUpscaler.UI.Controls
{
    public partial class CompareCanvas : UserControl
    {
        public event Action<string>? FileSelected;
        public event Action? BatchRequested;

        private readonly RectangleGeometry _clipGeometry = new RectangleGeometry();
        private bool _isSliderDragging = false;
        private bool _isPanning = false;
        private Point _panStartPoint;
        private double _startTranslateX;
        private double _startTranslateY;

        private double _splitPosition = 0.5; // 0.0 to 1.0
        private double _currentZoom = 1.0;

        public CompareCanvas()
        {
            InitializeComponent();
        }

        public void SetSingleImage(BitmapImage original)
        {
            FolderViewGrid.Visibility = Visibility.Collapsed;
            ImageTransformGroup.Visibility = Visibility.Visible;
            ZoomToolbar.Visibility = Visibility.Visible;

            OriginalImageControl.Clip = null; // 100% visible without any clipping mask!
            OriginalImageControl.Source = original;
            OriginalImageControl.Visibility = Visibility.Visible;

            UpscaledImageControl.Source = null;
            OverlayCanvas.Visibility = Visibility.Collapsed;

            ResetZoomAndPan();
        }

        public void SetComparisonImages(BitmapImage original, BitmapImage upscaled)
        {
            FolderViewGrid.Visibility = Visibility.Collapsed;
            ImageTransformGroup.Visibility = Visibility.Visible;
            ZoomToolbar.Visibility = Visibility.Visible;

            OriginalImageControl.Clip = _clipGeometry; // Apply split geometry clipping
            OriginalImageControl.Source = original;
            OriginalImageControl.Visibility = Visibility.Visible;

            UpscaledImageControl.Source = upscaled;
            OverlayCanvas.Visibility = Visibility.Visible;

            ResetZoomAndPan();
            UpdateLayoutAndClip();
        }

        public void SetFolderView(string folderPath, int imageCount)
        {
            ImageTransformGroup.Visibility = Visibility.Collapsed;
            OverlayCanvas.Visibility = Visibility.Collapsed;
            ZoomToolbar.Visibility = Visibility.Collapsed;

            FolderNameTextBlock.Text = Path.GetFileName(folderPath);
            if (string.IsNullOrEmpty(FolderNameTextBlock.Text))
            {
                FolderNameTextBlock.Text = folderPath;
            }
            FolderStatsTextBlock.Text = $"Folder contains {imageCount} supported image files.";

            FolderViewGrid.Visibility = Visibility.Visible;
        }

        private void OnFolderBatchButtonClick(object sender, RoutedEventArgs e)
        {
            BatchRequested?.Invoke();
        }

        private void OnSizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (OverlayCanvas.Visibility == Visibility.Visible)
            {
                UpdateLayoutAndClip();
            }
        }

        private void OnDragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Copy;
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }
            e.Handled = true;
        }

        private void OnDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (files.Length > 0)
                {
                    FileSelected?.Invoke(files[0]);
                }
            }
        }

        #region Split Slider Dragging

        private void OnHandleMouseDown(object sender, MouseButtonEventArgs e)
        {
            _isSliderDragging = true;
            SliderHandle.CaptureMouse();
            e.Handled = true;
        }

        private void OnHandleMouseUp(object sender, MouseButtonEventArgs e)
        {
            _isSliderDragging = false;
            SliderHandle.ReleaseMouseCapture();
            e.Handled = true;
        }

        private void OnHandleMouseMove(object sender, MouseEventArgs e)
        {
            if (_isSliderDragging)
            {
                Point pos = e.GetPosition(ContainerGrid);
                double w = ContainerGrid.ActualWidth;
                if (w > 0)
                {
                    _splitPosition = System.Math.Clamp(pos.X / w, 0.0, 1.0);
                    UpdateLayoutAndClip();
                }
                e.Handled = true;
            }
        }

        private void UpdateLayoutAndClip()
        {
            double w = ContainerGrid.ActualWidth;
            double h = ContainerGrid.ActualHeight;

            if (w <= 0 || h <= 0) return;

            double splitX = w * _splitPosition;

            // Move divider line and handle in container screen space
            DividerLine.X1 = splitX;
            DividerLine.X2 = splitX;

            Canvas.SetLeft(SliderHandle, splitX - (SliderHandle.Width / 2));
            Canvas.SetTop(SliderHandle, (h / 2) - (SliderHandle.Height / 2));

            // Align image clipping rectangle precisely with screen divider line
            if (OverlayCanvas.Visibility == Visibility.Visible && OriginalImageControl.IsVisible)
            {
                try
                {
                    GeneralTransform transform = ContainerGrid.TransformToVisual(OriginalImageControl);
                    Rect screenSplitRect = new Rect(-10000, -10000, splitX + 10000, h + 20000);
                    Rect localSplitRect = transform.TransformBounds(screenSplitRect);
                    _clipGeometry.Rect = localSplitRect;
                }
                catch
                {
                    _clipGeometry.Rect = new Rect(0, 0, splitX, h);
                }
            }
        }

        #endregion

        #region Zoom and Pan Logic

        private void OnCanvasMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (ImageTransformGroup.Visibility != Visibility.Visible) return;
            double zoomFactor = e.Delta > 0 ? 1.15 : 0.85;
            Point mousePos = e.GetPosition(ContainerGrid);
            ApplyZoom(zoomFactor, mousePos);
            e.Handled = true;
        }

        private void OnCanvasMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (ImageTransformGroup.Visibility != Visibility.Visible) return;
            if (e.ChangedButton == MouseButton.Middle || e.ChangedButton == MouseButton.Right || 
               (e.ChangedButton == MouseButton.Left && !_isSliderDragging))
            {
                _isPanning = true;
                _panStartPoint = e.GetPosition(ContainerGrid);
                _startTranslateX = CanvasTranslateTransform.X;
                _startTranslateY = CanvasTranslateTransform.Y;
                ContainerGrid.CaptureMouse();
            }
        }

        private void OnCanvasMouseMove(object sender, MouseEventArgs e)
        {
            if (_isSliderDragging)
            {
                OnHandleMouseMove(sender, e);
                return;
            }

            if (_isPanning)
            {
                Point currentPos = e.GetPosition(ContainerGrid);
                Vector delta = currentPos - _panStartPoint;
                CanvasTranslateTransform.X = _startTranslateX + delta.X;
                CanvasTranslateTransform.Y = _startTranslateY + delta.Y;
                UpdateLayoutAndClip();
            }
        }

        private void OnCanvasMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (_isPanning)
            {
                _isPanning = false;
                ContainerGrid.ReleaseMouseCapture();
            }
        }

        private void ApplyZoom(double factor, Point centerPoint)
        {
            double oldZoom = _currentZoom;
            double newZoom = System.Math.Clamp(oldZoom * factor, 0.5, 10.0);
            if (Math.Abs(newZoom - oldZoom) < 0.001) return;

            double scaleRatio = newZoom / oldZoom;

            // Keep the focal point (viewport center or mouse cursor position) stationary during zoom
            CanvasTranslateTransform.X = centerPoint.X - scaleRatio * (centerPoint.X - CanvasTranslateTransform.X);
            CanvasTranslateTransform.Y = centerPoint.Y - scaleRatio * (centerPoint.Y - CanvasTranslateTransform.Y);

            _currentZoom = newZoom;
            CanvasScaleTransform.ScaleX = _currentZoom;
            CanvasScaleTransform.ScaleY = _currentZoom;

            ZoomPercentText.Text = $"{(int)Math.Round(_currentZoom * 100)}%";
            UpdateLayoutAndClip();
        }

        private void ResetZoomAndPan()
        {
            _currentZoom = 1.0;
            CanvasScaleTransform.ScaleX = 1.0;
            CanvasScaleTransform.ScaleY = 1.0;
            CanvasTranslateTransform.X = 0;
            CanvasTranslateTransform.Y = 0;
            ZoomPercentText.Text = "100%";
            UpdateLayoutAndClip();
        }

        private void OnZoomInClick(object sender, RoutedEventArgs e)
        {
            Point center = new Point(ContainerGrid.ActualWidth / 2, ContainerGrid.ActualHeight / 2);
            ApplyZoom(1.25, center);
        }

        private void OnZoomOutClick(object sender, RoutedEventArgs e)
        {
            Point center = new Point(ContainerGrid.ActualWidth / 2, ContainerGrid.ActualHeight / 2);
            ApplyZoom(0.8, center);
        }

        private void OnZoomResetClick(object sender, RoutedEventArgs e)
        {
            ResetZoomAndPan();
        }

        #endregion
    }
}
