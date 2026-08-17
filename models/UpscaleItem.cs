using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Media.Imaging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ImageUpscaler.Models
{
    public class UpscaleItem : INotifyPropertyChanged
    {
        private string _filePath = string.Empty;
        private string _fileName = string.Empty;
        private string _originalDimensions = string.Empty;
        private string _upscaledDimensions = string.Empty;
        private Image<Rgba32>? _originalImage;
        private Image<Rgba32>? _upscaledImage;
        private BitmapImage? _originalBitmap;
        private BitmapImage? _upscaledBitmap;
        private BitmapImage? _thumbnailBitmap;
        private bool _isUpscaled;
        private string _modelName = string.Empty;
        private int _scale = 4;
        private string _status = "Ready";

        public string FilePath
        {
            get => _filePath;
            set
            {
                _filePath = value;
                FileName = Path.GetFileName(value);
                OnPropertyChanged();
            }
        }

        public string FileName
        {
            get => _fileName;
            set { _fileName = value; OnPropertyChanged(); }
        }

        public string OriginalDimensions
        {
            get => _originalDimensions;
            set { _originalDimensions = value; OnPropertyChanged(); }
        }

        public string UpscaledDimensions
        {
            get => _upscaledDimensions;
            set { _upscaledDimensions = value; OnPropertyChanged(); }
        }

        public Image<Rgba32>? OriginalImage
        {
            get => _originalImage;
            set { _originalImage = value; OnPropertyChanged(); }
        }

        public Image<Rgba32>? UpscaledImage
        {
            get => _upscaledImage;
            set { _upscaledImage = value; OnPropertyChanged(); }
        }

        public BitmapImage? OriginalBitmap
        {
            get => _originalBitmap;
            set { _originalBitmap = value; OnPropertyChanged(); }
        }

        public BitmapImage? UpscaledBitmap
        {
            get => _upscaledBitmap;
            set { _upscaledBitmap = value; OnPropertyChanged(); }
        }

        public BitmapImage? ThumbnailBitmap
        {
            get => _thumbnailBitmap;
            set { _thumbnailBitmap = value; OnPropertyChanged(); }
        }

        public bool IsUpscaled
        {
            get => _isUpscaled;
            set { _isUpscaled = value; OnPropertyChanged(); }
        }

        public string ModelName
        {
            get => _modelName;
            set { _modelName = value; OnPropertyChanged(); }
        }

        public int Scale
        {
            get => _scale;
            set { _scale = value; OnPropertyChanged(); }
        }

        public string Status
        {
            get => _status;
            set { _status = value; OnPropertyChanged(); }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public void DisposeImages()
        {
            _originalImage?.Dispose();
            _originalImage = null;
            _upscaledImage?.Dispose();
            _upscaledImage = null;
        }
    }
}
