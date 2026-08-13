using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Win32;
using ImageUpscaler.Models;
using ImageUpscaler.Services;

namespace ImageUpscaler.UI.Dialogs
{
    public partial class BatchDialog : Window
    {
        private readonly BatchProcessor _batchProcessor;
        private readonly ModelInfo _model;
        private readonly int _scale;
        private readonly int _tileSize;

        public BatchDialog(ModelManager modelManager, ModelInfo model, int scale, int tileSize)
        {
            InitializeComponent();
            _batchProcessor = new BatchProcessor(modelManager);
            _model = model;
            _scale = scale;
            _tileSize = tileSize;

            ModelInfoText.Text = $"Selected Model: {model.Name} ({scale}x)";
        }

        private void OnBrowseInputClick(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFolderDialog
            {
                Title = "Select Input Folder"
            };

            if (dlg.ShowDialog() == true)
            {
                InputFolderBox.Text = dlg.FolderName;
                if (string.IsNullOrEmpty(OutputFolderBox.Text))
                {
                    OutputFolderBox.Text = Path.Combine(dlg.FolderName, "Upscaled_Output");
                }
            }
        }

        private void OnBrowseOutputClick(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFolderDialog
            {
                Title = "Select Output Folder"
            };

            if (dlg.ShowDialog() == true)
            {
                OutputFolderBox.Text = dlg.FolderName;
            }
        }

        private async void OnStartClick(object sender, RoutedEventArgs e)
        {
            string inFolder = InputFolderBox.Text;
            string outFolder = OutputFolderBox.Text;

            if (string.IsNullOrEmpty(inFolder) || !Directory.Exists(inFolder))
            {
                MessageBox.Show("Please select a valid input folder.", "Invalid Folder", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            StartButton.IsEnabled = false;

            await _batchProcessor.ProcessFolderAsync(
                inFolder,
                outFolder,
                _model.Id,
                _scale,
                _tileSize,
                (pct, total, msg) =>
                {
                    Dispatcher.Invoke(() =>
                    {
                        BatchProgressBar.Value = pct;
                        StatusMessageText.Text = msg;
                    });
                });

            StartButton.IsEnabled = true;
        }

        private void OnCloseClick(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
