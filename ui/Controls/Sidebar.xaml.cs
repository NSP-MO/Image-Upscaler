using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using ImageUpscaler.Models;

namespace ImageUpscaler.UI.Controls
{
    public partial class Sidebar : UserControl
    {
        public event Action? SelectFileRequested;
        public event Action? UpscaleRequested;
        public event Action? SaveRequested;
        public event Action? BatchRequested;
        public event Action<ModelInfo>? ModelChanged;

        public Sidebar()
        {
            InitializeComponent();
        }

        public void PopulateModels(List<ModelInfo> models)
        {
            ModelComboBox.ItemsSource = models;
            if (models.Count > 0)
            {
                ModelComboBox.SelectedIndex = 0;
            }
        }

        public ModelInfo? SelectedModel => ModelComboBox.SelectedItem as ModelInfo;

        public int SelectedScale
        {
            get
            {
                if (Scale2xRadio.IsChecked == true) return 2;
                if (Scale8xRadio.IsChecked == true) return 8;
                return 4;
            }
        }

        public int SelectedTileSize
        {
            get
            {
                if (TileSizeComboBox.SelectedItem is ComboBoxItem item && int.TryParse(item.Tag?.ToString(), out int val))
                {
                    return val;
                }
                return -1; // Default: Auto VRAM Adaptive
            }
        }

        public void SetSaveButtonEnabled(bool enabled)
        {
            SaveButton.IsEnabled = enabled;
        }

        public void SetUpscaleButtonEnabled(bool enabled)
        {
            UpscaleButton.IsEnabled = enabled;
        }

        private void OnSelectFileButtonClick(object sender, RoutedEventArgs e)
        {
            SelectFileRequested?.Invoke();
        }

        private void OnModelSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (SelectedModel != null)
            {
                ModelDescText.Text = SelectedModel.Description;
                ModelChanged?.Invoke(SelectedModel);
            }
        }

        private void OnScaleChecked(object sender, RoutedEventArgs e)
        {
            // Scale changed
        }

        private void OnUpscaleButtonClick(object sender, RoutedEventArgs e)
        {
            UpscaleRequested?.Invoke();
        }

        private void OnSaveButtonClick(object sender, RoutedEventArgs e)
        {
            SaveRequested?.Invoke();
        }

        private void OnBatchButtonClick(object sender, RoutedEventArgs e)
        {
            BatchRequested?.Invoke();
        }
    }
}
