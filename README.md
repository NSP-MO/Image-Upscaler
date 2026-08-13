# Image Upscaler

Image Upscaler is a desktop application for image super-resolution, scaling, and photo enhancement. Built with WPF (.NET 10.0) and a PyTorch / ONNX inference backend, Image Upscaler supports pretrained super-resolution models and classical image interpolation algorithms.

---

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
  - [Directory Structure](#directory-structure)
  - [Processing Pipeline Lifecycle](#processing-pipeline-lifecycle)
  - [Data Flow Diagram](#data-flow-diagram)
  - [Model Registry Reference](#model-registry-reference)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Configure Python Backend Environment](#2-configure-python-backend-environment)
  - [3. GPU Acceleration Setup (Optional)](#3-gpu-acceleration-setup-optional)
  - [4. Build and Run WPF Application](#4-build-and-run-wpf-application)
- [Usage Guide](#usage-guide)
  - [Single Image Processing and Split Comparison](#single-image-processing-and-split-comparison)
  - [Batch Directory Upscaling](#batch-directory-upscaling)
  - [Model Weight Auto-Download](#model-weight-auto-download)
- [Available Commands and Scripts](#available-commands-and-scripts)
- [Distribution and Packaging](#distribution-and-packaging)
- [Troubleshooting](#troubleshooting)
- [License & Model Attributions](#license--model-attributions)

---

## Key Features

- **Neural Super-Resolution**: Integration of pretrained neural network architectures including Real-ESRGAN, SwinIR, BSRGAN, Remacri, and DAT (Vision Transformer ICCV 2023).
- **Hybrid Inference Engine**: Execution backends supporting PyTorch (GPU CUDA / CPU), ONNX Runtime, and C# algorithmic fallbacks.
- **Interactive Split-Screen Visual Inspection**: Custom WPF compare canvas featuring slider controls to evaluate original versus upscaled imagery side by side.
- **Batch Processing Dialog**: Multi-file folder batch processing with live progress reporting and output directory customization.
- **VRAM Adaptive Tiling Engine**: Spatial tiling with automatic VRAM sensing to help reduce Out-Of-Memory (OOM) errors during high-resolution processing.
- **Automated Pretrained Weights Downloader**: Background downloader that fetches missing `.pth` weights from HuggingFace and GitHub Releases.
- **Fast Algorithmic Baselines**: Built-in edge-preserving and pattern-based baseline upscalers including Fast Lanczos4, Fast NEDI (Edge-Directed), Guided Edge Filter, Google RAISR Patch Regression, xBRZ Pattern Engine, and Vector Contour Tracing.

---

## Tech Stack

### Frontend & Application Infrastructure
- **Framework**: .NET 10.0 WPF (Windows Presentation Foundation)
- **Language**: C# 13 / .NET 10
- **Image Processing**: SixLabors.ImageSharp (v2.1.9)
- **ONNX Acceleration**: Microsoft.ML.OnnxRuntime (v1.28.0)

### Neural Backend & Bridge
- **Runtime Environment**: Python 3.10+
- **Deep Learning Framework**: PyTorch (v2.0+) & Torchvision (v0.15+)
- **Computer Vision**: OpenCV (opencv-python v4.7+), Pillow (v9.5+), NumPy (v1.24+)
- **GUI & Automation**: PySide6 (v6.5+)

### Hardware Acceleration
- **GPU Backends**: NVIDIA CUDA (CUDA 12.x / 13.x, including Blackwell architecture)
- **CPU Fallback**: Multi-threaded CPU execution via PyTorch and SixLabors.ImageSharp

---

## System Architecture

### Directory Structure

```
image-upscaler/
├── image-upscaler.csproj       # .NET 10.0 WPF Project File
├── App.xaml                    # WPF Application Definition
├── App.xaml.cs                 # Application Entry Point & Exception Handlers
├── MainWindow.xaml             # Main Window Layout Definition
├── MainWindow.xaml.cs          # Main Controller & Event Orchestrator
├── AssemblyInfo.cs             # Theme & Assembly Attributes
├── requirements.txt            # Python Dependencies Specification
├── Services/                   # Core Application Services
│   ├── ModelManager.cs         # Model Registry, Weight Downloader & Factory
│   ├── BatchProcessor.cs       # Multi-file Batch Processing Service
│   └── ImageUtils.cs           # SixLabors ImageSharp & Bitmap Converters
├── Models/                     # Upscaler Implementation Classes
│   ├── ModelInfo.cs            # Data Model for Model Metadata & Downloads
│   ├── BaseUpscaler.cs         # Abstract Upscaler & Tiling Base Engine
│   ├── PytorchUpscaler.cs      # Subprocess Bridge to PyTorch Python Engine
│   ├── OnnxUpscaler.cs         # C# Native ONNX Runtime Upscaler
│   └── FastUpscalers.cs        # Classical Algorithmic Filters (NEDI, RAISR, xBRZ)
├── UI/                         # User Interface Controls & Styles
│   ├── Controls/
│   │   ├── DropZone.xaml       # Drag-and-Drop Drag Handle Area
│   │   ├── DropZone.xaml.cs    # Drag-and-Drop Logic
│   │   ├── CompareCanvas.xaml  # Interactive Split-Screen Comparison View
│   │   ├── CompareCanvas.xaml.cs # Split Slider Math & Render Context
│   │   ├── Sidebar.xaml        # Control Panel (Model, Scale, Actions)
│   │   └── Sidebar.xaml.cs     # Sidebar Event Handlers
│   ├── Dialogs/
│   │   ├── BatchDialog.xaml    # Multi-file Batch Progress Window
│   │   └── BatchDialog.xaml.cs # Batch Execution & Cancel Logic
│   └── Styles/
│       └── Theme.xaml          # Modern Dark Glassmorphism Styling Resources
├── models/                     # Python Neural Network Architecture Modules
│   ├── __init__.py             # Python Package Initializer
│   ├── base_upscaler.py        # Python Abstract Model & Tile Processing Class
│   ├── model_manager.py        # Python Model Registry & VRAM Adaptive Sensor
│   ├── real_esrgan.py          # Real-ESRGAN Model Architecture Definition
│   ├── swin_ir.py              # SwinIR Vision Transformer Architecture
│   ├── dat.py                  # DAT (Dual Aggregation Transformer) Model
│   └── fast_upscaler.py        # Python Fallback & Algorithmic Filters
└── weights/                    # Main Weights Directory (.pth Model Files)
```

### Processing Pipeline Lifecycle

```
[User Action] Selects File/Folder & Model in WPF Sidebar
      │
      ▼
[MainWindow.xaml.cs] Invokes ModelManager to load designated upscaler
      │
      ├───────────────────────┬───────────────────────┐
      ▼                       ▼                       ▼
(Neural Models)         (ONNX Models)           (Classical Filters)
PytorchUpscaler.cs      OnnxUpscaler.cs         FastUpscalers.cs
      │                       │                       │
      ▼                       ▼                       ▼
Spawns Subprocess       Executes ONNX           Executes Native C#
Python Bridge Engine    Inference Session       Spatial Filters
(PyTorch Models)        via DirectML / CPU      via ImageSharp
      │                       │                       │
      ├───────────────────────┴───────────────────────┘
      ▼
Reads Stdout Logs in Real-time & Emits [PROGRESS] Notifications
      │
      ▼
[CompareCanvas Control] Renders Side-by-Side Split View
```

### Data Flow Diagram

```
Input Image ──► Temporary Disk Buffer ──► PyTorch / ONNX Engine
                                                 │
                                                 ▼
Output Bitmap ◄── SixLabors ImageSharp ◄── Scaled Neural Tensor
      │
      ▼
Compare Canvas (Split Slider UI) / Saved Disk Output
```

### Model Registry Reference

| Model ID | Display Name | Type | Scale | Weight Filename | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `realesrgan_x4_photo` | Real-ESRGAN Photo | Neural ESRGAN | 4x | `RealESRGAN_x4plus.pth` | Super-resolution optimized for real-world photo textures. |
| `remacri_x4` | Remacri Details | Neural ESRGAN | 4x | `remacri_x4.pth` | Optimized for facial textures, skin tones, and fabric detail. |
| `bsrgan_x4` | BSRGAN Restorer | Neural ESRGAN | 4x | `bsrgan_x4.pth` | Restoration of heavily degraded, noisy, or compressed images. |
| `dat_x4` | DAT Transformer | Neural DAT | 4x | `dat_x4.pth` | ICCV 2023 Vision Transformer with dual spatial/channel attention. |
| `realesrgan_x4_anime` | Real-ESRGAN Anime | Neural ESRGAN | 4x | `RealESRGAN_x4plus_anime_6B.pth` | Specialized for anime, digital art, and vector graphics. |
| `realesrgan_x2_general` | Real-ESRGAN Fast | Neural ESRGAN | 2x | `RealESRGAN_x2plus.pth` | Balanced 2x upscale pass for quick enhancements. |
| `swinir_x4_classical` | SwinIR Classical | Neural SwinIR | 4x | `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth` | Swin Transformer trained on DIV2K for sharp photographic detail. |
| `swinir_x4_real` | Real-SwinIR Photo | Neural SwinIR | 4x | `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` | Swin Transformer GAN variant for artifact removal. |
| `fast_lanczos` | Fast Lanczos4 | Algorithmic | 4x | N/A | Classical 4th-order Lanczos mathematical resampling. |
| `fast_nedi` | Fast NEDI | Algorithmic | 4x | N/A | Edge-directed covariance interpolation for crisp diagonal lines. |
| `guided_edge` | Guided Edge Filter | Algorithmic | 4x | N/A | Edge-preserving filter preventing haloing artifacts. |
| `raisr_patch` | RAISR Patch Regression | Algorithmic | 4x | N/A | Google RAISR patch regression algorithm. |
| `xbrz_pattern` | xBRZ Pattern Engine | Algorithmic | 4x | N/A | Rule-based anti-aliasing pattern scale engine for pixel art. |
| `vector_contour` | Vector Contour Engine | Algorithmic | 4x | N/A | Vectorization and polygon curve tracing for logo upscaling. |

---

## Prerequisites

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **SDK & Runtime**: .NET 10.0 SDK or .NET 10.0 Desktop Runtime
- **Python Environment**: Python 3.10 or newer (installed globally or accessible via PATH)
- **Graphics Hardware**: NVIDIA GPU with CUDA support (Recommended: NVIDIA GeForce RTX Series). DirectML or CPU execution is supported as fallback.

---

## Getting Started

### 1. Clone the Repository

```powershell
git clone https://github.com/your-org/image-upscaler.git
cd image-upscaler
```

### 2. Configure Python Backend Environment

Install the required Python packages into your environment:

```powershell
pip install -r requirements.txt
```

Verify installed dependencies:

```powershell
python -c "import torch, torchvision, PIL, cv2; print('Python backend dependencies satisfied. PyTorch Version:', torch.__version__)"
```

### 3. GPU Acceleration Setup (Optional)

For modern NVIDIA architectures (such as NVIDIA GeForce RTX 50 Series / Blackwell architecture), install the appropriate CUDA preview/nightly PyTorch package:

```powershell
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
```

Verify CUDA GPU availability:

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. Build and Run WPF Application

Build the solution using the .NET CLI:

```powershell
dotnet build image-upscaler.csproj -c Debug
```

Launch the application:

```powershell
dotnet run --project image-upscaler.csproj
```

---

## Usage Guide

### Single Image Processing and Split Comparison

1. Launch Image Upscaler.
2. Drag and drop an image (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) onto the central `DropZone` or click **Browse File**.
3. Select your target **Model** and **Scale Factor** in the sidebar.
4. Click **Upscale Image**. The PyTorch backend will execute the neural pass.
5. Drag the interactive split slider left and right on the compare canvas to inspect fine image details between the original and upscaled versions.
6. Click **Save Output Image** to export the result.

### Batch Directory Upscaling

1. Drag a directory containing multiple images onto the application or select **Batch Processing** from the sidebar.
2. Configure the input folder, target output directory, and model selection in the **Batch Upscale Dialog**.
3. Click **Start Batch**. The application will process the queue sequentially, updating progress bars and counters in real time.

### Model Weight Auto-Download

If a neural model is selected whose `.pth` weight file is not found in the `weights/` directory, the application will automatically trigger an asynchronous HTTP download stream, saving the weights into the `weights/` folder before launching inference.

---

## Available Commands and Scripts

### .NET Application Commands

| Command | Purpose |
| :--- | :--- |
| `dotnet build image-upscaler.csproj` | Compile the WPF application project. |
| `dotnet run --project image-upscaler.csproj` | Run the desktop application locally. |
| `dotnet clean image-upscaler.csproj` | Remove compiled build artifacts in `bin/` and `obj/`. |

### Standalone Python Inference CLI

You can execute the neural inference engine directly via Python without starting the WPF GUI:

```powershell
python -m models.model_manager --model_id realesrgan_x4_photo --input input.png --output output.png --scale 4 --tile_size 512
```

Command-line parameters:
- `--model_id`: Model identifier (e.g. `realesrgan_x4_photo`, `dat_x4`, `swinir_x4_classical`, `remacri_x4`).
- `--input`: Path to input image file.
- `--output`: Destination path for saved output image.
- `--scale`: Upscaling factor integer multiplier (default: `4`).
- `--tile_size`: Spatial tile size in pixels (default: `-1` for automatic VRAM adaptive resolution).

---

## Distribution and Packaging

To build a standalone, self-contained Windows x64 executable package that does not require an installed .NET runtime:

```powershell
dotnet publish image-upscaler.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=false
```

Published binaries will be generated in `bin/Release/net10.0-windows/win-x64/publish/`. Ensure the `models/` and `weights/` directories accompany the executable in deployment builds.

---

## Troubleshooting

### CUDA Out-Of-Memory (OOM) Errors
- **Symptom**: Processing high-resolution images causes GPU memory allocation errors.
- **Solution**: Set tile size to automatic (`-1`) or manually restrict tile size to `256` or `384` pixels in configuration. The Python bridge automatically measures available VRAM and scales tile dimensions dynamically.

### Python Executable Not Found
- **Symptom**: `PytorchUpscaler` fails with executable resolution error.
- **Solution**: Ensure Python is added to system `PATH` or installed in standard system paths (`C:\Python\Python313` or `%LOCALAPPDATA%\Programs\Python`).

### Missing Weight Files or Download Interruption
- **Symptom**: Network error during initial model loading.
- **Solution**: Ensure internet access is enabled for HuggingFace / GitHub Release endpoints, or manually place `.pth` weight files directly into the `weights/` directory.

---

## License & Model Attributions

### Application License
The core **Image Upscaler** application codebase is licensed under the **MIT License**.

### Pretrained Model Licenses & Attributions

Each pretrained neural model and algorithmic filter integrated into Image Upscaler is subject to its original author's license terms and research attributions:

| Model ID | Model Name | Primary License | Authors / Repository |
| :--- | :--- | :--- | :--- |
| `realesrgan_x4_photo`<br>`realesrgan_x4_anime`<br>`realesrgan_x2_general` | Real-ESRGAN | **BSD 3-Clause License** | Xintao Wang et al. ([Real-ESRGAN Repository](https://github.com/xinntao/Real-ESRGAN)) |
| `remacri_x4` | Remacri Details | **CC BY-SA 4.0 / Open Community** | FacehugmanIII ([HuggingFace Repository](https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri)) |
| `bsrgan_x4` | BSRGAN Restorer | **Apache 2.0 License** | Kai Zhang et al. ([KAIR Repository](https://github.com/cszn/KAIR)) |
| `dat_x4` | DAT Transformer | **Apache 2.0 License** | Zheng Chen et al. (ICCV 2023) ([DAT Repository](https://github.com/zhengchen1999/DAT)) |
| `swinir_x4_classical`<br>`swinir_x4_real` | SwinIR & Real-SwinIR | **Apache 2.0 License** | Jingyun Liang et al. ([SwinIR Repository](https://github.com/JingyunLiang/SwinIR)) |
| `fast_lanczos`<br>`fast_nedi`<br>`guided_edge`<br>`raisr_patch`<br>`xbrz_pattern`<br>`vector_contour` | Classical & Edge Baselines | **MIT License** | Mathematical & Open-Source Algorithmic Baselines |

> [!NOTE]
> Pretrained model weights (`.pth`) downloaded automatically by the application remain the intellectual property of their respective creators and researchers. Please verify individual commercial usage terms for downstream commercial deployments.
