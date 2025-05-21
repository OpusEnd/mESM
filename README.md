# UNetSLSM: Cell Segmentation Using Attention U-Net

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

This project implements an Attention U-Net model for cell segmentation in microscopy images. The model combines attention gates with a U-Net architecture to improve segmentation accuracy.

### Project Structure

```
UNetSLSM/
│
├── src/                  # Source code
│   ├── __init__.py       # Package initialization
│   ├── model.py          # Attention U-Net model definition
│   ├── data.py           # Data loading and processing
│   ├── inference.py      # Inference functions
│   └── app.py            # Main application
│
├── config/               # Configuration files
│   ├── train_config.yaml     # Training configuration
│   ├── inference_config.yaml # Inference configuration
│   └── test_config.yaml      # Test configuration
│
├── data/                 # Data directory
│   └── cell_dataset/     # Cell images
│       ├── img_dir/      # Image directories
│       │   ├── train/    # Training images
│       │   └── val/      # Validation images
│       └── mask_dir/     # Mask directories
│           ├── train/    # Training masks
│           └── val/      # Validation masks
│
├── outputs/              # Output directory
│   └── test_results/     # Test results
│
├── runs/                 # Training run directories (models)
│
├── main.py               # Main script for inference
├── train.py              # Script for training
├── run.py                # Entry point script
└── requirements.txt      # Package dependencies
```

### Requirements

See `requirements.txt` for a complete list of dependencies. Main requirements include:

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- PyDenseCRF
- Albumentations
- scikit-learn
- PIL
- PyYAML

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your test images in the `data/cell_dataset/img_dir/train/` directory.
4. Ensure your model file is available in the path specified in the config file.

### Usage

The project provides two running modes: structure testing and model inference.

#### Structure Testing

Structure testing validates that the project structure and modules load correctly without needing actual model weights.

```bash
python run.py --test-only
```

#### Model Inference

Model inference uses a trained model to segment test images.

```bash
python run.py
```

For CPU-only execution (if you have GPU-related issues):

```bash
python run.py --force-cpu
```

### Configuration

Edit `config/test_config.yaml` to customize:

- `model_path`: Path to model file
- `threshold`: Segmentation threshold
- `test_img_dirs`: List of test image directories
- `batch_size`: Batch size
- `num_workers`: Number of data loading threads
- `device`: Computation device ("cuda", "cpu", or "auto")

### Post-processing & Analysis

In addition to the core segmentation functionality, this project includes tools for post-processing masks and analyzing cell intensity:

#### Post-processing (Mask Clean-up)

Using the `src/postprocess.py` module or the `cell_analysis.py` script, you can:
- Smooth segmentation masks
- Remove small regions below a size threshold
- Fill holes in segmentation masks
- Create overlays of masks on original images

```bash
# Example: Process a segmentation mask
python cell_analysis.py process --mask_path outputs/test_results/20250520_154823/test_mask_0.png --original_image data/cell_dataset/img_dir/train/test_img_0.tif
```

#### Intensity Analysis

Using the `src/analysis.py` module or the `cell_analysis.py` script, you can analyze the intensity of cells across image sequences:
- Extract intensity statistics from cell regions
- Analyze intensity in ring regions around cell boundaries
- Divide cells into quadrants and octants for regional analysis
- Generate numbered cell region annotation images
- Process multiple image sequences in parallel, outputting statistics in CSV format

```bash
# Example: Analyze cell intensity
python cell_analysis.py analyze --mask_path outputs/processed_masks/processed_mask_final.png --sequence_dirs data/image_sequences/sequence1 data/image_sequences/sequence2
```

#### Combined Workflow

You can also process a mask and analyze intensity in a single command:

```bash
# Example: Process mask and analyze intensity
python cell_analysis.py process_analyze --mask_path outputs/test_results/20250520_154823/test_mask_0.png --sequence_dirs data/image_sequences/sequence1
```

### Cell Classification and Regression Analysis

In addition to image analysis, this project includes tools for cell classification and regression analysis:

#### Classification

Using the `src/predict.py` module or the `cell_predict.py` script, you can build Random Forest classification models to classify cells based on their features:

- Train/test splitting with optional undersampling for class balance
- Hyperparameter tuning using grid search
- Model evaluation with confusion matrices and ROC curves
- SHAP analysis for feature importance interpretation

```bash
# Example: Build a classification model
python cell_predict.py classify --data_path data/cell_features.xlsx

# Example: Perform SHAP analysis on a trained classification model
python cell_predict.py classify_shap --model_path outputs/classification_results/classification_model.pkl --data_path data/cell_features.xlsx
```

#### Regression

You can also build Random Forest regression models to predict continuous values like dissociation constants:

- Advanced hyperparameter tuning
- Model evaluation with MSE, RMSE, and R² metrics
- Visualization of predictions vs true values
- SHAP analysis for feature importance interpretation

```bash
# Example: Build a regression model
python cell_predict.py regress --data_path data/cell_kinetics.xlsx --target_column kd

# Example: Perform SHAP analysis on a trained regression model
python cell_predict.py regress_shap --model_path outputs/regression_results/regression_model.pkl --data_path data/cell_kinetics.xlsx
```

#### MATLAB Kinetic Fitting

The project includes reference to a MATLAB script (`prd/NIHE1_11_2.m`) for kinetic fitting of cell data. This script is not directly called from Python code but is provided as a reference for users who wish to perform kinetic fitting in MATLAB.

For detailed usage information, run:

```bash
python cell_predict.py --help
```

### Output

The inference results are saved in the `outputs/test_results/[timestamp]` directory, including:

- Binary segmentation masks (`test_mask_*.png`)
- Overlay visualizations showing the segmentation results on top of the original image (`test_overlay_*.png`)

### Troubleshooting

If you encounter issues, check:

1. Ensure all dependencies are installed correctly
2. Make sure the model file exists at the path specified in the configuration
3. Ensure test image directories contain valid images
4. Check error messages in script output for more details

#### Device Issues

If you encounter device type mismatch errors (e.g., "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"), this is typically due to models and data not being on the same device (CPU/GPU). Solutions:

1. Ensure your CUDA environment is correctly installed (if using GPU)
2. Modify `config/test_config.yaml` to add device option:

   ```yaml
   # Add this line, use 'cuda' or 'cpu'
   device: 'cpu'  
   ```

3. Or force CPU operation:

   ```bash
   python run.py --force-cpu
   ```

### Features

- **Attention U-Net**: Enhanced U-Net architecture with attention gates
- **Preprocessing**: Contrast adjustment, Otsu thresholding, distance transform
- **Post-processing**: Conditional Random Fields for refined segmentations, mask cleaning, and hole filling
- **Intensity Analysis**: Extraction of intensity statistics from cell regions, including quadrant and octant analysis
- **Visualization**: Segmentation masks, overlays, and region visualizations

### Model Details

The Attention U-Net model combines the traditional U-Net architecture with attention gates that help the model focus on relevant features during segmentation. This is particularly useful for cell segmentation where boundaries can be complex and subtle.



---

<a name="chinese"></a>
## 中文版

本项目实现了一个用于显微镜图像中细胞分割的Attention U-Net模型。该模型结合了注意力门控机制和U-Net架构，以提高分割精度。

### 项目结构

```
UNetSLSM/
│
├── src/                  # 源代码
│   ├── __init__.py       # 包初始化
│   ├── model.py          # Attention U-Net 模型定义
│   ├── data.py           # 数据加载和处理
│   ├── inference.py      # 推理函数
│   └── app.py            # 主应用程序
│
├── config/               # 配置文件
│   ├── train_config.yaml     # 训练配置
│   ├── inference_config.yaml # 推理配置
│   └── test_config.yaml      # 测试配置
│
├── data/                 # 数据目录
│   └── cell_dataset/     # 细胞图像
│       ├── img_dir/      # 图像目录
│       │   ├── train/    # 训练图像
│       │   └── val/      # 验证图像
│       └── mask_dir/     # 掩码目录
│           ├── train/    # 训练掩码
│           └── val/      # 验证掩码
│
├── outputs/              # 输出目录
│   └── test_results/     # 测试结果
│
├── runs/                 # 训练运行目录（模型）
│
├── main.py               # 主脚本用于推理
├── train.py              # 训练脚本
├── run.py                # 入口脚本
└── requirements.txt      # 包依赖
```

### 系统要求

完整的依赖项请参见`requirements.txt`。主要要求包括：

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- PyDenseCRF
- Albumentations
- scikit-learn
- PIL
- PyYAML

### 安装说明

1. 克隆此仓库
2. 安装依赖项：

```bash
pip install -r requirements.txt
```

3. 将测试图像放入`data/cell_dataset/img_dir/train/`目录。
4. 确保配置文件中指定的路径中有可用的模型文件。

### 使用方法

项目提供了两种运行方式：结构测试和模型推理。

#### 结构测试

结构测试用于验证项目结构和模块是否正确加载，无需实际模型权重文件。

```bash
python run.py --test-only
```

#### 模型推理

模型推理用于使用训练好的模型对测试图像进行分割。

```bash
python run.py
```

对于仅CPU执行（如果您遇到GPU相关问题）：

```bash
python run.py --force-cpu
```

### 配置

编辑`config/test_config.yaml`以自定义：

- `model_path`: 模型文件的路径
- `threshold`: 分割阈值
- `test_img_dirs`: 测试图像目录列表
- `batch_size`: 批处理大小
- `num_workers`: 数据加载线程数
- `device`: 计算设备（"cuda"、"cpu"或"auto"）

### 后处理与分析

除了核心分割功能外，本项目还包括用于后处理掩码和分析细胞强度的工具：

#### 后处理（掩码清理）

使用`src/postprocess.py`模块或`cell_analysis.py`脚本，您可以：
- 平滑分割掩码
- 移除低于尺寸阈值的小区域
- 填充分割掩码中的孔洞
- 创建掩码与原始图像的叠加图

```bash
# 示例：处理分割掩码
python cell_analysis.py process --mask_path outputs/test_results/20250520_154823/test_mask_0.png --original_image data/cell_dataset/img_dir/train/test_img_0.tif
```

#### Intensity Analysis

使用`src/analysis.py`模块或`cell_analysis.py`脚本，您可以分析跨图像序列的细胞强度：
- 从细胞区域提取强度统计数据
- 分析细胞边界周围环形区域的强度
- 将细胞分成象限和八分区进行区域分析
- 生成带有编号的细胞区域标注图像
- 并行处理多个图像序列，输出CSV格式的统计结果

```bash
# 示例：分析细胞强度
python cell_analysis.py analyze --mask_path outputs/processed_masks/processed_mask_final.png --sequence_dirs data/image_sequences/sequence1 data/image_sequences/sequence2
```

#### 组合工作流

您还可以在一个命令中处理掩码并分析强度：

```bash
# 示例：处理掩码并分析强度
python cell_analysis.py process_analyze --mask_path outputs/test_results/20250520_154823/test_mask_0.png --sequence_dirs data/image_sequences/sequence1
```

### Cell Classification and Regression Analysis

In addition to image analysis, this project includes tools for cell classification and regression analysis:

#### Classification

Using the `src/predict.py` module or the `cell_predict.py` script, you can build Random Forest classification models to classify cells based on their features:

- Train/test splitting with optional undersampling for class balance
- Hyperparameter tuning using grid search
- Model evaluation with confusion matrices and ROC curves
- SHAP analysis for feature importance interpretation

```bash
# Example: Build a classification model
python cell_predict.py classify --data_path data/cell_features.xlsx

# Example: Perform SHAP analysis on a trained classification model
python cell_predict.py classify_shap --model_path outputs/classification_results/classification_model.pkl --data_path data/cell_features.xlsx
```

#### Regression

You can also build Random Forest regression models to predict continuous values like dissociation constants:

- Advanced hyperparameter tuning
- Model evaluation with MSE, RMSE, and R² metrics
- Visualization of predictions vs true values
- SHAP analysis for feature importance interpretation

```bash
# Example: Build a regression model
python cell_predict.py regress --data_path data/cell_kinetics.xlsx --target_column kd

# Example: Perform SHAP analysis on a trained regression model
python cell_predict.py regress_shap --model_path outputs/regression_results/regression_model.pkl --data_path data/cell_kinetics.xlsx
```

#### MATLAB Kinetic Fitting

The project includes reference to a MATLAB script (`prd/NIHE1_11_2.m`) for kinetic fitting of cell data. This script is not directly called from Python code but is provided as a reference for users who wish to perform kinetic fitting in MATLAB.

For detailed usage information, run:

```bash
python cell_predict.py --help
```

### 输出

推理结果保存在`outputs/test_results/[timestamp]`目录中，包括：

- 分割掩码图像（`test_mask_*.png`）
- 原始图像与分割结果的叠加图（`test_overlay_*.png`）

### 故障排除

如果遇到问题，请检查：

1. 确保所有依赖项都已正确安装
2. 确保模型文件存在于配置指定的路径
3. 确保测试图像目录包含有效图像
4. 查看脚本输出的错误信息以获取更多详情

#### 设备问题

如果您遇到设备类型不匹配错误（例如 "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"），这通常是由于模型和数据不在同一设备（CPU/GPU）上。解决方法：

1. 确保您的CUDA环境正确安装（如果使用GPU）
2. 修改`config/test_config.yaml`添加设备选项：

   ```yaml
   # 添加此行，使用'cuda'或'cpu'
   device: 'cpu'  
   ```

3. 或者强制使用CPU运行：

   ```bash
   python run.py --force-cpu
   ```

### 功能特点

- **Attention U-Net**：增强的U-Net架构，具有注意力门控机制
- **预处理**：对比度调整、Otsu阈值处理、距离变换
- **后处理**：条件随机场用于细化分割结果、掩码清理和孔洞填充
- **强度分析**：从细胞区域提取强度统计数据，包括象限和八分区分析
- **可视化**：分割掩码、叠加图和区域可视化

## 引用

如果您发现我们的工作有用，请考虑引用：

```bibtex
@Article{mESN,
author={ Caixin Huang
and Jingbo Zhang
and Zhaoyang Liu
and Min Wang
and Liangju Li
and Jiying Xu
and Yi Chen
and Ying Zhao
and Pengfei Zhang},
title={ AI-Driven Multi-Parametric Evanescent Scattering Microscopy Deciphers Subcellular Heterogeneity of Ligand Binding Kinetics and Cell Adhesion-Mediated Regulation},
}
```

## 联系方式

如有任何问题，请联系：

Ying Zhao: zhaoying@xxmu.edu.cn  
Pengfei Zhang: pfzhang@iccas.ac.cn
