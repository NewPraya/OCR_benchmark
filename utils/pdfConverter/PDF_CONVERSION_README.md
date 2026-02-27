# PDF to PNG Conversion Guide

本文档说明如何将 `data/prepare/` 目录中的所有PDF文件转换为PNG格式，并按序号重命名（从6开始）。

## 文件说明

- `convert_pdf_to_png.py` - PDF转PNG的主要脚本
- `requirements_pdf_convert.txt` - Python依赖包列表
- `setup_and_convert.sh` - 一键安装依赖并执行转换的脚本

## 快速开始

### 方法1：使用一键脚本（推荐）

```bash
./setup_and_convert.sh
```

这个脚本会自动完成以下操作：
1. 激活 `OCR_benchmark` conda环境
2. 安装 poppler（系统依赖）
3. 安装 Python依赖（pdf2image, Pillow）
4. 执行PDF转PNG转换

### 方法2：手动执行

如果你想分步执行，可以按以下步骤操作：

```bash
# 1. 激活conda环境
conda activate OCR_benchmark

# 2. 安装系统依赖（poppler）
conda install -c conda-forge poppler -y

# 3. 安装Python依赖
pip install -r requirements_pdf_convert.txt

# 4. 运行转换脚本
python convert_pdf_to_png.py
```

## 输出结果

转换后的PNG文件将保存在 `data/prepare/png_output/` 目录中，文件名为：
- `6.png`
- `7.png`
- `8.png`
- ...
- `49.png`

共44个文件（当前目录中有44个PDF文件）。

## 技术细节

- **DPI设置**: 300 DPI（高质量）
- **转换页数**: 仅转换PDF的第一页
- **输出格式**: PNG
- **命名规则**: 按字母顺序处理PDF文件，依次命名为6, 7, 8...

## 依赖说明

### 系统依赖
- **poppler**: PDF渲染引擎，`pdf2image`库需要此系统工具

### Python依赖
- **pdf2image** (>=1.17.0): 将PDF转换为图像
- **Pillow** (>=10.0.0): Python图像处理库

## 故障排除

### 如果遇到 "poppler not found" 错误

确保已安装poppler：
```bash
conda activate OCR_benchmark
conda install -c conda-forge poppler -y
```

### 如果某些PDF无法转换

脚本会跳过出错的文件并继续处理其他文件。查看控制台输出，会显示哪些文件处理成功/失败。

## 自定义设置

如果需要修改起始编号或其他参数，可以编辑 `convert_pdf_to_png.py` 文件：

```python
# 修改起始编号
convert_pdfs_to_pngs(input_directory, output_directory, start_number=1)  # 从1开始

# 修改DPI（影响图片质量和大小）
images = convert_from_path(pdf_file, dpi=150)  # 降低DPI
```
