# OCR Benchmark Framework

针对LLM视觉能力的OCR基准测试框架，支持V1文本提取和V2简化模式（手写文本 + Y/N判断）。

## 🚀 5分钟快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载测试数据
# 访问 https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=sharing
# 将图片和 dataset_split.json 下载到 data/ 目录

# 3. 配置API密钥
cp env.example .env
# 编辑 .env 填入你的API keys

# 4. 运行benchmark
python3 main.py -v v1 -m gemini -id gemini-2.0-flash-exp

# 5. 查看结果
streamlit run app.py
```

## 📋 完整操作流程

### ⚖️ 公平评测与鲁棒性优化 (Fair Evaluation)

项目内置了针对不同模型输出风格的鲁棒性优化方案，确保评测结果真实反映视觉能力：

- **Prompt 强化**：V2提示词统一为“手写文本 + Y/N判断”，不区分表格类型，强制关注勾选框状态并要求JSON输出。
- **V1 文本归一化**：自动处理标点符号、全半角转换、特殊字符干扰。即便模型自动修正了标点或添加了序号，也不会因此扣分。
- **V2 模糊匹配**：
  - **键名对齐**：Y/N项支持中英文键名自动映射（如识别出“心脏病”会自动对齐到“Heart Disease”）。
  - **逻辑值归一化**：将 `True/False`, `Yes/No`, `Checked/Unchecked`, `V/X` 统一映射为 `Y/N` 进行比对。
  - **手写文本匹配**：对手写内容做归一化并用编辑距离评估（CER/WER/NED）。

### 步骤1：下载测试数据

本项目的测试图片和标注数据托管在 Google Drive（约100MB）：

📦 **数据下载地址**：[https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=sharing](https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=sharing)

> **为什么使用 Google Drive？**  
> 图片数据较大（~100MB），直接上传到GitHub会导致仓库臃肿。使用Google Drive托管数据可以：
> - 保持Git仓库轻量（便于clone和fork）
> - 加快代码更新速度
> - 数据文件已在`.gitignore`中排除，不会意外提交

**数据包内容：**
- 48张测试图片（PNG格式，总计约100MB）
  - 手写文档（hw1.png - hw23.png）：23张手写表格
  - 打印表格（pn1.png - pn8.png）：8张打印表单
  - 混合文档（cf1.png - cf4.png）：4张复杂表格
  - 其他文档（hd1.png - hd4.png, ess1.png, rl1.png - rl4.png等）：13张
- `dataset_split.json`（V1/V2数据集划分配置）

**下载步骤：**
1. 访问上述链接
2. 点击右上角"⋮"菜单 → "下载"下载全部文件（或单独选择文件下载）
3. 将下载的图片和JSON文件放到项目的 `data/` 目录

### 步骤2：制作标准答案（Ground Truth）

#### 2.1 准备图片

如果你想使用自己的图片，可以将它们放到 `data/` 目录：
```bash
cp your_images/*.png data/
```

#### 2.2 创建Ground Truth JSON

**V1模式** - 创建 `data/sample_gt_v1.json`（旧版也兼容 `data/sample_gt.json`）：
```json
[
  {
    "file_name": "sample.png",
    "text": "识别的文本内容..."
  }
]
```

**V2模式** - 创建 `data/sample_gt_v2.json`（用于评估“手写文本 + Y/N”）：
```json
[
  {
    "file_name": "sample.png",
    "handwriting_text": "手写内容...\n第二行...",
    "yn_options": {
      "Question A": "Y",
      "Question B": "N"
    }
  }
]
```

**可选：用一个文件配置V1/V2图片清单**  
如果你想复用同一批图片、但让V1/V2评测各自选取子集，可在 `data/dataset_split.json` 中声明：
```json
{
  "v1": ["1.png", "2.png", "3.png"],
  "v2": ["1.png", "4.png"]
}
```
程序会自动读取该文件（存在即生效），你也可以通过 `--split` 指定其他路径。

#### 2.3 辅助标注（可选）

使用Gemini自动生成初稿，然后人工修正：

```bash
# V1模式
python3 utils/prep_labels.py -v v1 --split data/dataset_split.json
# 生成 labeling_v1/*.md 文件

# V2模式
python3 utils/prep_labels.py -v v2 --split data/dataset_split.json
# 生成 labeling_v2/*.md 文件

# 在Cursor/VSCode中：
# 1. 打开.md文件
# 2. 按Cmd+Shift+V预览（可看到图片）
# 3. 编辑文本/JSON
# 4. 保存

# 同步回GT JSON
python3 utils/sync_to_gt.py -v v1  # 或 -v v2
```

### 步骤3：运行Benchmark

#### 3.1 基本用法

```bash
# V1模式（文本OCR）
python3 main.py -v v1 -m gemini -id gemini-2.0-flash-exp --split data/dataset_split.json

# V2模式（简化：手写文本 + Y/N）
python3 main.py -v v2 -m gemini -id gemini-2.0-flash-exp --split data/dataset_split.json
```

#### 3.2 支持的模型

```bash
# Gemini
python3 main.py -v v1 -m gemini -id gemini-2.0-flash-exp

# OpenAI GPT-4V
python3 main.py -v v1 -m openai -id gpt-4o

# OpenAI 更快基线（建议先用 mini 版本排查慢请求）
python3 main.py -v v1 -m openai -id gpt-4.1-mini

# OpenAI GPT-5（如遇到 Request timed out，建议提高超时 + 开启重试）
# export OPENAI_TIMEOUT_SECONDS=180
# export OPENAI_OCR_MAX_ATTEMPTS=3
# export OPENAI_VERBOSE_RETRIES=true
# export OPENAI_RESPONSES_ONLY=true
python3 main.py -v v1 -m openai -id gpt-5

# Qwen
python3 main.py -v v1 -m qwen -id qwen-vl-max

# 测试用Dummy模型
python3 main.py -v v1 -m dummy -id dummy
```

#### 3.4 OpenAI 超时/重试参数（可选）

将下列配置写入你的 `.env`（或在 shell 里 `export`）：

- `OPENAI_TIMEOUT_SECONDS`：单次请求超时（秒），默认 120
- `OPENAI_MAX_RETRIES`：openai-python SDK 内部重试次数，默认 2
- `OPENAI_OCR_MAX_ATTEMPTS`：本项目外层重试次数（对 timeout/5xx/429 生效），默认 3
- `OPENAI_RETRY_BACKOFF_SECONDS` / `OPENAI_RETRY_BACKOFF_MAX_SECONDS`：指数退避参数
- `OPENAI_VERBOSE_RETRIES`：打印更详细的重试/回退日志（true/false）
- `OPENAI_RESPONSES_ONLY`：是否强制只走 `responses` API（默认 true）
- `OPENAI_FALLBACK_TO_CHAT`：是否允许 `responses` 失败后回退到 `chat.completions`
- `OPENAI_BASE_URL`：可选，代理/网关地址
- `OPENAI_IMAGE_MAX_SIDE`：上传前把图片长边缩放到指定像素（默认 1600，越小越快）
- `OPENAI_IMAGE_JPEG_QUALITY`：上传前 JPEG 压缩质量（默认 85）
- `OPENAI_IMAGE_DETAIL`：视觉细节级别（`low/high/auto`，默认 `low`，通常更快）
- `OPENAI_MAX_OUTPUT_TOKENS`：输出 token 上限（默认 2048，防止超长生成）
- `OPENAI_REASONING_EFFORT`：GPT-5 推理强度（`minimal/low/medium/high`，OCR 场景建议 `minimal`）

#### 3.3 V2模式输出格式（简化）

V2统一输出JSON，不依赖任何schema：
```json
{
  "handwriting_text": "手写内容...",
  "yn_options": {
    "Question A": "Y",
    "Question B": "N"
  }
}
```

#### 3.5 批量生成评估报告

运行benchmark后，会自动生成报告文件（`results/report_*.json`）。如果报告丢失或需要重新生成，可以使用工具脚本：

```bash
# 为所有版本生成报告
python3 utils/generate_reports.py

# 只为V1生成报告
python3 utils/generate_reports.py --version v1

# 只为V2生成报告
python3 utils/generate_reports.py --version v2
```

**功能说明：**
- 自动扫描 `results/preds_*.json` 预测文件
- 读取对应的Ground Truth（`data/sample_gt_v1.json` 或 `data/sample_gt_v2.json`）
- 计算评估指标并生成 `results/report_*.json` 报告文件
- 跳过已存在的报告（避免重复计算）

**使用场景：**
- 删除了report文件后需要重新生成
- 更新了Ground Truth需要重新评估
- 修改了评估逻辑需要重算所有结果

### 步骤4：查看评估结果

#### 4.1 启动Dashboard

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

#### 4.2 Dashboard功能

**Tab 1: 📊 Leaderboard（排行榜）**
- 查看所有模型的排名
- 对比各项指标
- 查看汇总统计

**V1指标：**
- CER（字符错误率）- 越低越好
- WER（词错误率）- 越低越好
- NED（归一化编辑距离）- 越低越好
- Precision（精确率）- 越高越好
- Recall（召回率）- 越高越好
- BoW F1（词袋F1）- 越高越好
- Exact Match（完全匹配率）- 越高越好

**V2指标：**
- Weighted Score（加权总分）- 越高越好
- Y/N Acc（Y/N准确率）
- Handwriting CER/WER/NED（手写文本错误率，越低越好）

**Tab 2: 🔍 Detailed View（详细对比）**
- 选择图片查看原图
- 对比Ground Truth和预测结果
- 多模型并排对比

**Tab 3: 📈 Statistical Analysis（统计分析）**
- 选择两个模型对比
- 查看p-value、置信区间
- 判断差异是否显著
- 箱线图可视化

**Tab 4: 📤 Export（导出）**
- **LaTeX表格**：直接复制到论文
- **CSV文件**：用于Excel分析
- **JSON文件**：原始数据

### 步骤5：导出论文用的表格

#### 5.1 在Dashboard中导出

1. 打开Dashboard → Export标签
2. 输入表格标题
3. 点击"Generate LaTeX"
4. 复制代码到论文

#### 5.2 LaTeX表格示例

```latex
\begin{table}
\caption{OCR Benchmark Results (V2 Mode)}
\begin{tabular}{lrrrrr}
\toprule
Model ID & Weighted Score & Y/N Acc & HW CER & HW WER & Samples \\
\midrule
gemini-2.0 & 0.8742 & 0.9286 & 0.1200 & 0.1800 & 1 \\
gpt-4o & 0.8521 & 0.9143 & 0.1400 & 0.2000 & 1 \\
\bottomrule
\end{tabular}
\label{tab:results}
\end{table}
```

#### 5.3 统计分析结果

在Statistical Analysis标签：
1. 选择指标（如Weighted Score）
2. 选择两个模型
3. 运行统计测试
4. 记录p-value和置信区间

**论文中可以写**：
> Model A achieved significantly higher performance than Model B (Weighted Score: 0.87 ± 0.03 vs. 0.81 ± 0.04, p < 0.001).

## 💡 实用技巧

### V2标注建议

- **手写内容**：尽量保留原始行/顺序，避免自行纠错
- **Y/N判断**：只要明确勾选/圈选才记Y，其他情况一律记N

### Ground Truth制作建议

**质量控制：**
- 至少2人标注关键样本
- 使用`prep_labels.py`生成初稿节省时间
- 在Markdown预览中对照图片修正

**样本数量：**
- 最少：10-20个（快速测试）
- 建议：30-50个（统计有效）
- 论文：50+个（学术标准）

### 多模型对比

批量运行多个模型：
```bash
# 创建脚本 run_all.sh
for model_id in gemini-2.0-flash-exp gpt-4o qwen-vl-max; do
  python3 main.py -v v2 -m gemini -id $model_id
done

# 运行
bash run_all.sh

# 在Dashboard中对比所有结果
streamlit run app.py
```

## 🔧 常见问题

### Q1: 如何获取测试数据？
从 Google Drive 下载：[https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=sharing](https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=sharing)

将下载的文件放到 `data/` 目录即可。数据包包含48张测试图片和数据集划分配置文件。

### Q2: 如何添加新文档类型？
V2已经统一为“手写文本 + Y/N”模式，不需要额外配置。只要保证GT包含Y/N和手写相关字段即可。

### Q3: 为什么我的结果这么低？
- 检查Ground Truth是否正确
- 确认prompt是否清晰
- V2模式：检查是否严格输出JSON（`handwriting_text` + `yn_options`）
- 尝试不同的模型对比

### Q4: 如何处理中英文混合文档？
- 框架完全支持中英文混合
- Ground Truth中直接写中英文
- Y/N标签可以中英文混用，评估会做键名归一化

### Q5: Dashboard显示No results怎么办？
- 确认benchmark已运行：`ls results/preds_*.json`
- 检查文件命名格式：`preds_v1_模型名.json`
- 确认Ground Truth存在：`ls data/sample_gt*.json`

### Q6: 如何报告置信区间？
在Statistical Analysis标签：
1. 运行Bootstrap置信区间
2. 记录95% CI范围
3. 论文中写：`0.87 (95% CI: [0.84, 0.90])`

### Q7: LaTeX表格导出后格式有问题？
- 确保使用`booktabs`包：`\usepackage{booktabs}`
- 检查列对齐：`lrrr`（l=左对齐，r=右对齐）
- 手动调整小数位数

### Q8: 如何加快benchmark速度？
- 使用更快的模型（如gemini-flash vs gemini-pro）
- 减少样本数量快速测试
- 使用dummy模型调试流程

## 📁 项目结构

```
OCR_benchmark/
├── data/                          # 图片和Ground Truth
│   ├── *.png                      # 测试图片（48张，从Google Drive下载）
│   │                              # 链接：https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf
│   ├── sample_gt.json             # V1 Ground Truth（旧版兼容）
│   ├── sample_gt_v1.json          # V1 Ground Truth
│   ├── sample_gt_v2.json          # V2 Ground Truth
│   └── dataset_split.json         # V1/V2图片清单（从Google Drive下载）
│
├── schemas/                       # 历史schema配置（已弃用）
│   ├── medical_form.yaml
│   └── invoice.yaml
│
├── models/                        # 模型实现
│   ├── gemini_model.py
│   ├── openai_model.py
│   └── qwen_model.py
│
├── evaluators/                    # 评估器
│   ├── evaluator.py               # V1评估器
│   ├── evaluator_v2.py            # V2评估器（简化模式）
│   ├── schema_evaluator.py        # 通用Schema评估器（已弃用）
│   ├── metrics.py                 # 指标计算
│   └── statistical_tests.py       # 统计检验
│
├── results/                       # 输出结果
│   └── preds_v1_*.json            # 预测结果
│
├── utils/                         # 工具脚本
│   ├── prompts.py                 # Prompt定义
│   ├── prep_labels.py             # 辅助标注
│   └── sync_to_gt.py              # 同步GT
│
├── main.py                        # 主运行脚本
├── app.py                         # Streamlit Dashboard
└── requirements.txt               # 依赖包
```

## 📚 命令速查表

```bash
# 安装
pip install -r requirements.txt

# 配置API密钥
cp env.example .env
# 编辑 .env 填入密钥

# 运行benchmark
python3 main.py -v v1 -m gemini -id gemini-3-flash-preview    # V1模式
python3 main.py -v v2 -m gemini -id gemini-3-flash-preview    # V2模式

# 辅助制作标注
python3 utils/prep_labels.py -v v2
python3 utils/sync_to_gt.py -v v2

# 批量生成评估报告
python3 utils/generate_reports.py              # 所有版本
python3 utils/generate_reports.py --version v1 # 仅V1
python3 utils/generate_reports.py --version v2 # 仅V2

# 启动Dashboard
streamlit run app.py

# schema模式已弃用，无需验证
```

## 📊 指标说明

**V1指标：** CER↓, WER↓, NED↓, Precision↑, Recall↑, BoW F1↑, Exact Match↑  
**V2指标：** Weighted Score↑, Y/N Acc↑, Handwriting CER/WER/NED↓

（↑越高越好，↓越低越好）

---

**完整技术文档**: 见 `IMPLEMENTATION_SUMMARY.md`

