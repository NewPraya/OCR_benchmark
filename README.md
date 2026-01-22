# OCR Benchmark Framework

针对LLM视觉能力的OCR基准测试框架，支持V1文本提取和V2结构化提取两种模式。

## 🚀 5分钟快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置API密钥
cp env.example .env
# 编辑 .env 填入你的API keys

# 3. 运行benchmark
python3 main.py -v v1 -m gemini -id gemini-2.0-flash-exp

# 4. 查看结果
streamlit run app.py
```

## 📋 完整操作流程

### ⚖️ 公平评测与鲁棒性优化 (Fair Evaluation)

项目内置了针对不同模型输出风格的鲁棒性优化方案，确保评测结果真实反映视觉能力：

- **Prompt 强化**：内置 Prompt 使用了“精确数据员”指令，强制模型关注勾选框状态（V/X/O/Circle）而非盲猜，并严格限制输出格式。
- **V1 文本归一化**：自动处理标点符号、全半角转换、特殊字符干扰。即便模型自动修正了标点或添加了序号，也不会因此扣分。
- **V2 模糊匹配**：
  - **键名对齐**：支持中英文键名自动映射（如识别出“心脏病”会自动对齐到“Heart Disease”）。
  - **逻辑值归一化**：将 `True/False`, `Yes/No`, `Checked/Unchecked`, `V/X` 统一映射为 `Y/N` 进行比对。
  - **实体模糊匹配**：支持子串匹配，解决识别文本微小差异导致的得分断崖。

### 步骤1：制作Schema配置（V2模式）

**如果使用V1模式（纯文本OCR），跳过此步骤。**

#### 1.1 复制模板

```bash
# 使用医疗表单模板（默认）
cp schemas/medical_form.yaml schemas/my_schema.yaml

# 或使用发票模板
cp schemas/invoice.yaml schemas/my_schema.yaml
```

#### 1.2 编辑Schema

编辑 `schemas/my_schema.yaml`：

```yaml
schema_name: "my_document"
version: "v2"
description: "你的文档类型描述"

fields:
  # 字段1：分类字典（Y/N选择、单选题等）
  - name: "field1_name"
    type: "categorical_dict"     # 类型：categorical_dict, entity_list, text_dict, numerical_dict
    evaluation: "accuracy"        # 评估：accuracy, f1, pairing, exact_match
    weight: 0.3                   # 权重：0-1之间，会自动归一化
    description: "字段说明"
    
  # 字段2：实体列表（关键词提取等）
  - name: "field2_name"
    type: "entity_list"
    evaluation: "f1"
    weight: 0.4
    description: "字段说明"

# LLM提取prompt
prompt_template: |
  请分析这个文档，返回JSON对象包含：
  1. 'field1_name': {...}
  2. 'field2_name': [...]
  只返回JSON，不要markdown代码块。
```

**字段类型速查：**
- `categorical_dict`: 字典 `{"q1": "Y", "q2": "N"}` → 用于选择题
- `entity_list`: 列表 `["实体1", "实体2"]` → 用于关键词提取
- `text_dict`: 字典 `{"字段": "文本"}` → 用于字段配对
- `numerical_dict`: 字典 `{"total": 100.5}` → 用于数值字段

**评估方法速查：**
- `accuracy`: 精确匹配 → 用于categorical_dict
- `f1`: F1分数 → 用于entity_list
- `pairing`: 模糊匹配 → 用于text_dict
- `exact_match`: 严格相等 → 用于numerical_dict

#### 1.3 验证Schema

```bash
python3 -c "
from schemas.schema_base import SchemaLoader
schema = SchemaLoader.load_schema('schemas/my_schema.yaml')
print('✓ Schema加载成功')
print(f'字段: {[f.name for f in schema.fields]}')
print(f'权重: {schema.weights}')
"
```

### 步骤2：制作标准答案（Ground Truth）

#### 2.1 准备图片

将图片放到 `data/` 目录：
```bash
cp your_images/*.png data/
```

#### 2.2 创建Ground Truth JSON

**V1模式** - 创建 `data/sample_gt.json`：
```json
[
  {
    "file_name": "sample.png",
    "text": "识别的文本内容..."
  }
]
```

**V2模式** - 创建 `data/sample_gt_v2.json`（结构要匹配schema）：
```json
[
  {
    "file_name": "sample.png",
    "field1_name": {"q1": "Y", "q2": "N"},
    "field2_name": ["实体1", "实体2"]
  }
]
```

#### 2.3 辅助标注（可选）

使用Gemini自动生成初稿，然后人工修正：

```bash
# V1模式
python3 utils/prep_labels.py -v v1
# 生成 labeling_v1/*.md 文件

# V2模式
python3 utils/prep_labels.py -v v2
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
python3 main.py -v v1 -m gemini -id gemini-2.0-flash-exp

# V2模式（结构化提取，默认医疗表单）
python3 main.py -v v2 -m gemini -id gemini-2.0-flash-exp
```

#### 3.2 支持的模型

```bash
# Gemini
python3 main.py -v v1 -m gemini -id gemini-2.0-flash-exp

# OpenAI GPT-4V
python3 main.py -v v1 -m openai -id gpt-4o

# OpenAI GPT-5（如遇到 Request timed out，建议提高超时 + 开启重试）
# export OPENAI_TIMEOUT_SECONDS=180
# export OPENAI_OCR_MAX_ATTEMPTS=3
# export OPENAI_VERBOSE_RETRIES=true
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
- `OPENAI_FALLBACK_TO_CHAT`：是否允许 `responses` 失败后回退到 `chat.completions`
- `OPENAI_BASE_URL`：可选，代理/网关地址

#### 3.3 使用自定义Schema（V2模式）

项目支持“双用模式”，你可以选择使用内置的医疗表单逻辑，或者使用更灵活的 YAML Schema。

**方式 A：使用内置医疗表单逻辑（默认）**
此模式使用 `utils/prompts.py` 中预定义的提示词和 `evaluators/evaluator_v2.py` 中的硬编码评估逻辑。
```bash
python3 main.py -v v2 -m gemini -id gemini-2.0-flash-exp
```

**方式 B：使用自定义 Schema（推荐，动态加载）**
通过 `-s` 参数指定 YAML 配置文件。系统会自动从 YAML 中读取 `prompt_template`，并使用通用的 `SchemaBasedEvaluator` 进行评估。这种方式更适合扩展到不同类型的文档（如发票、合同）。
```bash
python3 main.py -v v2 -s schemas/medical_form.yaml -m gemini -id gemini-2.0-flash-exp
```

> **提示**：当你使用 `-s` 模式时，系统将**完全绕过** `utils/prompts.py` 中的提示词，转而使用 YAML 中的配置。



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
- Logical Acc（逻辑值准确率）
- Disease Acc（疾病状态准确率）
- Entity F1（实体F1分数）
- Entity Precision & Recall
- Pairing Acc（字段配对准确率）

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
Model ID & Weighted Score & Logical Acc & Entity F1 & Pairing Acc & Samples \\
\midrule
gemini-2.0 & 0.8742 & 0.9286 & 0.8500 & 0.8125 & 1 \\
gpt-4o & 0.8521 & 0.9143 & 0.8200 & 0.8000 & 1 \\
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

### Schema设计建议

**权重分配：**
- 核心字段（如金额、ID）：0.3-0.4
- 重要字段（如日期、名称）：0.2-0.3
- 次要字段（如备注）：0.1-0.2

**字段数量：**
- 建议2-6个字段
- 太多会影响评估效率

**评估方法选择：**
| 字段内容 | 推荐方法 | 示例 |
|---------|---------|------|
| Y/N选项、单选题 | `accuracy` | {"q1": "Y"} |
| 关键词、实体提取 | `f1` | ["NPC", "RT"] |
| 文本配对、地址 | `pairing` | {"地址": "北京市..."} |
| 金额、ID号 | `exact_match` | {"total": 100.5} |

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

### Q1: 如何添加新文档类型？
1. 复制schema模板：`cp schemas/medical_form.yaml schemas/新文档.yaml`
2. 编辑字段定义和prompt
3. 准备对应的ground truth
4. 修改main.py使用新schema

### Q2: 为什么我的结果这么低？
- 检查Ground Truth是否正确
- 确认prompt是否清晰
- V2模式：检查JSON格式是否匹配schema
- 尝试不同的模型对比

### Q3: 如何处理中英文混合文档？
- 框架完全支持中英文混合
- Ground Truth中直接写中英文
- Schema字段名建议用英文，描述可以用中文

### Q4: Dashboard显示No results怎么办？
- 确认benchmark已运行：`ls results/preds_*.json`
- 检查文件命名格式：`preds_v1_模型名.json`
- 确认Ground Truth存在：`ls data/sample_gt*.json`

### Q5: 如何报告置信区间？
在Statistical Analysis标签：
1. 运行Bootstrap置信区间
2. 记录95% CI范围
3. 论文中写：`0.87 (95% CI: [0.84, 0.90])`

### Q6: LaTeX表格导出后格式有问题？
- 确保使用`booktabs`包：`\usepackage{booktabs}`
- 检查列对齐：`lrrr`（l=左对齐，r=右对齐）
- 手动调整小数位数

### Q7: 如何加快benchmark速度？
- 使用更快的模型（如gemini-flash vs gemini-pro）
- 减少样本数量快速测试
- 使用dummy模型调试流程

## 📁 项目结构

```
OCR_benchmark/
├── data/                          # 图片和Ground Truth
│   ├── *.png                      # 测试图片
│   ├── sample_gt.json             # V1 Ground Truth
│   └── sample_gt_v2.json          # V2 Ground Truth
│
├── schemas/                       # Schema配置（V2模式）
│   ├── medical_form.yaml          # 医疗表单schema
│   └── invoice.yaml               # 发票schema（示例）
│
├── models/                        # 模型实现
│   ├── gemini_model.py
│   ├── openai_model.py
│   └── qwen_model.py
│
├── evaluators/                    # 评估器
│   ├── evaluator.py               # V1评估器
│   ├── evaluator_v2.py            # V2评估器（医疗表单）
│   ├── schema_evaluator.py        # 通用Schema评估器
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
python3 main.py -v v1 -m gemini -id gemini-2.0-flash-exp    # V1模式
python3 main.py -v v2 -m gemini -id gemini-2.0-flash-exp    # V2模式

# 辅助制作标注
python3 utils/prep_labels.py -v v2
python3 utils/sync_to_gt.py -v v2

# 启动Dashboard
streamlit run app.py

# 验证Schema（V2模式）
python3 -c "from schemas.schema_base import SchemaLoader; print(SchemaLoader.load_schema('schemas/medical_form.yaml'))"
```

## 📊 指标说明

**V1指标：** CER↓, WER↓, NED↓, Precision↑, Recall↑, BoW F1↑, Exact Match↑  
**V2指标：** Weighted Score↑, Logical Acc↑, Entity F1↑, Pairing Acc↑

（↑越高越好，↓越低越好）

---

**完整技术文档**: 见 `IMPLEMENTATION_SUMMARY.md`

