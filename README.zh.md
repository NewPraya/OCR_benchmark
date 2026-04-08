# OCR Benchmark 框架

面向视觉大模型的 OCR 基准评测框架，支持可复现实验、消融评估与可视化分析。

## 1. 项目能力

- 统一多模型调用：`openai`、`gemini`、`qwen`、`ollama`、`dummy`
- 两种评测模式：
1. `v1`：整段文本 OCR
2. `v2`：手写文本 + Y/N 选项识别（JSON 输出）
- 支持后处理消融：`--no-postprocess`
- 提供统计检验与 Streamlit Dashboard

## 2. 目录说明

- `main.py`：命令行跑批入口
- `app.py`：可视化看板
- `models/`：模型适配层
- `evaluators/`：指标与评估逻辑
- `utils/`：提示词、数据划分、报表等工具
- `tests/`：核心评估回归测试
- `data/`：图片与 GT
- `results/`：预测与报告输出

## 3. 安装与配置

```bash
pip install -r requirements.txt
cp env.example .env
```

在 `.env` 配置：
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `DASHSCOPE_API_KEY`

可复现参数：
- `OCR_BENCHMARK_SEED`（默认 `42`，用于 bootstrap 采样）

## 4. 数据要求

需要至少准备：
- `data/sample_gt_v1.json`（或兼容旧版 `data/sample_gt.json`）
- `data/sample_gt_v2.json`
- 可选：`data/dataset_split.json`（控制 v1/v2 子集）

图片文件不随仓库分发，请先下载到 `data/` 目录，文件名需与 GT 中 `file_name` 对齐。

当前图片下载地址：
- Google Drive: [dataset images](https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=drive_link)

## 5. 运行评测

```bash
# V1 文本 OCR
python main.py -v v1 -m openai -id gpt-4.1-mini

# V2 手写文本 + Y/N
python main.py -v v2 -m gemini -id gemini-2.0-flash-exp

# 关闭后处理（消融）
python main.py -v v2 -m openai -id gpt-4o --no-postprocess
```

当前行为说明：
- 推理失败样本会写入预测文件（`failed=true`），并纳入失败率统计字段。

## 6. 批量生成报告

```bash
python utils/generate_reports.py
python utils/generate_reports.py --version v1
python utils/generate_reports.py --version v2 --no-postprocess
```

## 7. 可视化看板

```bash
streamlit run app.py
```

主要页面：
1. Leaderboard
2. Detailed View
3. Statistical Analysis
4. Export

## 8. 测试

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OCR_benchmark
python -m unittest discover -s tests -p 'test_*.py' -v
```

## 9. 指标概览

`v1`：
- CER、WER、NED（越低越好）
- Precision、Recall、BoW F1、Exact Match（越高越好）

`v2`：
- Weighted Score（越高越好）
- Y/N Accuracy（越高越好）
- Handwriting CER/WER/NED（越低越好）

## 10. 维护说明

- `requirements.txt` 已锁定版本，降低环境漂移。
- 统计模块默认固定随机种子，提升可复现性。
- 已移除弃用的 schema 评估链路，主线更聚焦、结论更清晰。

## 11. 最近更新

- Statistical Analysis 页面中的 Pairwise 表格已直接显示 `95% CI`。
- Pairwise 统计结果支持在页面直接下载 CSV。
- `app.py` 统计分析区域已拆分为 helper 函数，维护成本更低。
