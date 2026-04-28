# OCR Benchmark 框架

OCR Benchmark 是一个面向视觉语言模型的文档理解评测仓库。当前版本面向外部研究者，目标是提供一条清晰、可复现的路径，用于运行 OCR 评测、比较模型，并复现实验汇总结果。

## 1. 仓库状态

仓库当前围绕五个稳定模块组织：
- `runner`：`main.py` 中的评测执行入口
- `evaluators`：指标与任务评分逻辑
- `model adapters`：`models/` 中的模型适配层
- `dashboard`：Streamlit 分析与导出界面
- `reproduction utilities`：`utils/` 中的报告重建与 multi-run 汇总脚本

属于开源 benchmark 主路径的内容：
- benchmark runner 与模型适配器
- `v1` 和 `v2` 的评测代码
- 用于分析结果的 dashboard
- multi-run 实验的汇总脚本
- 示例 GT 文件与 split 定义

不属于 benchmark API 表面的内容：
- 需要单独下载的图片数据
- 被忽略目录中的历史实验缓存
- 论文草稿与本地研究笔记
- `labeling_v1/`、`labeling_v2/` 这类标注流程目录

## 2. 任务定义

当前仓库支持两个 benchmark 任务：

- `v1`：整段文本 OCR 转写质量
- `v2`：手写文本转写 + Y/N 选项抽取

单次运行和 multi-run 使用同一套 runner。Multi-run 本质上就是同一套 benchmark 流程配合 `--runs-per-image N`，其中 `N > 1`。

## 3. 仓库结构

主要入口如下：
- `main.py`：CLI benchmark runner
- `app.py`：Streamlit dashboard
- `models/`：`openai`、`gemini`、`qwen`、`ollama`、`dummy` 的模型适配器
- `evaluators/`：指标、评测器、统计检验
- `dashboard/`：leaderboard、detailed view、statistics、export、multi-run 页面
- `utils/`：报告重建、multi-run 汇总与辅助脚本
- `data/`：GT JSON 与可选 split 定义
- `tests/`：评测行为回归测试

工作区里可能仍然存在一些可选或非核心资产，但运行 benchmark 主流程并不依赖它们。

## 4. 安装与配置

```bash
pip install -r requirements.txt
cp env.example .env
```

按需填写 API Key：
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `DASHSCOPE_API_KEY`

可选复现参数：
- `OCR_BENCHMARK_SEED=42`

## 5. 数据准备

必需 GT 文件：
- `data/sample_gt_v1.json`，或兼容旧格式的 `data/sample_gt.json`
- `data/sample_gt_v2.json`

可选 split 文件：
- `data/dataset_split.json`

图片文件不随仓库分发。请将图片放到 `data/` 目录，并确保文件名与 GT JSON 中的 `file_name` 字段一致。

当前图片下载地址：
- Google Drive: [dataset images](https://drive.google.com/drive/folders/1yLLiAzUoAwD28IMYE2tZUibX_I-f5gXf?usp=drive_link)

## 6. Quickstart

最快的 smoke test 路径：

```bash
python main.py -v v1 -m dummy -id dummy-smoke
python main.py -v v2 -m dummy -id dummy-smoke-v2
```

会生成：
- `results/preds_v1_dummy-smoke.json`
- `results/report_v1_dummy-smoke.json`
- `results/preds_v2_dummy-smoke-v2.json`
- `results/report_v2_dummy-smoke-v2.json`

## 7. 运行 Benchmark

CLI 接口：

```bash
python main.py -v {v1|v2} -m {provider} -id {model_id...} [--resume] [--no-postprocess] [--split PATH] [--runs-per-image N]
```

主要输入参数：
- `-m`, `--model`：模型提供方，支持 `dummy`、`gemini`、`qwen`、`openai`、`ollama`
- `-id`, `--model_id`：一个或多个模型 ID
- `-v`, `--version`：`v1` 或 `v2`
- `--gt`：自定义 GT JSON 路径
- `--split`：可选 split JSON
- `--resume`：从已有预测文件断点续跑
- `--no-postprocess`：关闭 evaluator 后处理
- `--runs-per-image N`：每张图重复运行 `N` 次

示例：

```bash
# 单次运行 v1
python main.py -v v1 -m openai -id gpt-4.1-mini

# 单次运行 v2
python main.py -v v2 -m gemini -id gemini-2.0-flash-exp

# 关闭 evaluator 后处理的消融实验
python main.py -v v2 -m openai -id gpt-4o --no-postprocess

# 每张图跑 3 次的 multi-run 实验
python main.py -v v1 -m openai -id gpt-4.1-mini --runs-per-image 3
```

行为说明：
- 失败样本会以 `failed=true` 写入预测文件，并计入报告层面的失败统计
- 瞬时网络错误不会写入失败记录，后续可通过 `--resume` 自动补跑
- `--runs-per-image 1` 就是标准单次 benchmark
- 当 `--runs-per-image > 1` 时，第 1 次 run 仍会同步写入单次输出路径，以兼容 dashboard

## 8. 结果文件约定

标准 benchmark 输出：
- `results/preds_{version}_{model}.json`：dashboard 使用的主预测文件
- `results/report_{version}_{model}.json`：评测汇总报告

Multi-run 输出：
- `results/multirun/preds_{version}_{model}__run{n}.json`：每次 run 的原始预测
- `results/multirun/meta_{version}_{model}.json`：multi-run 元信息与运行环境快照

预计算的 multi-run 汇总工件：
- `results/multirun/per_run_{version}.json`
- `results/multirun/leaderboard_{version}.json`
- `results/multirun/leaderboard_std_{version}.json`
- `results/multirun/distribution_{version}.json`
- `results/multirun/summary_meta_{version}.json`

这些文件都是生成产物，通常不建议提交到 git。

## 9. 生成 Multi-run 汇总

为 dashboard 的 multi-run 页面生成预计算工件：

```bash
python utils/generate_multirun_summary.py --version all --write-csv
```

也可以按任务分别生成：

```bash
python utils/generate_multirun_summary.py --version v1 --write-csv
python utils/generate_multirun_summary.py --version v2 --write-csv
```

这个脚本会读取 `results/multirun/preds_*__runN.json`，并把规范化后的汇总文件写回 `results/multirun/`。

## 10. 重建报告

如果已经有预测文件、只想重新生成 report：

```bash
python utils/generate_reports.py
python utils/generate_reports.py --version v1
python utils/generate_reports.py --version v2 --no-postprocess
```

## 11. Dashboard

启动方式：

```bash
streamlit run app.py
```

Dashboard 的数据依赖：
- leaderboard、detailed view、statistics、export 页面读取 `results/preds_*` 与 `results/report_*`
- multi-run 页面读取 `results/multirun/` 下的预计算汇总工件
- 论文风格 figure 视图默认隐藏，只有在所需 multi-run 汇总工件已存在时才会显示

## 12. 指标概览

`v1` 指标：
- CER、WER、NED
- Precision、Recall、BoW F1、Exact Match

`v2` 指标：
- Weighted Score
- Y/N Accuracy
- Y-positive Precision、Recall、F1、Balanced Accuracy
- Handwriting CER、WER、NED

## 13. 测试

示例命令：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

发布前推荐 smoke test 顺序：
- 跑一次 `dummy` 的 `v1` 单次任务
- 跑一次 `dummy` 的 `v2` 单次任务
- 跑一次 `dummy` 的 `--runs-per-image 3` multi-run 任务
- 生成 multi-run summary
- 启动 Streamlit dashboard

## 14. 限制与说明

- 图片数据不包含在仓库内
- 仓库中可能存在本地忽略的论文目录或数据维护目录，但 benchmark 主流程不依赖它们
- `utils/` 中有些脚本用于复现实验或数据维护，而不是普通 benchmark 使用场景
- dashboard 默认依赖上文列出的结果文件命名规范
