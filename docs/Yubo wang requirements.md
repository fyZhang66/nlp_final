# Yubo Wang — 需求文档

## Pipeline 集成 + Cross-Domain 实验 + Error Analysis

> **项目：** End-to-End Aspect-Based Sentiment Analysis for Reviews
> **负责人：** Yubo Wang
> **上游依赖：** Gousu Ding (ATE Pipeline) · Fangyuan Zhang (ASC Pipeline) · Yunzhu Chen (DeBERTa 替换)

---

## 一、总体职责概述

你负责三件事：将 ATE 和 ASC 两个独立模块对接为完整的端到端 pipeline；在此 pipeline 上跑跨域实验；对全链路结果做系统性的错误分析。你的工作是整个项目"从能跑通到能看懂结果"的关键环节。

---

## 二、任务 1：Pipeline 集成（ATE → ASC）

### 2.1 需要从上游接收的内容

| 来源 | 交付物 | 格式要求 |
|------|--------|---------|
| Gousu (ATE) | ATE 模型的预测输出 | 每条句子对应一个 aspect span 列表，包含起止 token index 和抽取出的文本 |
| Fangyuan (ASC) | ASC 模型的推理接口/函数 | 接受 `(sentence, aspect_term)` 对作为输入，输出三分类标签 |
| Yunzhu (DeBERTa) | DeBERTa 版本的 ATE 与 ASC | 接口格式应与 BERT 版本完全一致，仅模型权重不同 |

### 2.2 ATE → ASC 中间数据格式

ATE 的输出需要被转换为 ASC 可消费的输入。中间数据应统一为以下 JSON Lines 格式（每行一个 JSON 对象）：

```
文件名: ate_predictions_{domain}_{model}.jsonl

每行格式:
{
  "sentence_id": "rest_test_0042",
  "sentence": "The pasta was delicious but the service was terrible.",
  "predicted_aspects": [
    {
      "term": "pasta",
      "start_token": 1,
      "end_token": 1,
      "confidence": 0.97
    },
    {
      "term": "service",
      "start_token": 7,
      "end_token": 7,
      "confidence": 0.91
    }
  ]
}
```

### 2.3 端到端 Pipeline 输出格式

Pipeline 最终输出应为完整的 `(aspect, sentiment)` 对列表，同时保留 gold label 以便后续评估：

```
文件名: e2e_predictions_{domain}_{model}.jsonl

每行格式:
{
  "sentence_id": "rest_test_0042",
  "sentence": "The pasta was delicious but the service was terrible.",
  "gold_aspects": [
    {"term": "pasta", "sentiment": "positive"},
    {"term": "service", "sentiment": "negative"}
  ],
  "predicted_aspects": [
    {"term": "pasta", "sentiment": "positive", "ate_confidence": 0.97},
    {"term": "service", "sentiment": "negative", "ate_confidence": 0.91}
  ]
}
```

### 2.4 集成阶段需产出的结果表

**表 A：端到端 Pipeline 结果总表**

| Model | Domain | ATE F1 | ASC Acc (gold) | ASC Acc (pred) | ASC Macro-F1 (gold) | ASC Macro-F1 (pred) |
|-------|--------|--------|----------------|----------------|----------------------|---------------------|
| BERT  | Restaurant | — | — | — | — | — |
| BERT  | Laptop     | — | — | — | — | — |
| DeBERTa | Restaurant | — | — | — | — | — |
| DeBERTa | Laptop     | — | — | — | — | — |

其中：
- **ASC Acc (gold)**：使用标注的 gold aspect 作为 ASC 输入时的准确率（上界）
- **ASC Acc (pred)**：使用 ATE 预测的 aspect 作为 ASC 输入时的准确率（真实性能）
- 两者之差即为 **error propagation gap**，是本项目核心观察指标之一

---

## 三、任务 2：Cross-Domain 实验

### 3.1 实验矩阵设计

需要覆盖以下所有训练→测试组合：

| 实验编号 | Train Domain | Test Domain | Model | 实验类型 |
|---------|-------------|-------------|-------|---------|
| 1 | Restaurant | Restaurant | BERT | In-domain baseline |
| 2 | Restaurant | Restaurant | DeBERTa | In-domain baseline |
| 3 | Restaurant | Laptop | BERT | Cross-domain |
| 4 | Restaurant | Laptop | DeBERTa | Cross-domain |
| 5 | Laptop | Restaurant | BERT | Cross-domain (反向) |
| 6 | Laptop | Restaurant | DeBERTa | Cross-domain (反向) |
| 7 | Laptop | Laptop | BERT | In-domain baseline |
| 8 | Laptop | Laptop | DeBERTa | In-domain baseline |

> 注：所有 8 组实验都需要在完整的端到端 pipeline 上运行（ATE + ASC），不能只跑单阶段。

### 3.2 Cross-Domain 结果输出格式

**表 B：Cross-Domain ATE 对比**

| Train → Test | BERT F1 | DeBERTa F1 | Δ (vs in-domain) |
|-------------|---------|-----------|-------------------|
| Rest → Rest | — | — | baseline |
| Rest → Laptop | — | — | — |
| Laptop → Laptop | — | — | baseline |
| Laptop → Rest | — | — | — |

**表 C：Cross-Domain ASC 对比（使用 predicted aspects）**

同上格式，指标换为 Accuracy 和 Macro-F1。

### 3.3 Cross-Domain 需额外记录的信息

对于每组跨域实验，需要记录以下定性观察（写入最终分析报告）：

- ATE 在目标域中常见的**漏提取** aspect 类型（如 Restaurant 模型在 Laptop 域漏掉 "battery life"）
- ATE **误提取**的 token（如把非 aspect 词识别为 aspect）
- ASC 在跨域场景下对**多义词**的情感判断变化（如 "hot" 在食物 vs. 电脑领域的语义差异）

---

## 四、任务 3：Confusion Matrix + Error Analysis

### 4.1 需产出的 Confusion Matrix

**矩阵 1：ASC 三分类混淆矩阵**

针对每组实验（共 8 组），各产出一个 3×3 混淆矩阵：

```
              Predicted
              Pos    Neu    Neg
Actual  Pos [  TP_p   ...   ... ]
        Neu [  ...    ...   ... ]
        Neg [  ...    ...   TP_n ]
```

需同时提供：
- 绝对数值版本
- 归一化版本（按行归一化，即每个 actual class 的分布）

重点关注的 cell：**Neutral vs Negative 的混淆**（PPT 中明确提到这一对是已知难点，且 Neutral 样本量仅占 ~20%，容易被模型忽略）。

**矩阵 2：ATE 的 Span-level 错误分类**

不是传统的混淆矩阵，而是一个错误类型统计表：

| 错误类型 | 定义 | 数量 | 占比 | 示例 |
|---------|------|------|------|------|
| Missing | Gold 有但 Pred 没有的 aspect | — | — | gold: "clam chowder", pred: 未识别 |
| Spurious | Pred 有但 Gold 没有的 aspect | — | — | pred: "the", gold: 不是 aspect |
| Boundary Error | 部分重叠但 span 不完全匹配 | — | — | gold: "clam chowder", pred: "chowder" |
| Correct | 完全匹配 | — | — | — |

### 4.2 端到端 Error Tracing

这是本项目最核心的分析产出。需要对每条端到端预测错误进行归因，判断错误来源于哪个阶段：

**Error Tracing 分类表：**

| 错误归因类别 | 定义 | 统计字段 |
|-------------|------|---------|
| ATE Miss → Sentiment Lost | ATE 未提取到该 aspect，导致无法进行情感分类 | count, % of total errors |
| ATE Boundary Error → Sentiment Wrong | ATE 提取了部分正确的 span，ASC 基于错误 span 给出了错误情感 | count, % |
| ATE Correct → ASC Wrong | ATE 正确提取了 aspect，但 ASC 给出了错误的情感标签 | count, % |
| ATE Spurious → False Positive | ATE 多提取了一个不存在的 aspect，ASC 为其分配了情感 | count, % |

### 4.3 Error Case 示例集

需从测试集中挑选有代表性的错误样例，按以下格式整理：

```
文件名: error_examples_{domain}_{model}.jsonl

每行格式:
{
  "sentence_id": "rest_test_0073",
  "sentence": "The restaurant was not bad but nothing special either.",
  "gold": [{"term": "restaurant", "sentiment": "neutral"}],
  "predicted": [{"term": "restaurant", "sentiment": "positive"}],
  "error_type": "ATE Correct → ASC Wrong",
  "analysis_note": "否定句式 'not bad' 被模型误判为 positive，模型未能捕捉 hedging/litotes 修辞"
}
```

每种 error type 至少提供 **3 个**有代表性的示例，涵盖以下现象（如果在数据中观察到的话）：
- 否定 / 双重否定 / litotes（"not bad"）
- 多义词跨域语义差异（"hot"）
- 多词 aspect 的边界识别困难（"clam chowder"）
- 隐式情感（未使用显式情感词）
- Neutral 样本被误判为 Pos/Neg（类别不均衡问题的体现）

### 4.4 汇总分析报告结构

最终 Error Analysis 部分应包含以下内容（作为 Final Report 的一个章节交付）：

1. **Error Propagation 量化：** gold aspect vs predicted aspect 下 ASC 性能差距的具体数值，配合表 A
2. **错误归因饼图/柱状图数据：** 四类端到端错误的占比分布
3. **Cross-domain 退化分析：** 哪种错误类型在跨域场景下增幅最大
4. **Per-class 分析：** Positive / Neutral / Negative 三类的 per-class precision, recall, F1，找出最弱的类别
5. **结论性发现：** 回答项目核心问题——"对于端到端 ABSA，提升 ATE 还是 ASC 更有效？"

---

## 五、交付清单

| 序号 | 交付物 | 格式 | 状态 |
|------|--------|------|------|
| 1 | Pipeline 集成代码 | Python 模块，可从命令行调用 | ☐ |
| 2 | 中间数据文件 `ate_predictions_*.jsonl` | JSON Lines | ☐ |
| 3 | 端到端输出 `e2e_predictions_*.jsonl` | JSON Lines | ☐ |
| 4 | 表 A：Pipeline 结果总表 | Markdown / LaTeX 表格 | ☐ |
| 5 | 表 B & C：Cross-domain 对比表 | Markdown / LaTeX 表格 | ☐ |
| 6 | 8 组 ASC 混淆矩阵（绝对值 + 归一化） | 图片 (matplotlib) + 数值 CSV | ☐ |
| 7 | ATE Span-level 错误分类统计 | 表格 | ☐ |
| 8 | 端到端 Error Tracing 归因统计 | 表格 + 饼图/柱状图 | ☐ |
| 9 | Error Case 示例集 `error_examples_*.jsonl` | JSON Lines | ☐ |
| 10 | Error Analysis 章节文稿（用于 Final Report） | Markdown / Google Docs | ☐ |

---

## 六、与上游的接口约定

为了确保集成顺畅，需要与队友确认以下接口细节：

1. **ATE 输出格式一致性：** Gousu 和 Yunzhu 需确保 BERT 和 DeBERTa 的 ATE 输出均符合 2.2 节的 JSON Lines 格式
2. **ASC 推理函数签名：** Fangyuan 和 Yunzhu 需提供形如 `predict(sentence: str, aspect_term: str) → str` 的统一接口（返回 "positive" / "neutral" / "negative"），或提供 batch 版本
3. **数据分割一致：** 所有人使用相同的 SemEval 2014 train/test split，不做额外的 random split
4. **Tokenizer 对齐：** ATE 输出的 token index 需与原始句子的 word-level 位置对应，若 ATE 使用 subword tokenizer，需在输出阶段还原为 word-level span