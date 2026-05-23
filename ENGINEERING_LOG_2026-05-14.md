# 工程日志：2026-05-14

## 目标

今天主要围绕 RAG 幻觉防控与回答准确性验证做工程化增强。核心目标是让系统从“检索到相关内容就回答”，升级为“检索、相关性判断、证据覆盖校验、证据约束生成、准确性测试”完整闭环。

## 修改范围

- `figure_agent/rag/paper_retriever.py`
- `figure_agent/agent/research_agent.py`
- `figure_agent/rag/accuracy_tester.py`
- `tests/test_rag_accuracy.py`
- `tests/rag_accuracy_cases.json`

## 主要改动

### 1. RAG 幻觉防控主流程

在 `PaperRAGRetriever` 中增强了 RAG 检索链路：

1. 先进行意图识别。
2. 根据意图决定检索范围、召回策略和证据阈值。
3. 先定位候选文档。
4. 执行混合召回。
5. 融合排序后进入 reranker 和相关性判断。
6. 对证据是否足够支撑回答做门控。
7. 根据结果返回 `PASS / RETRY / REFUSE`。

状态含义：

- `PASS`：证据充分，可以回答。
- `RETRY`：证据部分缺失，使用缺失需求点触发补充检索。
- `REFUSE`：多次检索后仍不足，明确拒答。

### 2. BM25 + 向量混合检索

新增了两路召回：

- BM25：负责关键词精确匹配。
- 向量检索：负责语义相似召回。

随后使用 RRF 融合排序：

```python
scores[cid] += 1 / (k + rank)
```

融合后的候选继续进入 reranker、相关性门控和证据覆盖校验。

### 3. Chunk 相关性评估

对每个候选 chunk 增加三信号综合判断：

- `embedding_score`
- `rerank_score`
- `keyword_score`

综合分数：

```python
combined_score =
    embedding_score * 0.38
    + rerank_score * 0.42
    + keyword_score * 0.20
```

只有通过 `Relevance Gate` 的 chunk 才能进入后续证据判断和回答上下文。

### 4. 证据覆盖校验

新增需求点拆解和覆盖检查：

- `_extract_requirements()`
- `_coverage_check()`
- `_requirement_support_score()`

流程是先把用户问题拆成多个需求点，再逐项判断是否有证据支持。覆盖不完整时触发补充检索；补充检索后仍不完整则 `REFUSE`。

示例：

```python
{
    "Kafka 消息堆积处理": True,
    "消息可靠性保证": False,
    "消息不丢": False,
}
```

### 5. 证据约束生成

新增证据约束生成规则，要求大模型只能基于给定证据回答：

- 不能使用模型参数知识自由补充。
- 证据中没有明确提到的内容必须说“资料中未提到”。
- 每个关键结论必须标注 `[来源: 证据N]`。
- 找不到证据支撑的内容不能写成结论。

`research_agent.py` 中也接入了 `REFUSE` 逻辑：证据不足时直接拒答，不再把问题交给 LLM 硬答。

### 6. 回答准确性测试模块

新增 `figure_agent/rag/accuracy_tester.py`，用于专门测试 RAG 回答准确性。

它会检查：

- 事实性结论是否都有 `[来源: 证据N]`。
- 引用的证据编号是否存在。
- 结论是否能被对应证据支持。
- “资料中未提到 / 证据不足 / 不确定”等拒答声明允许不带来源。

返回结果包含：

- `PASS / FAIL`
- 总结论数
- 通过结论数
- 无来源结论
- 不存在来源
- 证据不支持的结论

### 7. 准确性测试用例数据集

新增 `tests/rag_accuracy_cases.json`。

共 20 条用例：

- CNN：5 条
- LSTM：5 条
- RNN：5 条
- Transformer：5 条

每条格式为：

```json
{
  "question": "...",
  "gold_evidence": ["..."],
  "gold_claims": ["..."],
  "should_refuse": false
}
```

`tests/test_rag_accuracy.py` 会自动加载该 JSON，构造证据上下文和带来源答案，并调用 `RAGAccuracyTester` 批量验证。

## 验证结果

已执行并通过：

```powershell
python tests\test_rag_accuracy.py
python -m json.tool tests\rag_accuracy_cases.json
python -m py_compile figure_agent\rag\accuracy_tester.py tests\test_rag_accuracy.py
python -m py_compile figure_agent\rag\paper_retriever.py figure_agent\agent\research_agent.py
python backend\manage.py check
```

关键结果：

```text
RAG accuracy tests passed.
System check identified no issues (0 silenced).
```

## 当前效果

现在系统的 RAG 回答链路变为：

```text
用户问题
  -> 意图识别
  -> BM25 召回
  -> 向量召回
  -> RRF 融合排序
  -> reranker 重排
  -> chunk 相关性判断
  -> 需求点覆盖校验
  -> PASS / RETRY / REFUSE
  -> 证据约束生成
  -> 回答准确性测试
```

这使系统能够在证据不足时停止硬答，在证据充分时要求答案逐条引用证据，并提供可重复运行的准确性测试数据集。
