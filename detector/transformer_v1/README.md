# Transformer SD

这里实现了基于 Transformer 的代码缺陷预测模型，训练数据来自 PR 标题和 diff 内容。

## 模型设计
- 分词与词表：使用 torchtext 的 basic_english；词表由 PR_TITLE 与 DIFF_CODE 共同构建。
- 输入：PR 标题 + diff payload（通过 extract_diff_payload 提取 hunk 内容）。
- 编码器：共享 embedding + 正弦位置编码，标题与 diff 各自使用独立的 TransformerEncoder 堆叠。
- 池化：对非 padding 的 token 做 masked mean pooling。
- 分类器：拼接标题/差异向量，MLP 输出缺陷概率（推理时做 sigmoid）。

## 训练
在 `detector/transformer_sd/train` 目录下运行：
`python transformer_sd.py`

脚本将数据按 80/10/10（分层）切分，并导出：
- `model.pt`
- `vocab_stoi.json`
- `config.json`

## 推理
`detector/transformer_sd/__init__.py` 负责加载模型并对每个文件 diff 打分，最终取最大值作为 commit 风险。
默认只评估 `.java` 文件；可修改 `file_allowlist` 来调整覆盖范围。
