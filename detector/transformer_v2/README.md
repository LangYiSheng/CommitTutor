# Transformer V2

这里是基于 Transformer V1 的优化版本，池化策略为“平均池化 + 注意力残差”，在均值表示上叠加注意力加权结果。

## 主要修改
- 池化方式：由 masked mean pooling 改为平均池化 + 注意力残差。
- 池化参数：标题与 diff 分别使用独立的注意力线性层生成权重，并与均值向量相加。

## 训练
在 `detector/transformer_v2/train` 目录下运行：
`python transformer_v2.py`

脚本将数据按 80/10/10（分层）切分，并导出：
- `model.pt`
- `vocab_stoi.json`
- `config.json`

## 推理
`detector/transformer_v2/__init__.py` 负责加载模型并对每个文件 diff 打分，最终取最大值作为 commit 风险。
默认只评估 `.java` 文件；可修改 `file_allowlist` 来调整覆盖范围。
