# 缺陷检测模型接入指南

本目录用于存放 CommitTutor 的缺陷检测模型与注册表。每个模型建议独立成子文件夹，以便携带权重文件或其他资源，避免顶层包变得杂乱。

## 快速开始

1. 新建一个模型文件夹，例如 `detector/transformer/`。
2. 在该文件夹内实现一个继承 `detector.model.DefectDetector` 的检测器类，并使用唯一名称注册。
3. 在检测器内部实现延迟加载逻辑，确保只有在 `analyze` 执行时才加载权重。

## 目录结构示例

```
detector/
  transformer/
    __init__.py
    weights.bin
    tokenizer.json
```

## 最小实现示例

```python
# detector/transformer/__init__.py
from detector.model import DefectDetector
from detector.registry import register_detector


@register_detector("Transformer")
class TransformerDefectDetector(DefectDetector):
    def load(self):
        # TODO: Load model weights/resources.
        self._loaded = True

    def analyze(self, commit_info):
        # TODO: Replace with real scoring logic.
        self._ensure_loaded()
        return 0.5
```

## 测试

开发阶段可以使用 `detector/test_detector.py` 做简单的对比测试。新增模型后，
将它们加入该脚本即可。

## 管理器

`detector/manager.py` 提供一个简单的实例管理器，支持：
- 按名称缓存检测器实例，避免重复加载。
- 切换模型时自动调用旧实例的 `unload()`。
- 程序结束时调用 `shutdown()` 释放所有资源。

## 训练数据与 CommitData 使用建议

`CommitData` 的字段较多，训练时应根据任务目标自行拼接与提取特征，建议：

- 组合提交级统计特征（如 `files_changed`、`total_lines_added` 等）作为数值特征。
- 从 `files` 中筛选目标语言或类型（如仅 `java`/`xml`），避免噪声。
- 将 `diff_text` 作为文本特征时，建议自行清洗并按文件或提交聚合。
- `FileDiff` 的路径与 diff 适合构造“结构化 + 文本”混合特征，方便模型学习变更模式。
