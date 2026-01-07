class LLMAdvisor:
    def __init__(self, config):
        self.config = config

    def generate_feedback(self, commit_info, score, history=None):
        # TODO: Replace with real LLM call.
        if history:
            return (
                "（此处由大语言模型生成自然语言评价文本，\n"
                "包含上一轮优化建议作为上下文，用于持续跟进修复。）"
            )
        return (
            "（此处由大语言模型生成自然语言评价文本，\n"
            "从可读性、结构、潜在问题等角度进行说明，\n"
            "避免直接使用“错误”“缺陷”等词语）"
        )


__all__ = ["LLMAdvisor"]
