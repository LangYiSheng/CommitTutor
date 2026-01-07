class LLMAdvisor:
    def generate_feedback(self, commit_info, score):
        # TODO: Replace with real LLM call.
        return (
            "（此处由大语言模型生成自然语言评价文本，\n"
            "从可读性、结构、潜在问题等角度进行说明，\n"
            "避免直接使用“错误”“缺陷”等词语）"
        )


__all__ = ["LLMAdvisor"]
