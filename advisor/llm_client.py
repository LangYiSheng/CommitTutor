from typing import List, Optional

from openai import OpenAI


class LLMAdvisor:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def generate_feedback(self, commit_info, score, history: Optional[List[str]] = None):
        history = history or []
        messages = [
            {
                "role": "system",
                "content": (
                    "你是教学向代码提交辅导助手。请避免使用“错误”或“缺陷”等词语，"
                    "用友好、启发式的语气给出简单的改进建议。"
                    "输出的内容应当使用中文纯文本，允许使用序号和换行，不能使用markdown格式，并且尽量简洁，控制在200字左右。"
                ),
            },
        ]

        if history:
            for item in history[-3:]:
                messages.append({"role": "assistant", "content": item})

        messages.append(
            {
                "role": "user",
                "content": self._build_prompt(commit_info, score),
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.6,
            )
        except Exception as e:
            print(f"LLM request failed: {e}")
            return "暂时无法获取大语言模型的反馈，请稍后再试。"

        return response.choices[0].message.content.strip()

    def _build_prompt(self, commit_info, score):
        summary_lines = [
            f"Commit ID: {commit_info.commit_id}",
            f"Author: {commit_info.author}",
            f"Message: {commit_info.message}",
            f"Timestamp: {commit_info.timestamp}",
            f"可能为缺陷提交的概率: {score:.2f}",
            f"Files changed: {commit_info.files_changed}",
            f"Lines added: {commit_info.total_lines_added}",
            f"Lines deleted: {commit_info.total_lines_deleted}",
            f"Lines changed: {commit_info.total_lines_changed}",
        ]

        file_sections = []
        for file_diff in commit_info.files[:3]:
            diff_text = file_diff.diff_text.strip()
            if len(diff_text) > 2000:
                diff_text = diff_text[:2000] + "\n... (truncated)"
            file_sections.append(
                "\n".join(
                    [
                        f"File: {file_diff.file_path}",
                        f"Type: {file_diff.file_type}",
                        f"Added: {file_diff.lines_added}, Deleted: {file_diff.lines_deleted}",
                        "Diff:",
                        diff_text or "(no diff text)",
                    ]
                )
            )

        prompt_parts = [
            "以下是一次代码提交的摘要与局部 diff，请给出教学向的改进建议。",
            "\n".join(summary_lines),
            "\n\n".join(file_sections) if file_sections else "No file diffs available.",
        ]

        return "\n\n".join(prompt_parts)


__all__ = ["LLMAdvisor"]
