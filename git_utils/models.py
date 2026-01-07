from dataclasses import dataclass
from typing import List


@dataclass
class FileDiff:
    file_path: str
    diff_text: str
    lines_added: int
    lines_deleted: int
    file_type: str


@dataclass
class CommitData:
    commit_id: str
    author: str
    message: str
    timestamp: int
    files: List[FileDiff]
    files_changed: int
    total_lines_added: int
    total_lines_deleted: int
    total_lines_changed: int


__all__ = ["FileDiff", "CommitData"]
