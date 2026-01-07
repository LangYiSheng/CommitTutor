from dataclasses import dataclass


@dataclass
class CommitInfo:
    commit_id: str
    author: str
    message: str
    timestamp: str


__all__ = ["CommitInfo"]
