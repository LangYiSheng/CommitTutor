from git_utils.models import CommitInfo


def get_latest_commit():
    # TODO: Replace with real git inspection.
    return CommitInfo(
        commit_id="xxxxxxx",
        author="student",
        message="Update lesson scaffold",
        timestamp="2024-01-01 12:00",
    )


__all__ = ["get_latest_commit"]
