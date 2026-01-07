from git_utils.models import CommitData, FileDiff


def get_latest_commit():
    # TODO: Replace with real git inspection.
    files = [
        FileDiff(
            file_path="src/Main.java",
            diff_text="--- a/src/Main.java\n+++ b/src/Main.java\n@@\n+// TODO: sample diff\n",
            lines_added=1,
            lines_deleted=0,
            file_type="java",
        )
    ]
    return CommitData(
        commit_id="xxxxxxx",
        author="student",
        message="Update lesson scaffold",
        timestamp=1704091200,
        files=files,
        files_changed=len(files),
        total_lines_added=sum(item.lines_added for item in files),
        total_lines_deleted=sum(item.lines_deleted for item in files),
        total_lines_changed=sum(
            item.lines_added + item.lines_deleted for item in files
        ),
    )


__all__ = ["get_latest_commit"]
