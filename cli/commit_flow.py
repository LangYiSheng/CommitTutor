from git_utils.repo import get_latest_commit


def fetch_latest_commit():
    print("正在尝试获取本地仓库的最新一次提交...\n")
    try:
        return get_latest_commit()
    except RuntimeError as exc:
        print(f"无法获取提交信息：{exc}")
        return None


def show_commit_info(commit):
    print("✔ 已获取最新提交")
    print("提交信息：")
    print(f"- Commit ID: {commit.commit_id}")
    print(f"- Author: {commit.author}")
    print(f"- Message: {commit.message}")
    print(f"- Time: {commit.timestamp}")
    print("")


__all__ = ["fetch_latest_commit", "show_commit_info"]
