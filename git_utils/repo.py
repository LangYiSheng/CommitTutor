import subprocess

from git_utils.models import CommitData, FileDiff


def _run_git(args, repo_path=None):
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=repo_path,
    )
    return result.stdout or ""


def _ensure_git_available(repo_path=None):
    try:
        _run_git(["--version"], repo_path=repo_path)
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 git 命令，请先安装并配置 git。") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("无法执行 git 命令，请检查安装或环境配置。") from exc


def _ensure_git_repo(repo_path=None):
    try:
        output = _run_git(
            ["rev-parse", "--is-inside-work-tree"], repo_path=repo_path
        ).strip().lower()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("当前目录未找到 git 仓库。") from exc

    if output != "true":
        raise RuntimeError("当前目录未找到 git 仓库。")

    try:
        _run_git(["rev-parse", "HEAD"], repo_path=repo_path)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("当前 git 仓库暂无提交记录。") from exc


def _normalize_renamed_path(path):
    if "=>" not in path:
        return path
    if "{" in path and "}" in path:
        prefix, rest = path.split("{", 1)
        rename_part, suffix = rest.split("}", 1)
        old_part, new_part = rename_part.split("=>", 1)
        return f"{prefix}{new_part.strip()}{suffix}".strip()
    return path.split("=>", 1)[1].strip()


def _file_type_for_path(path):
    lowered = path.lower()
    if lowered.endswith(".java"):
        return "java"
    if lowered.endswith(".xml"):
        return "xml"
    if lowered.endswith(".gradle") or lowered.endswith(".gradle.kts"):
        return "gradle"
    return "other"


def _parse_numstat(output):
    entries = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_raw, deleted_raw, path = parts[0], parts[1], parts[2]
        added = int(added_raw) if added_raw.isdigit() else 0
        deleted = int(deleted_raw) if deleted_raw.isdigit() else 0
        entries.append((added, deleted, _normalize_renamed_path(path)))
    return entries


def _parse_diffs(output):
    diffs = {}
    current_path = None
    current_lines = []

    def flush():
        if current_path is None:
            return
        diffs[current_path] = "\n".join(current_lines).strip("\n")

    for line in output.splitlines():
        if line.startswith("diff --git "):
            flush()
            current_lines = [line]
            parts = line.split()
            if len(parts) >= 4:
                bpath = parts[3].lstrip("b/").strip("\"")
                current_path = _normalize_renamed_path(bpath)
            else:
                current_path = None
            continue
        if current_path is not None:
            current_lines.append(line)

    flush()
    return diffs


def get_latest_commit(repo_path=None):
    _ensure_git_available(repo_path=repo_path)
    _ensure_git_repo(repo_path=repo_path)

    commit_id = _run_git(["log", "-1", "--format=%H"], repo_path=repo_path).strip()
    author = _run_git(["log", "-1", "--format=%an"], repo_path=repo_path).strip()
    message = _run_git(["log", "-1", "--format=%s"], repo_path=repo_path).strip()
    timestamp_raw = _run_git(
        ["log", "-1", "--format=%ct"], repo_path=repo_path
    ).strip()
    timestamp = int(timestamp_raw) if timestamp_raw.isdigit() else 0

    numstat_output = _run_git(
        ["show", "--numstat", "--format=", "HEAD"], repo_path=repo_path
    )
    diff_output = _run_git(
        ["show", "-U0", "--no-color", "--format=", "HEAD"], repo_path=repo_path
    )

    diff_map = _parse_diffs(diff_output)
    files = []
    for added, deleted, path in _parse_numstat(numstat_output):
        files.append(
            FileDiff(
                file_path=path,
                diff_text=diff_map.get(path, ""),
                lines_added=added,
                lines_deleted=deleted,
                file_type=_file_type_for_path(path),
            )
        )

    return CommitData(
        commit_id=commit_id,
        author=author,
        message=message,
        timestamp=timestamp,
        files=files,
        files_changed=len(files),
        total_lines_added=sum(item.lines_added for item in files),
        total_lines_deleted=sum(item.lines_deleted for item in files),
        total_lines_changed=sum(
            item.lines_added + item.lines_deleted for item in files
        ),
    )


__all__ = ["get_latest_commit"]
