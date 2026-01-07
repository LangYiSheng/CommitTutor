import re

def reverse_git_diff(diff_str):
    lines = diff_str.splitlines()
    reversed_lines = []

    # 用于缓存一个连续变动块中的行，以便重新排序
    hunk_cache = []

    def flush_hunk_cache():
        """将缓存的变动行按先减后加的顺序输出"""
        if not hunk_cache:
            return
        # 反转符号：原来的 - 变成 +，原来的 + 变成 -
        processed = []
        for line in hunk_cache:
            if line.startswith('-'):
                processed.append('+' + line[1:])
            elif line.startswith('+'):
                processed.append('-' + line[1:])

        # 重新排序：所有以 '-' 开头的排在前面，'+' 排在后面
        processed.sort(key=lambda l: 0 if l.startswith('-') else 1)
        reversed_lines.extend(processed)
        hunk_cache.clear()

    for line in lines:
        # 处理文件头：交换 --- 和 +++
        if line.startswith('--- '):
            flush_hunk_cache() # 切换文件前清空缓存
            reversed_lines.append('+++ ' + line[4:])
        elif line.startswith('+++ '):
            reversed_lines.append('--- ' + line[4:])

        # 处理 index 交换
        elif line.startswith('index '):
            parts = line.split()
            hashes = parts[1].split('..')
            if len(hashes) == 2:
                reversed_lines.append(f"index {hashes[1]}..{hashes[0]} {' '.join(parts[2:])}".strip())
            else:
                reversed_lines.append(line)

        # 处理区块头交换数字
        elif line.startswith('@@'):
            flush_hunk_cache()
            match = re.match(r'@@ -(\d+,?\d*) \+(\d+,?\d*) @@(.*)', line)
            if match:
                old_range, new_range, tail = match.groups()
                reversed_lines.append(f"@@ -{new_range} +{old_range} @@{tail}")
            else:
                reversed_lines.append(line)

        # 核心逻辑：处理内容变动行
        elif line.startswith('+') or line.startswith('-'):
            hunk_cache.append(line)
        else:
            # 遇到上下文行（空格开头）或 diff --git 行，先清空之前的变动缓存
            flush_hunk_cache()
            reversed_lines.append(line)

    # 最后一行处理完后清空剩余缓存
    flush_hunk_cache()

    return '\n'.join(reversed_lines)

def extract_diff_payload(diff_str):
    lines = diff_str.splitlines()
    payload_start_index = -1

    # 寻找第一个以 @@ 开头的行
    for i, line in enumerate(lines):
        if line.startswith('@@'):
            # 我们需要的是从 @@ 行的下一行开始的所有内容
            payload_start_index = i + 1
            break

    if payload_start_index != -1:
        # 提取剩余行并重新组合成字符串
        return '\n'.join(lines[payload_start_index:])

    return ""  # 如果没找到 @@，返回空字符串