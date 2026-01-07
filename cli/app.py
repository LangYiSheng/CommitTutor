from advisor.llm_client import LLMAdvisor
from config.manager import config_exists, load_config, save_config
from detector.model import DefectDetector
from git_utils.repo import get_latest_commit
from workflow.decision import should_request_advice


def _prompt_initial_config():
    print("未检测到配置文件，需要进行初始配置。")
    base_url = input("请输入大语言模型 BASE_URL：\n> ").strip()
    api_key = input("\n请输入大语言模型 API_KEY：\n> ").strip()
    model_name = input("\n请输入大语言模型 MODEL_NAME：\n> ").strip()
    return {"base_url": base_url, "api_key": api_key, "model_name": model_name}


def _confirm(prompt):
    reply = input(f"{prompt} ").strip().lower()
    return reply in {"y", "yes"}


def run():
    print("CommitTutor v1.0")
    print("----------------------------------------")

    if not config_exists():
        config_data = _prompt_initial_config()
        config = save_config(config_data)
        print("\n配置已保存，初始化完成。")
        print("----------------------------------------")
    else:
        config = load_config()

    print("正在尝试获取本地仓库的最新一次提交...\n")
    commit = get_latest_commit()

    print("✔ 已获取最新提交")
    print("提交信息：")
    print(f"- Commit ID: {commit.commit_id}")
    print(f"- Author: {commit.author}")
    print(f"- Message: {commit.message}")
    print(f"- Time: {commit.timestamp}")
    print("")

    if not _confirm("是否以该次提交进行代码辅导分析？(Y/N)"):
        print("已退出。")
        return

    print("\n正在分析本次代码提交...")
    print("[##########------] 60%\n")

    detector = DefectDetector()
    score = detector.analyze(commit)

    print("分析完成，正在综合评价...\n")
    print(f"改进可能性评分：{score:.2f}\n")

    advisor = LLMAdvisor(config)
    if should_request_advice(score):
        feedback = advisor.generate_feedback(commit, score)
        print("综合评价：")
        print(feedback)
        print("\n建议你根据以上反馈对代码进行修改，")
        input("完成新的提交后，输入 F 继续辅导流程：\n> ")
    else:
        print("本次提交整体表现良好，未发现明显需要改进的地方。")
        if _confirm("是否仍希望查看进一步的优化建议？(Y/N)"):
            feedback = advisor.generate_feedback(commit, score)
            print("综合评价：")
            print(feedback)


__all__ = ["run"]
