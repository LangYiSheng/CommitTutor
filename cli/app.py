from advisor.llm_client import LLMAdvisor
from config.manager import config_exists, load_config, save_config
from detector import DetectorManager, list_detectors
from git_utils.repo import get_latest_commit
from workflow.decision import should_request_advice


def _prompt_initial_config():
    print("未检测到配置文件，需要进行初始配置。")
    base_url = input("请输入大语言模型 BASE_URL：\n> ").strip()
    api_key = input("\n请输入大语言模型 API_KEY：\n> ").strip()
    model_name = input("\n请输入大语言模型 MODEL_NAME：\n> ").strip()
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
        "detector_model": "",
    }


def _select_detector_model(current_name=""):
    detectors = list_detectors()
    if not detectors:
        print("未检测到可用的缺陷检测模型。")
        return current_name

    print("请选择缺陷检测模型：")
    for idx, name in enumerate(detectors, start=1):
        marker = " (当前)" if name == current_name else ""
        print(f"{idx}. {name}{marker}")

    while True:
        choice = input("> ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(detectors):
            return detectors[int(choice) - 1]
        print("输入无效，请重新选择。")


def _update_llm_config(config):
    while True:
        print("当前大语言模型配置：")
        print(f"1. BASE_URL: {config.base_url}")
        print(f"2. API_KEY: {config.api_key}")
        print(f"3. MODEL_NAME: {config.model_name}")
        print("4. 返回")
        choice = input("> ").strip()

        if choice == "1":
            config.base_url = input("请输入新的 BASE_URL：\n> ").strip()
        elif choice == "2":
            config.api_key = input("请输入新的 API_KEY：\n> ").strip()
        elif choice == "3":
            config.model_name = input("请输入新的 MODEL_NAME：\n> ").strip()
        elif choice == "4":
            print("")
            return
        else:
            print("输入无效，请重新选择。\n")
            continue

        save_config(config.to_dict())
        print("已更新大语言模型配置。\n")


def _update_detector_model(config, manager):
    detector_name = _select_detector_model(config.detector_model)
    if detector_name:
        config.detector_model = detector_name
        manager.set_current(detector_name)
        save_config(config.to_dict())
        print(f"已选择缺陷检测模型：{detector_name}\n")


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

    manager = DetectorManager()

    if not config.detector_model:
        _update_detector_model(config, manager)
    else:
        manager.set_current(config.detector_model)

    print("正在尝试获取本地仓库的最新一次提交...\n")
    try:
        commit = get_latest_commit()
    except RuntimeError as exc:
        print(f"无法获取提交信息：{exc}")
        return

    print("✔ 已获取最新提交")
    print("提交信息：")
    print(f"- Commit ID: {commit.commit_id}")
    print(f"- Author: {commit.author}")
    print(f"- Message: {commit.message}")
    print(f"- Time: {commit.timestamp}")
    print("")

    while True:
        print("请选择操作：")
        print(f"1. 使用 {config.detector_model} 模型进行缺陷可能性检测")
        print("2. 重新获取最新提交")
        print("3. 重新选择检测模型")
        print("4. 重新配置大模型")
        print("5. 退出")
        choice = input("> ").strip()

        if choice == "1":
            break
        if choice == "2":
            print("\n正在重新获取最新提交...\n")
            commit = get_latest_commit()
            print("✔ 已获取最新提交")
            print("提交信息：")
            print(f"- Commit ID: {commit.commit_id}")
            print(f"- Author: {commit.author}")
            print(f"- Message: {commit.message}")
            print(f"- Time: {commit.timestamp}")
            print("")
            continue
        if choice == "3":
            _update_detector_model(config, manager)
            continue
        if choice == "4":
            _update_llm_config(config)
            continue
        if choice == "5":
            print("已退出。")
            manager.shutdown()
            return
        print("输入无效，请重新选择。\n")

    print("\n正在分析本次代码提交...")
    print("[##########------] 60%\n")

    detector = manager.get_current()
    if not detector:
        print("未找到对应的缺陷检测模型，已退出。")
        manager.shutdown()
        return
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

    manager.shutdown()


__all__ = ["run"]
