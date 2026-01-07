from advisor.llm_client import LLMAdvisor
from cli.commit_flow import fetch_latest_commit, show_commit_info
from cli.config_flow import update_detector_model, update_llm_config
from cli.prompts import confirm, pause


def _should_request_advice(score, threshold=0.6):
    return score >= threshold


def _select_action(config):
    print("请选择操作：")
    print(f"1. 使用 {config.detector_model} 模型进行缺陷可能性检测")
    print("2. 重新获取最新提交")
    print("3. 重新选择检测模型")
    print("4. 重新配置大模型")
    print("5. 退出")
    return input("> ").strip()


def run_analysis_loop(config, manager):
    advisor = LLMAdvisor(config)
    feedback_history = []
    continuing_issue = False

    while True:
        commit = fetch_latest_commit()
        if commit is None:
            return

        show_commit_info(commit)

        while True:
            choice = _select_action(config)
            if choice == "1":
                break
            if choice == "2":
                commit = fetch_latest_commit()
                if commit is None:
                    return
                show_commit_info(commit)
                continue
            if choice == "3":
                update_detector_model(config, manager)
                continue
            if choice == "4":
                update_llm_config(config)
                continue
            if choice == "5":
                print("已退出。")
                manager.shutdown()
                return
            print("输入无效，请重新选择。\n")

        detector = manager.get_current()
        if not detector:
            print("未找到对应的缺陷检测模型，已退出。")
            manager.shutdown()
            return

        print("\n正在分析本次代码提交...")

        score = detector.analyze(commit)

        print("分析完成，正在综合评价...\n")
        print(f"改进可能性评分：{score:.2f}\n")

        if not _should_request_advice(score):
            continuing_issue = False
            feedback_history = []
            print("本次提交整体表现良好，未发现明显需要改进的地方。")
            if confirm("是否仍希望查看进一步的优化建议？(Y/N)"):
                feedback = advisor.generate_feedback(commit, score)
                print("综合评价：")
                print(feedback)
            pause("输入回车继续获取最新提交")
            continue

        history_to_send = feedback_history if continuing_issue else []
        feedback = advisor.generate_feedback(commit, score, history=history_to_send)
        print("综合评价：")
        print(feedback)

        if confirm("是否忽视本次风险并继续？(Y/N)"):
            continuing_issue = False
            feedback_history = []
            pause("输入回车继续获取最新提交")
            continue

        pause("完成新的提交后，输入 F 继续辅导流程：")
        continuing_issue = True
        feedback_history.append(feedback)
        feedback_history = feedback_history[-3:]


__all__ = ["run_analysis_loop"]
