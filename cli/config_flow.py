from config.manager import save_config
from detector import list_detectors


def prompt_initial_config():
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


def select_detector_model(current_name=""):
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


def update_llm_config(config):
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


def update_detector_model(config, manager):
    detector_name = select_detector_model(config.detector_model)
    if detector_name:
        config.detector_model = detector_name
        manager.set_current(detector_name)
        save_config(config.to_dict())
        print(f"已选择缺陷检测模型：{detector_name}\n")


__all__ = [
    "prompt_initial_config",
    "select_detector_model",
    "update_llm_config",
    "update_detector_model",
]
