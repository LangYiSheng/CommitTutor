from cli.analysis_flow import run_analysis_loop
from cli.config_flow import prompt_initial_config, update_detector_model
from config.manager import config_exists, load_config, save_config
from detector import DetectorManager


def run():
    print(r"""
________  ________  _____ ______   _____ ______   ___  _________        _________  ___  ___  _________  ________  ________     
|\   ____\|\   __  \|\   _ \  _   \|\   _ \  _   \|\  \|\___   ___\     |\___   ___\\  \|\  \|\___   ___\\   __  \|\   __  \    
\ \  \___|\ \  \|\  \ \  \\\__\ \  \ \  \\\__\ \  \ \  \|___ \  \_|     \|___ \  \_\ \  \\\  \|___ \  \_\ \  \|\  \ \  \|\  \   
 \ \  \    \ \  \\\  \ \  \\|__| \  \ \  \\|__| \  \ \  \   \ \  \           \ \  \ \ \  \\\  \   \ \  \ \ \  \\\  \ \   _  _\  
  \ \  \____\ \  \\\  \ \  \    \ \  \ \  \    \ \  \ \  \   \ \  \           \ \  \ \ \  \\\  \   \ \  \ \ \  \\\  \ \  \\  \| 
   \ \_______\ \_______\ \__\    \ \__\ \__\    \ \__\ \__\   \ \__\           \ \__\ \ \_______\   \ \__\ \ \_______\ \__\\ _\ 
    \|_______|\|_______|\|__|     \|__|\|__|     \|__|\|__|    \|__|            \|__|  \|_______|    \|__|  \|_______|\|__|\|__|
                                                                                                                                
                                                       CommitTutor v1.0                                                     """)
    # print("CommitTutor v1.0")
    print("----------------------------------------")

    if not config_exists():
        config_data = prompt_initial_config()
        config = save_config(config_data)
        print("\n配置已保存，初始化完成。")
        print("----------------------------------------")
    else:
        config = load_config()

    manager = DetectorManager()

    if not config.detector_model:
        update_detector_model(config, manager)
    else:
        manager.set_current(config.detector_model)

    run_analysis_loop(config, manager)
    manager.shutdown()


__all__ = ["run"]
