import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".committutor"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class UserConfig:
    base_url: str
    api_key: str

    @classmethod
    def from_dict(cls, data):
        return cls(base_url=data.get("base_url", ""), api_key=data.get("api_key", ""))

    def to_dict(self):
        return {"base_url": self.base_url, "api_key": self.api_key}


def config_exists(path=CONFIG_FILE):
    return path.exists()


def load_config(path=CONFIG_FILE):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return UserConfig.from_dict(data)


def save_config(config_data, path=CONFIG_FILE):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = UserConfig.from_dict(config_data)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=True, indent=2)
    return config


__all__ = ["UserConfig", "config_exists", "load_config", "save_config"]
