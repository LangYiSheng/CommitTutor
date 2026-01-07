# CommitTutor

CommitTutor 是一个面向学生开发者的代码提交辅导工具，聚焦于对单次提交的
分析与教学式反馈。当前版本提供清晰可扩展的项目框架与基础流程，核心算法
与模型调用均为占位实现。

## 功能概览

- CLI 交互入口，提供教学向的引导流程
- 本地配置管理（LLM BASE_URL / API_KEY / MODEL_NAME / detector 模型）
- 最新一次提交的信息抽取与 diff 解析
- 缺陷检测模型接口与可扩展注册机制
- LLM 评价层接口与占位输出

## 目录结构

```
CommitTutor/
  advisor/             # 大模型调用层（占位实现）
  cli/                 # CLI 交互层
  config/              # 配置管理层
  detector/            # 缺陷检测模型层
  git_utils/           # Git 提交抽取与解析
  workflow/            # 决策逻辑
  main.py              # 入口
```

## 快速开始

1) 进入仓库并运行：

```
python main.py
```

2) 首次运行会引导创建配置文件。

## 开发说明

- 缺陷检测模型需要继承 `detector.model.DefectDetector`，实现 `load` / `analyze`
  并在 `analyze` 内调用 `_ensure_loaded()` 实现延迟加载。
- 模型接入指南详见 `detector/README.md`。

## 说明

本项目当前为工程框架搭建阶段，所有模型与业务逻辑均为占位实现。
