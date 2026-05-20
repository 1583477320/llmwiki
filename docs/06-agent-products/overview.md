# Agent 产品总览

> 主流 AI Agent 产品的架构对比与选型指南

## 产品矩阵

| 产品 | 类型 | 模型后端 | 开源 | 核心场景 |
|------|------|---------|------|---------|
| Claude Code | CLI 编程 Agent | Claude 专属 | 否 | 终端结对编程 |
| Codex CLI | CLI 编程 Agent | 多后端 | 是 | 多模型终端编程 |
| Aider | CLI 代码编辑 | 多后端 | 是 | Git 集成的 AI 编程 |
| Cursor | AI IDE | 多后端 | 否 | IDE 深度集成 |
| GitHub Copilot | IDE 插件 + Agent | GPT-4o / Claude | 否 | 代码补全与编辑 |
| Windsurf | AI IDE | 多后端 | 否 | Flow 并行协作 |
| Devin | 全自主开发 Agent | 多后端 | 否 | Issue→PR 全流程 |
| OpenClaw | 桌面自动化 | 多后端 | 是 | GUI 操作自动化 |
| AutoGPT | 通用 Agent 框架 | 多后端 | 是 | 自主任务分解 |

## 按场景选型

```
你的需求？
  │
  ├── 终端中与 AI 结对编程
  │   ├── Claude 订阅用户 → Claude Code
  │   ├── 需要多模型 / 本地模型 → Codex CLI
  │   └── 仅需代码编辑 + Git 管理 → Aider
  │
  ├── IDE 中深度集成
  │   ├── VS Code 用户 → Cursor / GitHub Copilot
  │   └── 追求创新体验 → Windsurf
  │
  ├── 全自主开发（Issue → PR）→ Devin
  │
  ├── 桌面 GUI 自动化 → OpenClaw
  │
  └── 通用任务分解与执行 → AutoGPT
```

## 架构模式对比

| 模式 | 代表产品 | 优点 | 缺点 |
|------|---------|------|------|
| 单 Agent + 工具 | Claude Code, Aider | 简洁高效，延迟低 | 复杂任务缺乏分工 |
| 多 Agent 协作 | Codex CLI, Devin | 分工明确，质量更高 | Token 消耗大，编排复杂 |
| IDE 融合 | Cursor, Copilot | 无缝编码体验 | 限于 IDE 生态 |
| 环境控制 | OpenClaw | 操作真实 GUI | 安全性挑战大 |
