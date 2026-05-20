# Codex CLI

> OpenAI 开源的多模型终端编程 Agent —— 沙箱隔离 + 多 Agent 协作

## 产品定位

Codex CLI 是 OpenAI 开源的轻量级终端编程 Agent。与 Claude Code 不同，它从设计之初就支持**多模型后端**和**沙箱执行**，并且完全开源（Apache 2.0 协议）。

## 架构原理

```
┌──────────────────────────────────────────┐
│              Codex CLI                    │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │         Multi-Agent System          │  │
│  │                                    │  │
│  │  ┌──────────┐  ┌───────────────┐  │  │
│  │  │ Planner   │  │ Executor      │  │  │
│  │  │ (任务规划) │  │ (代码生成+执行) │  │  │
│  │  └──────────┘  └───────────────┘  │  │
│  │  ┌──────────┐  ┌───────────────┐  │  │
│  │  │ Reviewer  │  │ Sandbox       │  │  │
│  │  │ (代码审查) │  │ (Docker 隔离)  │  │  │
│  │  └──────────┘  └───────────────┘  │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │         Model Router                │  │
│  │  OpenAI │ Anthropic │ Ollama │ ...  │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### 多 Agent 分工

| Agent | 职责 | 使用模型 |
|-------|------|---------|
| **Planner** | 将用户任务分解为子任务，确定执行顺序 | 强推理模型（如 Claude Opus） |
| **Executor** | 生成代码、执行 Shell 命令、读写文件 | 编码模型（如 GPT-5-Codex） |
| **Reviewer** | 审查 Executor 的输出，检查安全性与正确性 | 可同 Executor 或独立模型 |
| **Sandbox** | Docker 容器隔离，控制文件系统和网络访问 | 系统级 |

## 部署与安装

```bash
# npm 全局安装
npm install -g @openai/codex

# 验证
codex --version

# 配置 API Key
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Docker（沙箱模式需要）
docker --version
```

### 模型后端配置

```bash
# 使用 OpenAI 后端
codex config set model gpt-5-codex

# 使用 Anthropic Claude
codex config set model claude-sonnet-4-6
codex config set provider anthropic

# 使用本地模型
codex config set model llama3.1:70b
codex config set provider ollama
codex config set ollama_url http://localhost:11434

# 查看当前配置
codex config list
```

## 沙箱机制

Codex CLI 的核心安全特性是 Docker 沙箱，所有代码执行默认在隔离容器中进行：

### 沙箱配置 (.codex.yaml)

```yaml
sandbox:
  enabled: true
  image: node:22
  workdir: /workspace
  volumes:
    - ./src:/workspace/src:ro        # 只读挂载源码
    - ./output:/workspace/output:rw  # 读写挂载输出目录
  network: none                       # 禁止网络访问
  capabilities:
    drop: [ALL]                       # 去除所有 Linux Capabilities
  limits:
    memory: 4g
    cpu: 2
```

### 安全层级

| 模式 | 描述 | 命令 |
|------|------|------|
| **sandbox** | Docker 完全隔离，默认 | `codex --sandbox run "..."` |
| **safe** | 非隔离但所有操作需确认 | `codex --safe` |
| **auto** | 自动批准低风险操作 | `codex --auto` |
| **yolo** | 无任何确认（极度危险） | `codex --yolo` |

## 使用模式

### 交互式会话

```bash
# 启动
codex

# 在会话中
> 用 Python 写一个数据分析脚本，读取 sales.csv 并生成月度报表
```

### 单次执行

```bash
# 沙箱中运行
codex exec "修复 src/ 下的所有 TypeScript 编译错误" --sandbox

# 带特定模型
codex exec "重构认证模块" --model claude-sonnet-4-6

# 输出详细日志
codex exec "运行 npm test 并修复失败用例" --log-level debug
```

### 代码审查

```bash
# 审查暂存区变更
codex review --staged

# 审查特定文件
codex review src/auth/login.ts src/auth/middleware.ts

# 审查 + 自动修复
codex review --staged --fix
```

### 批处理模式

```bash
# 批量处理多个文件
codex batch "为每个文件添加 JSDoc 注释" --files "src/**/*.ts"

# 并行处理
codex batch "运行 ESLint 并修复" --files "src/**/*.ts" --parallel 4
```

## 权限控制

```yaml
# .codex.yaml
permissions:
  # 路径白名单
  allow_paths:
    - src/**
    - tests/**
    - package.json
    - tsconfig.json

  # 路径黑名单
  deny_paths:
    - .env
    - .env.*
    - **/credentials.*
    - **/secrets/**

  # 命令白名单
  allow_commands:
    - npm
    - npx
    - git
    - node
    - python

  # 命令黑名单
  deny_commands:
    - "rm -rf"
    - "git push --force"
    - "curl * | bash"
    - "sudo"
    - "chmod 777"
```

## 与 Claude Code 的关键差异

| 维度 | Claude Code | Codex CLI |
|------|------------|-----------|
| **开源** | 否（闭源产品） | 是（Apache 2.0） |
| **模型后端** | Claude 系列专属 | OpenAI / Anthropic / Ollama / 自定义 |
| **沙箱隔离** | 无内置沙箱 | Docker 沙箱，可完全断网 |
| **Agent 架构** | 单 Agent + Skills | 多 Agent 协作（Plan/Exec/Review） |
| **本地模型** | 不支持 | 支持（via Ollama） |
| **记忆系统** | MEMORY.md 持久记忆 | 无内置长效记忆 |
| **代码审查** | /review Skill | 内置 codex review 命令 |
| **配置文件** | settings.json + CLAUDE.md | .codex.yaml |
| **优势场景** | Claude 深度用户，简洁高效 | 多模型灵活切换，安全敏感场景 |

## 实践建议

1. **安全场景首选 Codex**：需要 Docker 沙箱隔离执行不可信代码时
2. **本地模型用户**：可通过 Ollama 使用本地模型，数据不出本机
3. **多模型协作**：用 Planner 走 Claude（强推理），Executor 走 GPT-5-Codex（编码强）
4. **CI/CD 集成**：开源的 Codex 更适合集成到自动化流水线
