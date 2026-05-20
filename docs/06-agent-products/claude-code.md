# Claude Code

> Anthropic 出品的命令行 AI 编程 Agent —— 终端中的结对编程伙伴

## 产品定位

Claude Code 是 Anthropic 官方推出的命令行 Agent 工具，将 Claude 模型的能力直接嵌入开发者的终端工作流。它不是一个代码补全工具，而是一个拥有完整文件系统访问权限的**自主编程 Agent**。

## 架构原理

```
┌─────────────────────────────────────────┐
│           Claude Code CLI (Node.js)       │
│                                          │
│  ┌───────────────┐  ┌────────────────┐  │
│  │  Agent Loop    │  │  Tool System   │  │
│  │  (ReAct 范式)  │  │                │  │
│  │               │  │  Read / Write   │  │
│  │  Thought →    │  │  Edit / Bash    │  │
│  │  Action →     │  │  Agent / Web*   │  │
│  │  Observe →    │  │  Task / Skill   │  │
│  │  Loop         │  │                │  │
│  └───────┬───────┘  └────────────────┘  │
│          │                  │            │
│          ▼                  ▼            │
│     Claude API        Filesystem /       │
│   (Opus/Sonnet)       Shell / Git       │
└─────────────────────────────────────────┘
```

Claude Code 基于 ReAct 循环：模型输出 Thought（推理）+ Tool Call（行动），工具执行后结果返回模型继续推理，直到任务完成。

## 部署与安装

```bash
# 方式一：npm 全局安装（推荐）
npm install -g @anthropic-ai/claude-code

# 方式二：一键脚本
curl -fsSL https://claude.ai/install.sh | bash

# 认证登录
claude login

# 验证安装
claude --version
```

### 系统要求

- Node.js 18+
- 操作系统：macOS / Ubuntu 20+ / Windows (WSL)
- API 认证（需 Anthropic Console API Key 或 Claude 订阅）

## 核心工具集

Claude Code 拥有 16+ 内置工具：

| 工具 | 功能 | 示例 |
|------|------|------|
| **Read** | 读取文件内容 | 阅读源码、配置、日志 |
| **Write** | 创建/覆盖文件 | 新建文件、完整重写 |
| **Edit** | 精确字符串替换 | 局部修改、重构 |
| **Bash** | 执行 Shell 命令 | npm、git、curl、docker |
| **Agent** | 启动子 Agent | 并行处理复杂任务 |
| **Task** | 任务管理与追踪 | 多步骤任务的状态跟踪 |
| **WebFetch** | 获取网页内容 | 读取文档、API 响应 |
| **WebSearch** | 网络搜索 | 查找最新信息 |
| **AskUserQuestion** | 向用户提问 | 澄清需求、选择方案 |
| **Skill** | 调用专用技能 | /review、/init 等斜杠命令 |

## 权限系统

Claude Code 的权限控制通过 `~/.claude/settings.json` 实现：

```json
{
  "permissions": {
    "allow": [
      "Bash(npm run:*)",
      "Bash(npm test:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(npx:*:*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(git push --force:*)",
      "Bash(curl:* | bash)"
    ]
  }
}
```

### 权限层级

| 层级 | 路径 | 作用域 |
|------|------|--------|
| 全局 | `~/.claude/settings.json` | 所有项目 |
| 项目 | `.claude/settings.json` | 当前项目 |
| 本地 | `.claude/settings.local.json` | 本地覆盖，不提交 Git |

## 高级功能

### 1. CLAUDE.md — 项目指令文件

在项目根目录放置 `CLAUDE.md`，定义项目规则：

```markdown
# CLAUDE.md

## 代码风格
- 使用 TypeScript 严格模式
- 不使用 any 类型
- 函数优先使用箭头函数

## 项目架构
- src/ 为源代码
- 组件放在 src/components/
- API 调用统一通过 src/lib/api.ts

## 禁止事项
- 不要使用 eval()
- 不要在组件中直接操作 DOM
- 不要引入新的依赖包，先询问
```

### 2. Hooks — 事件驱动的自动化

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "command": "npx prettier --write $CLAUDE_TOOL_INPUT_FILE_PATH"
        }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash(npm test)",
        "hooks": [{
          "command": "echo 'Running tests...'"
        }]
      }
    ]
  }
}
```

### 3. 记忆系统

Claude Code 拥有持久化记忆，存储在 `.claude/projects/<project>/memory/`：

| 记忆类型 | 用途 | 示例 |
|---------|------|------|
| **user** | 用户偏好与背景 | "用户是资深后端，不熟悉前端" |
| **feedback** | 用户反馈与纠正 | "不要在测试中 mock 数据库" |
| **project** | 项目上下文与决策 | "API 重构是为了合规" |
| **reference** | 外部系统指针 | "Bug 追踪在 Linear 项目 INGEST" |

### 4. Skills — 可复用的专业能力

通过 `/` 调用内置或自定义 Skills：

| Skill | 功能 |
|-------|------|
| `/review` | PR 代码审查 |
| `/init` | 生成项目的 CLAUDE.md |
| `/security-review` | 安全审计 |
| `/simplify` | 代码简化与重构 |
| `/loop` | 定时循环运行任务 |

## 使用模式

### 交互模式

```bash
# 启动交互会话
claude

# 在会话中直接对话
> 帮我重构 src/utils 目录下的所有文件，添加完整的 TypeScript 类型注解
```

### 单次模式

```bash
# 单次问答
claude -p "解释这个项目的认证流程"

# 管道模式
cat error.log | claude -p "分析这些错误，给出修复建议"

# 文件输入
claude -p "审查这个 PR 的代码质量" < changes.diff
```

### 自动化模式

```bash
# 跳过所有权限确认（危险，仅在可信场景使用）
claude --dangerously-skip-permissions -p "运行测试并修复所有失败的用例"

# 输出模式
claude -p "生成 API 文档" --output-format stream-json
```

## 实践建议

1. **编写高质量的 CLAUDE.md**：这是提升 Claude Code 表现最直接的方式，定义清楚项目规则、代码风格、架构约定
2. **善用权限配置**：信任的操作放入 allow 列表减少交互摩擦，危险操作放入 deny 列表
3. **大任务用 Agent 子进程**：复杂的多文件任务可以拆分为多个并行的子 Agent
4. **利用记忆系统**：当 Claude Code 做出不符合预期的行为时，直接纠正，它会记住反馈
5. **定期 `/compact`**：长会话中 Token 会累积，主动压缩历史避免上下文溢出
