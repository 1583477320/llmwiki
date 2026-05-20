# Aider

> 开源 AI 结对编程工具 —— Git 原生的代码编辑专家

## 产品定位

Aider 是一款专注代码编辑的开源命令行工具。与 Claude Code / Codex CLI 的全能 Agent 路线不同，Aider 核心哲学是**只做代码编辑，做到极致**。它深度集成 Git，每次成功的代码修改自动生成规范的 commit。

## 核心机制

### Map-Refine 架构

Aider 使用两阶段方法让模型理解大型代码库：

```
阶段 1 — Map（地图构建）
  整个仓库 → 提取函数签名、类定义、文件结构 → 仓库地图

阶段 2 — Refine（精炼编辑）
  用户指令 + 仓库地图 + 相关文件 → LLM → 编辑代码
```

```
┌─────────────────────────────────────────┐
│              Aider                       │
│                                          │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ Repo Map     │  │ Edit Engine      │ │
│  │ (代码结构索引)│  │                  │ │
│  │              │  │ 1. 读取相关文件   │ │
│  │ 函数/类/接口  │  │ 2. 生成 Search/   │ │
│  │ 依赖关系图    │  │    Replace 块    │ │
│  └──────┬───────┘  │ 3. 应用编辑       │ │
│         │          │ 4. 运行测试       │ │
│         ▼          │ 5. git commit     │ │
│    提供上下文       └──────────────────┘ │
│                                          │
│  支持模型：                               │
│  Claude / GPT / Gemini / 本地模型         │
└─────────────────────────────────────────┘
```

### Search/Replace 编辑模式

Aider 使用结构化的 Search/Replace 块来编辑代码，而非重写整个文件：

```
math.py
<<<<<<< SEARCH
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
=======
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
>>>>>>> REPLACE
```

**优势**：精确编辑，不触碰未修改的代码，更省 Token，降低引入新 Bug 的概率。

## 部署与安装

```bash
# pip 安装
pip install aider-chat

# 或从源码安装（最新功能）
git clone https://github.com/Aider-AI/aider
cd aider
pip install -e .

# Playwright（网页抓取功能需要）
playwright install --with-deps chromium
```

### 模型配置

```bash
# Claude
export ANTHROPIC_API_KEY="sk-ant-..."
aider --model claude-sonnet-4-6

# OpenAI
export OPENAI_API_KEY="sk-..."
aider --model gpt-4o

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
aider --model deepseek/deepseek-chat

# 本地 Ollama 模型
aider --model ollama/qwen3:32b
```

## 基本使用

```bash
# 在项目目录中启动（最好先 git init）
cd my-project
git init && git add . && git commit -m "initial"
aider

# 指定初始文件
aider src/main.py src/utils.py

# 指定模型
aider --model claude-sonnet-4-6

# 在交互界面中
> 给 User 类添加 email 字段，更新 __init__ 和 __repr__ 方法
> 为所有 public 方法添加类型注解
> 把这个函数改成异步的
```

## 核心功能

### 1. 自动 Git 集成

```bash
> 重构 auth.py 的登录逻辑

# Aider 自动：
# 1. 读取 auth.py 和相关依赖文件
# 2. 生成 Search/Replace 编辑
# 3. 应用编辑
# 4. 运行测试（如果配置了 test-cmd）
# 5. git add + git commit（带规范的 commit message）
```

### 2. 多文件编辑

```bash
> 把整个项目的 API 路径从 /api/v1/ 迁移到 /api/v2/

# Aider 会：
# 1. 搜索所有引用 /api/v1/ 的文件
# 2. 同时编辑所有相关文件
# 3. 一次性 commit（保持修改一致性）
```

### 3. 代码库感知

```bash
> 给这个项目添加一个日志中间件

# Aider：
# 1. 通过 Repo Map 理解项目结构
# 2. 找到现有的中间件注册位置
# 3. 创建新文件 + 修改注册点
# 4. 自动 import 和依赖关联
```

### 4. 只读文件

```bash
# 将约定文件设为只读，供 AI 参考但不修改
aider --read CONVENTIONS.md --read docs/ARCHITECTURE.md

# 在 .aider.conf.yml 中配置
read:
  - CONVENTIONS.md
  - docs/API_DESIGN.md
```

### 5. Voice 模式

```bash
# 语音编码（需要安装 portaudio）
pip install aider-chat[voice]

# 启动语音模式
aider --voice

# 对着麦克风说：
# "add error handling to the login function"
```

## 配置文件 (.aider.conf.yml)

```yaml
# 模型
model: claude-sonnet-4-6

# 编辑风格
edit-format: search_replace
auto-commits: true
auto-test: true
test-cmd: "npm run test -- --reporter=dot"
lint-cmd: "npm run lint"

# 上下文
read:
  - CONVENTIONS.md
  - docs/ARCHITECTURE.md
  - CLAUDE.md

# 地图配置
map-tokens: 2048
map-refresh: auto

# 外观
dark-mode: true

# Git
git: true
gitignore: true

# 分析
analytics: false
```

## 与 Claude Code / Codex 的差异

| 特性 | Aider | Claude Code | Codex CLI |
|------|-------|------------|-----------|
| 定位 | 纯代码编辑 | 全能编程 Agent | 全能编程 Agent |
| 编辑方式 | Search/Replace 精确编辑 | Edit + Write | Edit + Write |
| Git 集成 | 原生自动 commit | 需手动 | 需手动 |
| Shell 执行 | 有限 | 完整 | 完整（沙箱） |
| 浏览器自动化 | 无 | 无 | 无 |
| 擅长场景 | 跨文件一致性修改 | 探索性开发 | 安全敏感开发 |

::: tip 选择建议
Aider 最适合的场景：已经明确知道要改什么，需要精确的多文件编辑，并且希望每个语义修改都有独立的 Git 记录。对于探索性任务（"帮我看看这个 Bug 的原因"），Claude Code 更合适。
:::
