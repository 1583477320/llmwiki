# Cursor

> AI-first 编程 IDE —— 从代码补全到全自动 Agent 的完整覆盖

## 产品定位

Cursor 不是 VS Code 的 AI 插件，而是从底层重新设计的 **AI-first IDE**。它基于 VS Code 内核，但深度集成了多模态 Agent、代码库索引（RAG）和多模型支持。当前最强的商业化 AI IDE 之一。

## 架构原理

```
┌────────────────────────────────────────────┐
│               Cursor IDE                    │
│                                            │
│  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Tab          │  │ Composer (Agent 模式) │ │
│  │ (内联代码补全)│  │                      │ │
│  │             │  │ 全自动多文件编辑       │ │
│  │ 准实时建议   │  │ 自主搜索代码库         │ │
│  │ 多光标编辑   │  │ 运行终端命令           │ │
│  │ 智能跳转     │  │ 理解项目结构           │ │
│  └─────────────┘  └──────────────────────┘ │
│                                            │
│  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Chat         │  │ Codebase Indexing    │ │
│  │ (对话面板)    │  │ (后台代码库索引)      │ │
│  │             │  │                      │ │
│  │ 问答 & 解释  │  │ 持续构建 Embedding    │ │
│  │ 代码片段生成 │  │ RAG 检索相关上下文    │ │
│  │ @ 文件引用   │  │ 语义搜索代码          │ │
│  └─────────────┘  └──────────────────────┘ │
│                                            │
│  模型后端：GPT / Claude / Gemini / 自定义  │
└────────────────────────────────────────────┘
```

## 三种 AI 模式

| 模式 | 能力 | 适用场景 | 快捷键 |
|------|------|---------|--------|
| **Tab** | 内联代码补全、多光标编辑、智能跳转 | 编码时自动触发 | Tab 接受 |
| **Ask (Chat)** | 单次问答，不修改代码；@ 引用文件/文档 | 理解代码、提问 | `Cmd+L` |
| **Agent (Composer)** | 自主编辑多文件、搜索代码库、运行终端 | 复杂多步任务 | `Cmd+I` |

### Tab 模式示例

```
// 输入
function calculateStats(da

// Tab 自动建议
function calculateStats(data: number[]): Stats {
  const sorted = [...data].sort((a, b) => a - b);
  const sum = data.reduce((acc, val) => acc + val, 0);
  return {
    mean: sum / data.length,
    median: sorted[Math.floor(sorted.length / 2)],
    min: sorted[0],
    max: sorted[sorted.length - 1]
  };
}
```

### Agent 模式示例

```
用户（Cmd+I）：
"在 src/ 下创建 API 客户端模块，封装 fetch 请求，
 添加 JWT 自动刷新、请求重试和错误处理"

Agent 执行：
1. 搜索现有代码了解项目结构
2. 创建 src/lib/api-client.ts
3. 添加 src/lib/auth.ts 中 Token 刷新逻辑
4. 创建 src/types/api.ts 类型定义
5. 更新 src/index.ts 导出新模块
6. 运行 npm run typecheck 验证
```

## Codebase Indexing（RAG）

Cursor 的后台服务持续索引你的代码库：

```json
// .cursor/index.json（自动生成）
{
  "indexed_files": 1523,
  "embeddings_model": "text-embedding-3-small",
  "index_version": 2
}
```

当你提问时，Cursor 自动检索最相关的代码片段注入 Context：

```
用户：@Codebase 这个项目的缓存策略是什么？

Cursor：
  1. 在向量数据库中搜索"缓存"
  2. 检索到：
     - src/lib/cache.ts (相关性 0.92)
     - docs/ARCHITECTURE.md 第 45-67 行 (相关性 0.87)
     - src/middleware/cache.ts (相关性 0.81)
  3. 将这些内容 + 用户提问发给 LLM
  4. 给出准确回答
```

## 项目规则系统

### .cursor/rules (Rules for AI)

```json
// .cursor/rules
{
  "rules": [
    "使用 TypeScript 严格模式，禁止 any 类型",
    "所有 API 请求统一通过 @/lib/api-client 发送",
    "React 组件默认使用 Server Component，需要交互时加 'use client'",
    "Tailwind CSS 类名按 base → layout → typography → colors → states 排序",
    "新增依赖前必须先询问用户"
  ]
}
```

### .cursorignore

```gitignore
# 不让 Cursor 索引的文件
node_modules/
dist/
.next/
*.generated.*
.env*
**/secrets/
*.log
```

## 模型配置

| 模型用途 | 推荐选择 |
|---------|---------|
| Tab 补全 | Cursor 专用补全模型（延迟最低） |
| Chat/Ask | Claude Sonnet（理解能力好） |
| Agent | Claude Sonnet / GPT-4o（推理 + 代码能力均衡） |
| 长任务编排 | Claude Opus（强推理，计划型任务） |

## 与 Claude Code / Codex CLI 的差异

| 维度 | Cursor | Claude Code | Codex CLI |
|------|--------|------------|-----------|
| 使用界面 | GUI IDE | 终端 CLI | 终端 CLI |
| 上手门槛 | 低（可视化操作） | 中（命令行） | 中（命令行） |
| 代码库理解 | 后台自动索引 + RAG | 基于文件读取 | 基于文件读取 |
| 多文件编辑 | Agent 模式自主编辑 | Agent 循环自主编辑 | 多 Agent 协作 |
| 离线可用 | 部分功能 | 否 | 可（本地模型） |
| 价格 | $20/月 | API 按量计费 | API 按量计费 |
| 安全管控 | IDE 内置权限 | settings.json | .codex.yaml |

## 实践建议

1. **`.cursor/rules` 是核心**：写得越具体，Agent 产出越精准
2. **Agent 模式用 @file / @folder 引用**：减少 Agent 的搜索时间，精准指导
3. **大项目用 .cursorignore**：避免索引 `node_modules` 等无关文件
4. **长任务结合 Claude Code**：IDE 里短步操作用 Cursor，完整的终端级任务用 Claude Code
