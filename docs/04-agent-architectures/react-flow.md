# ReAct 状态机控制流

> 感知-思考-行动循环 —— Agent 认知架构的核心范式

## ReAct 范式

ReAct（Reasoning + Acting）是当前 Agent 系统最核心的认知范式，将大模型的推理能力与工具调用能力交织进行。

### 核心理念

传统 LLM 使用是「一次推理、一次输出」，而 ReAct Agent 在一个循环中交替执行：

```
观察 (Observation) → 思考 (Thought) → 行动 (Action) → 观察 (Observation) → ...
```

## ReAct 循环的完整状态机

```
┌─────────────────────────────────────────┐
│               Agent Loop                 │
│                                          │
│  ┌──────────┐    ┌──────────┐           │
│  │ OBSERVE  │───→│  THINK   │           │
│  │ 感知环境  │    │ 推理分析  │           │
│  └──────────┘    └──────────┘           │
│       ↑               │                  │
│       │               ↓                  │
│  ┌──────────┐    ┌──────────┐           │
│  │ REFLECT  │←───│   ACT    │           │
│  │ 反思评估  │    │ 执行动作  │           │
│  └──────────┘    └──────────┘           │
│       │               │                  │
│       └───────────────┘                  │
│                                          │
│         完成条件满足 → 终止               │
└─────────────────────────────────────────┘
```

## 各阶段详解

### 1. 感知 (Observe)

从环境中获取当前状态信息：

- 用户输入的原始文本/图像
- 工具调用的返回结果
- 系统消息与历史对话
- 外部 API 返回的结构化数据
- 浏览器/桌面截图等视觉信号

### 2. 推理 (Think)

LLM 基于当前 Context 进行分析：

- 理解用户意图与子目标
- 分析当前状态是否满足终止条件
- 决定下一步需要什么信息
- 规划需要调用哪些工具
- 评估风险与不确定性

### 3. 行动 (Act)

执行具体操作：

- **Tool Call**：调用外部函数/API
- **代码执行**：运行生成的代码片段
- **环境交互**：点击、输入、滚动等 GUI 操作
- **知识检索**：查询向量数据库或文档库

### 4. 反思 (Reflect)

评估上一步行动的结果：

- 结果是否符合预期？
- 是否需要修正策略？
- 是否存在幻觉或事实错误？
- 是否需要自我纠错？

## ReAct Prompt 模板

典型的 ReAct Prompt 结构：

```
System: You are an agent that can use tools.
        Use the following format:

        Thought: your reasoning about what to do next
        Action: the tool to use (must be one of [...])
        Action Input: the input to the tool
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation)
        Thought: I now have the final answer
        Final Answer: the final answer to the user

User: <task description>
```

## 关键设计考量

### 终止条件

- **显式**：模型输出 `Final Answer` 或特殊终止 Token
- **步数限制**：设置最大迭代步数（如 15 步），防止无限循环
- **超时机制**：设置全局超时时间
- **重复检测**：检测连续相同的 Action，强制终止

### Context 管理

ReAct 循环会在每轮追加 Observation 和新的 Thought/Action，Context 快速膨胀：

- **滑动窗口**：仅保留最近 N 轮交互
- **压缩/摘要**：对历史步骤进行摘要压缩
- **分层记忆**：将历史交互移至外部记忆系统

### 思维链变体

| 变体 | 描述 |
|------|------|
| **Self-Consistency** | 多次采样取多数结果 |
| **Tree of Thoughts** | 树形探索多条推理路径 |
| **Graph of Thoughts** | 图结构，支持聚合与回溯 |

::: tip 工程实践
ReAct 循环的数量通常通过 `max_steps` 参数控制。在工程中，将上限设为 10-15 步通常足够，过多的步数不仅消耗 Token，也增加了状态漂移的风险。
:::
