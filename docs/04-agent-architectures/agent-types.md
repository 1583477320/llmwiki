# Agent 类型与使用范式

> 从单 Agent 到多 Agent 协作 —— 不同架构范式的选择与落地

## Agent 的本质

Agent 是一个能够**感知环境、推理规划、执行动作**的自主系统。其核心公式：

$$
\text{Agent} = \text{LLM} + \text{Tools} + \text{Memory} + \text{Orchestration}
$$

不同 Agent 类型的差异，本质上在于**编排层（Orchestration）** 的设计不同。

---

## 单 Agent 范式

### 1. ReAct Agent

最经典的范式，交替进行推理和行动。

**流程**：`Thought → Action → Observation → Thought → ...`

**适用场景**：
- 需要多步推理的工具调用任务
- 信息检索与聚合
- 代码生成与调试

**示例（伪代码）**：

```python
for step in range(max_steps):
    thought, action = llm.reason(context, tools)
    if action.is_final:
        return action.answer
    observation = execute_tool(action)
    context.append(observation)
```

### 2. Plan-and-Execute Agent

先制定完整计划，再逐步执行。计划可动态修正。

**流程**：`Plan → Execute(Step1) → Verify → Execute(Step2) → ... → Finish`

**适用场景**：
- 复杂多步骤任务（如「帮我做一个数据分析报告」）
- 子任务间有明确依赖关系
- 需要人对计划进行审核的场景

**优势**：全局规划能力强，减少短视行为
**代价**：初始规划耗时，计划可能过时需修正

### 3. Router Agent

轻量级分类 Agent，根据用户意图路由到不同的下游处理逻辑。

**流程**：`Input → Classify Intent → Route to Handler → Response`

**适用场景**：
- 客服系统（退款/咨询/投诉分流）
- 多工具系统中决定调用哪类工具
- 作为多 Agent 系统的入口网关

**实现要点**：
- 分类 Prompt 需精心设计，添加 Few-shot 示例
- 分类结果需要置信度阈值，低于阈值走兜底逻辑
- Router 本身应轻量，可用更小、更快的模型

---

## 多 Agent 协作范式

### 1. 顺序流水线（Sequential Pipeline）

多个 Agent 按固定顺序执行，前一个的输出是后一个的输入。

```
[Research Agent] → [Writing Agent] → [Review Agent] → 最终输出
```

**适用场景**：
- 内容生产流水线（调研→撰写→审核）
- 数据处理 ETL 流程
- 代码生成→审查→测试的 CI 流程

**优点**：逻辑清晰，易于调试和监控
**缺点**：缺乏灵活性，某步失败会阻塞整条流水线

### 2. 层级式（Hierarchical Agent）

一个 Master Agent 负责任务分解和分发，多个 Worker Agent 执行具体子任务。

```
              ┌─────────────┐
              │ Master Agent │ (规划/分发/汇总)
              └──────┬──────┘
         ┌───────────┼───────────┐
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │ Worker 1│ │ Worker 2│ │ Worker 3│
    │ (搜索)   │ │ (分析)   │ │ (编码)   │
    └─────────┘ └─────────┘ └─────────┘
```

**适用场景**：
- 复杂项目（如「构建一个完整的 Web 应用」）
- 需要多个专业领域知识协作的任务
- 任务可以被自然分解为独立子任务

**关键设计**：
- Master 需要强大的任务分解能力（通常用最强模型）
- Worker 可使用更小、更专项的模型以节省成本
- 需要合理管理子任务间的依赖和结果汇总

### 3. 对话式协作（Collaborative / Group Chat）

多个 Agent 在一个共享对话中交流，自发协作完成任务。

```
[User] → [Agent A]: 我们来讨论这个方案
       → [Agent B]: 我同意，但有个风险...
       → [Agent C]: 我可以负责实现部分...
       → [Agent A]: 好的，那我们这样分工...
```

**适用场景**：
- 头脑风暴和创意生成
- 复杂决策需要多视角评估
- 需要不同角色（PM/Dev/QA）模拟的场景

**挑战**：
- 对话可能发散，需 Speaker Selection 机制
- Token 消耗大（多 Agent 的上下文都会膨胀）
- 幻觉风险叠加

---

## 主流框架对比

| 框架 | 核心范式 | 特点 | 适合场景 |
|------|---------|------|---------|
| **LangChain** | ReAct + Chain | 生态丰富，抽象层次多 | 快速原型 |
| **LangGraph** | 状态机图 | 精细控制流，支持循环和分支 | 复杂 Agent 编排 |
| **AutoGen** | 对话协作 | 多 Agent 对话，微软出品 | 多 Agent 研究 |
| **CrewAI** | 角色扮演 | 预定义角色，简化多 Agent 设置 | 内容生产 |
| **OpenAI Agents SDK** | ReAct + Handoff | 原生 Function Call，Agent 间可移交 | 生产环境 |
| **Dify / Coze** | 低代码编排 | 可视化拖拽编排 | 非技术用户 |

---

## 选择 Agent 架构的决策树

```
任务复杂度？
  │
  ├── 单步可完成 → 直接 LLM 调用，无需 Agent
  │
  ├── 2-5步，线性依赖 → ReAct Agent
  │
  ├── 5+步，复杂依赖 → Plan-and-Execute Agent
  │
  ├── 可分解为独立子任务 → Hierarchical Multi-Agent
  │
  ├── 需要多角色视角 → 对话式协作 Multi-Agent
  │
  └── 仅需分类路由 → Router Agent
```

::: tip 核心原则
**用最简单的架构解决问题**。不要为一个简单的 API 调用包装一个 Multi-Agent 系统。Agent 架构的复杂度应与任务的实际复杂度匹配 —— 每增加一层编排，就增加一层延迟、成本和出错概率。
:::

---

## 生产落地实践要点

### Token 预算控制

多 Agent 系统容易 Token 消耗失控：

```
单 Agent 调用：~2K input + ~500 output = ~2.5K tokens
多 Agent 系统（4 Agent，各3轮对话）：~30K+ tokens
```

**控制策略**：
- 用更小的模型做 Worker
- 压缩 Agent 间传递的信息（传摘要而非全量上下文）
- 设置每 Agent 的 Token 预算上限

### 错误传播与隔离

多 Agent 系统的级联故障风险：

```python
# 隔离模式：每个 Worker 独立上下文，失败不影响其他
results = []
for worker_task in tasks:
    try:
        result = worker_agent.run(worker_task, isolated_context=True)
        results.append(result)
    except AgentException as e:
        results.append({"error": str(e), "status": "failed"})

# Master 统一处理失败
final = master_agent.synthesize(results, handle_errors=True)
```

### 可观测性

多 Agent 系统必须追踪的信息：

- 每个 Agent 的输入/输出 Trace
- Tool Call 的参数与返回值
- Token 消耗（按 Agent 和按步骤）
- 每步耗时（首 Token 时间 + 生成时间）
- 错误发生的位置与原因

推荐使用 **LangSmith / Phoenix / OpenInference** 等工具进行 Agent Trace 的可视化追踪。
