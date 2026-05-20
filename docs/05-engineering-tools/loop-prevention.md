# 防止 Agent 动作死循环

> 工程容错的反思机制 —— 当 Agent 陷入困境时如何自救

## 问题定义

Agent 在 ReAct 循环中可能出现**动作死循环**（Action Loop）—— 模型在数个状态间反复横跳，无法向目标推进：

```
Thought: 我需要点击登录按钮
Action: click(button="登录")
Observation: 页面没有变化
Thought: 可能没点到，再试一次
Action: click(button="登录")
Observation: 页面依然没有变化
Thought: 再试一次...
→ 无限循环
```

---

## 死循环的根源分类

### 1. 目标混淆

模型对用户目标理解有偏差，在错误的方向上反复尝试。

### 2. 工具使用错误

模型不知道如何正确使用工具（参数错误、选错工具），导致每次执行失败。

### 3. 环境反馈不清

Observation 信息不足，模型无法判断操作是否成功，只能盲目重试。

### 4. 幻觉循环

模型「认为」自己做对了，但实际上产生了幻觉，基于幻觉结果继续推理。

### 5. 策略坍缩

在不确定性高的场景中，模型收敛到少数几个「看起来安全」的动作，放弃探索。

---

## 检测机制

### 1. 动作重复检测

最直接的方法 —— 检测连续相同 Action：

```python
def detect_action_loop(history: list[Action], threshold: int = 3) -> bool:
    if len(history) < threshold:
        return False
    recent = history[-threshold:]
    # 检查最近的 N 个动作是否完全相同
    return all(a == recent[0] for a in recent)
```

### 2. 语义重复检测

更强大的方案 —— 通过嵌入相似度检测语义层面的重复：

```python
def detect_semantic_loop(history: list[str], window: int = 5, sim_threshold: float = 0.9) -> bool:
    recent = history[-window:]
    embeddings = [embed(a) for a in recent]
    similarities = cosine_matrix(embeddings)
    # 检查是否有密集的高相似度聚类
    return (similarities > sim_threshold).sum() > window * 1.5
```

### 3. 状态轨迹分析

| 检测维度 | 方法 |
|---------|------|
| **工具分布异常** | 同一工具的调用占比超过阈值（如 > 60%） |
| **规划长度异常** | 步数超过预期范围的 2-3 倍 |
| **进度停滞** | 连续 N 步没有新的 Observation 或实质进展 |
| **Token 消耗异常** | Context 增长速度远超正常速率 |

---

## 反思与干预机制

### Layer 1：自我反思 (Self-Reflection)

让 LLM 在做出下一步决策前，先反思当前状态：

```
Prompt 注入：
"You have taken {N} steps so far. The last {K} actions resulted in
 similar observations. Pause and reflect:
 1. What exactly are you trying to achieve?
 2. What have you already tried?
 3. What alternative approaches could you take?
 4. Is there information you're missing?"
```

### Layer 2：策略注入 (Strategy Injection)

当检测到潜在循环时，向 Prompt 注入「打破循环」的启发式策略：

```
"Consider these strategies:
 - Try a fundamentally different tool, not a variant of the current one
 - Ask the user for clarification instead of guessing
 - Break the task into smaller sub-steps
 - Verify your understanding of the tool's behavior"
```

### Layer 3：环境强制干预 (Environmental Intervention)

如果反思和策略注入都失败，环境层面强制执行干预：

| 干预手段 | 描述 |
|---------|------|
| **重置 Context** | 回退到循环开始前的 Context + 摘要 |
| **切换策略模式** | 从不限步骤模式切换到强制规划模式 |
| **降级处理** | 放弃当前子任务，输出部分结果 |
| **上报用户** | 暂停执行，展示当前状态请求人工决策 |
| **强制终止** | 硬终止，防止资源耗尽 |

---

## 完整的防死循环状态机

```
┌──────────────────────────────────────────────────────┐
│                  Agent 执行循环                        │
│                                                      │
│  ┌──────────┐    检测通过    ┌──────────┐            │
│  │  执行动作  │─────────────→│  动作记录  │            │
│  └──────────┘               └──────────┘            │
│       │                          │                   │
│       │                     ┌────▼────────┐          │
│       │                     │  循环检测器   │          │
│       │                     └────┬────────┘          │
│       │                ┌─────────┼─────────┐        │
│       │          无循环 │         │ 疑似循环  │        │
│       │                │    ┌────▼────┐     │        │
│       │                │    │ 置信度？  │     │        │
│       │                │    └────┬────┘     │        │
│       │                │    ┌────┼────┐     │        │
│       │                │ 低  │      高│     │        │
│       │                │    │   ┌───▼──┐   │        │
│       │                │    │   │ 置信   │   │        │
│       │                │    │   │ 度？   │   │        │
│       │                │    │   └─┬─┬──┘   │        │
│       │                │    │  中 │ │高    │        │
│       ▼                ▼    ▼     ▼ ▼      │        │
│  ┌────────┐      ┌────────┐   ┌────────┐  │        │
│  │ 继续执行 │      │ 自我反思 │   │ 环境干预 │  │        │
│  └────────┘      └────────┘   └────────┘  │        │
│       │                │           │       │        │
│       └────────────────┴───────────┘       │        │
│                      │                     │        │
│                      ▼                     │        │
│              返回执行循环                    │        │
│                      └─────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

## 工程最佳实践

### 配置示例

```yaml
loop_prevention:
  # 动作级检测
  exact_action_repeat_threshold: 3
  semantic_repeat_threshold: 4
  semantic_similarity_cutoff: 0.85

  # 全局约束
  max_total_steps: 15
  max_same_tool_ratio: 0.6

  # 干预策略
  reflection_on_suspicion: true
  force_terminate_on_high_confidence: true

  # Token 预算
  max_context_tokens: 32000
  inject_warning_at_token_ratio: 0.7
```

### 监控指标

在生产环境中应持续监控：

- **循环发生率**：需要干预的请求占比
- **平均干预层级**：大多数干预停留在自我反思层（健康），还是频繁触发强制终止（需优化）
- **误报率**：正常探索行为被误判为循环的比例
- **干预后的任务完成率**：干预是否有效帮助任务继续推进

::: tip 核心哲学
防止死循环不是「让 Agent 不犯错」，而是「让 Agent 在犯错时能被发现并被干预」。好的循环防护系统应该是**多层渐进式**的 —— 先提醒、再引导、最后强制介入。
:::
