# 结构化输出与 Tool Call

> 基于 JSON Schema 的强约束 —— 让 Agent 可靠地执行结构化操作

## 为什么需要结构化输出？

Agent 需要与外部系统交互 —— 调用 API、读写数据库、操作系统文件。这些交互要求输出具有严格的格式约束，而非自由文本。

### 自由文本 vs 结构化输出

| 自由文本 | 结构化输出 |
|---------|-----------|
| "调用搜索功能，关键词是猫" | `{"tool": "search", "query": "猫"}` |
| 正则解析，容错性差 | JSON 反序列化，类型安全 |
| 格式不可控 | Schema 强约束 |

## JSON Schema 约束

### 工作原理

通过 JSON Schema 定义工具的参数规范，模型在生成时受 Schema 约束，确保输出合法的 JSON。

```json
{
  "name": "get_weather",
  "description": "获取指定城市的天气信息",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名称"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "温度单位"
      }
    },
    "required": ["city"]
  }
}
```

### 约束实现方案

| 方案 | 原理 | 代表 |
|------|------|------|
| **Grammar-Guided** | 使用形式文法约束 Token 选择 | llama.cpp grammar |
| **Logit Masking** | 动态屏蔽不符合 Schema 的 Token | Outlines, Guidance |
| **FST 约束** | 构建有限状态转换器控制生成路径 | OpenAI Structured Outputs |
| **Rejection Sampling** | 生成后校验，失败则重试 | 早期 Tool Call 实现 |

### OpenAI Structured Outputs

OpenAI 的 Structured Outputs 保证输出 100% 符合 JSON Schema：

- 后端将 JSON Schema 编译为有限状态机
- 在每一生成步骤，仅允许符合状态机路径的 Token
- 支持嵌套对象、枚举、联合类型等复杂 Schema

## Function Calling / Tool Call

Tool Call 是结构化输出在 Agent 场景的具体应用 —— 模型决定调用哪个函数、传递什么参数。

### 完整流程

```
1. User: "北京今天天气怎么样？"
2. LLM Output:
   {
     "tool_calls": [{
       "id": "call_abc123",
       "function": {
         "name": "get_weather",
         "arguments": "{\"city\": \"北京\", \"unit\": \"celsius\"}"
       }
     }]
   }
3. System: 调用 get_weather(city="北京", unit="celsius")
4. Tool Response: {"temperature": 22, "condition": "晴"}
5. LLM: "北京今天晴天，气温22°C。"
```

### 多工具编排

当 Agent 拥有多个工具时，需要**工具选择**策略：

- **全部暴露**：将所有工具定义放入 System Prompt，由模型自主选择
- **分层路由**：先由轻量级分类器决定工具类别，再调用 LLM 填参
- **并行调用**：当多个工具不相互依赖时，一次生成返回多个 Tool Call

::: tip 设计原则
每个工具只做一件事，但做好这件事。工具的粒度应当匹配 Agent 的推理粒度 —— 太粗模型难以灵活组合，太细会增加调用轮次和延迟。
:::

## 约束性生成在工程中的挑战

### 延迟

Logit Masking 和 FST 约束增加了每步推理的计算量。优化方向：

- 提前编译 Schema 到约束图
- 缓存约束状态
- 仅对工具参数而非整条消息施加约束

### 鲁棒性

即使有 Schema 约束，仍需处理：

- **幻觉参数**：模型编造不存在的参数值
- **类型错误**：字符串 vs 数字 vs 布尔的混淆
- **参数遗漏**：Required 字段被遗漏

**最佳实践**：在 Agent 框架侧对 Tool Call 参数做二次校验，不信任模型的原始输出。
