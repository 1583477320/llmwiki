# LLM 核心原理

> Token 机制、Context 窗口与采样控制 —— 驾驭大模型的基础能力

## Token 与 Tokenization

LLM 不直接处理文本，而是处理 **Token**（词元）。Tokenization 是将文本切分为 Token 序列的过程。

### 主流分词算法

| 算法 | 代表模型 | 特点 |
|------|---------|------|
| BPE | GPT 系列 | 从字符级开始逐步合并高频对 |
| WordPiece | BERT | 基于似然的合并策略 |
| SentencePiece | LLaMA, T5 | 语言无关，直接处理原始文本 |
| Unigram | 部分多语言模型 | 基于概率的减量式分词 |

### 关键概念

- **词表大小（Vocabulary Size）**：影响嵌入矩阵大小和计算量
- **Special Tokens**：`<s>`, `</s>`, `<pad>`, `<unk>`, `<|user|>`, `<|assistant|>` 等
- **Token 压缩率**：中文约 1.5-2 字符/token，英文约 4 字符/token

::: tip 工程实践
Token 计数直接影响 API 成本和 Context 窗口预算。在构建 Agent 系统时，需要精确控制 Prompt 的 Token 占用量。
:::

## Context 窗口

Context 窗口决定了模型单次能「看到」的 Token 数量上限。

### Context 长度演进

- **GPT-2**：1,024 tokens
- **GPT-3**：2,048 tokens
- **GPT-4**：8K → 128K tokens
- **Claude 3**：200K tokens
- **Gemini 1.5 Pro**：1M tokens

### 位置插值与外推

超长 Context 需要特殊的扩展技术：

- **Position Interpolation**：线性缩放位置索引
- **NTK-aware Scaling**：基于 Neural Tangent Kernel 的非线性缩放
- **YaRN**：结合 NTK 与温度调节的综合方案

### 注意力机制挑战

长 Context 面临的计算瓶颈：标准注意力的复杂度为 $O(n^2)$，因此 FlashAttention、RingAttention 等高效实现成为长 Context 推理的必备技术。

---

## 采样参数控制

理解以下参数是精确控制 LLM 输出的关键。

### Temperature（温度）

控制输出的随机性。在 softmax 之前进行调节：

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

| 温度值 | 效果 | 适用场景 |
|--------|------|---------|
| $T \to 0$ | 确定性输出 | 数学计算、代码生成 |
| $T = 0.7$ | 平衡 | 通用对话、写作 |
| $T > 1.0$ | 高随机性 | 创意生成、头脑风暴 |

### Top-K 采样

只从概率最高的 K 个 Token 中采样，截断低概率尾部。

### Top-P（Nucleus Sampling）

从累积概率超过 P 的最小 Token 集合中采样。相比 Top-K 更灵活 —— 动态调整候选集大小。

### 其他关键参数

- **Repetition Penalty**：惩罚已出现的 Token，防止重复
- **Frequency Penalty**：按出现频率惩罚
- **Presence Penalty**：按是否出现过惩罚

---

## 预训练与后训练

### 预训练（Pre-training）

在海量语料上进行自回归语言建模训练：

$$
\mathcal{L} = -\sum_{t} \log P(x_t | x_{<t})
$$

核心涉及：
- 数据质量过滤与去重
- 混合精度训练（FP16/BF16）
- 分布式训练策略（数据并行、模型并行、流水线并行）

### 后训练（Post-training）

#### SFT（监督微调）

在高质量指令-回复对上微调，使模型学会遵循指令格式。

#### RLHF（人类反馈强化学习）

1. 训练奖励模型（Reward Model）
2. 使用 PPO 优化策略模型
3. 可选：DPO 直接优化偏好，无需显式奖励模型
