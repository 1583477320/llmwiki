# Transformer 架构详解

> 自注意力机制与位置编码 —— 现代大模型的基石

## 为什么需要 Transformer？

在 Transformer 出现之前，序列建模依赖于 RNN/LSTM 的循环结构，存在两个根本性局限：

1. **串行计算**：无法并行化，训练效率低
2. **长程依赖衰减**：梯度在长序列中消失或爆炸

Transformer 通过**自注意力机制**一次性解决了这两个问题。

## 自注意力机制

自注意力（Self-Attention）的核心思想：让序列中的每个位置都直接与所有其他位置交互：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### QKV 三元组

- **Q (Query)**：「我在找什么？」
- **K (Key)**：「我有什么信息？」
- **V (Value)**：「我的实际内容是什么？」

三个矩阵均来自同一输入的线性投影。

### 缩放因子 $\sqrt{d_k}$

除以 $\sqrt{d_k}$ 的原因：当 $d_k$ 较大时，点积 $QK^T$ 的方差会增大，导致 softmax 进入饱和区（梯度消失）。缩放保持了梯度的健康流动。

## 多头注意力

单头注意力的表达能力有限。**多头注意力（Multi-Head Attention）** 并行运行多个注意力头，每个头关注不同的表示子空间：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

## 位置编码

注意力机制本身是**置换不变**的 —— 打乱输入顺序，输出不变。位置编码（Positional Encoding）为模型注入序列顺序信息。

### 正弦位置编码（原始方案）

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### RoPE（旋转位置编码）

现代 LLM（LLaMA、Qwen 等）普遍采用的方案，通过旋转矩阵将位置信息编码到注意力计算中，具有良好的外推性。

## Transformer 完整架构

```
输入 → [Embedding + Positional Encoding]
    → [Multi-Head Self-Attention]
    → [Add & Norm]
    → [Feed-Forward Network]
    → [Add & Norm]
    → 输出
```

### Encoder-Decoder vs Decoder-Only

| 架构类型 | 代表模型 | 特点 |
|---------|---------|------|
| Encoder-Decoder | T5, BART | 适合 seq2seq 任务 |
| Encoder-Only | BERT | 适合理解任务 |
| Decoder-Only | GPT 系列 | 因果注意力掩码，自回归生成 |

::: tip 关键演进
现代 LLM 几乎全部采用 Decoder-Only 架构，因为因果注意力掩码天然适配自回归语言建模，且训练效率最高。
:::
