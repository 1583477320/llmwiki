# Transformer 分类与注意力机制全景

> 从 Vanilla Transformer 到 FlashAttention —— 架构变体与注意力机制的演进图谱

---

## Transformer 三大架构分支

### 架构演进树

```
                     Vanilla Transformer (2017)
                     "Attention Is All You Need"
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    Encoder-Only         Decoder-Only       Encoder-Decoder
    (双向注意力)           (因果注意力)         (交叉注意力)
          │                   │                   │
    ┌─────┴──────┐      ┌────┴────────┐     ┌────┴─────┐
    │ BERT       │      │ GPT 系列     │     │ T5       │
    │ RoBERTa    │      │ LLaMA       │     │ BART     │
    │ DeBERTa    │      │ Mistral     │     │ mT5      │
    │ ALBERT     │      │ Qwen        │     │ Flan-T5  │
    │ ELECTRA    │      │ DeepSeek    │     │ BLOOM    │
    └────────────┘      └─────────────┘     └──────────┘
```

### 各分支核心差异

| 维度 | Encoder-Only | Decoder-Only | Encoder-Decoder |
|------|-------------|-------------|-----------------|
| **注意力掩码** | 双向（全可见） | 因果掩码（仅左侧） | 编码器双向 + 解码器因果 + 交叉注意力 |
| **预训练目标** | MLM (Masked LM) | CLM (Causal LM) | Span Corruption / Seq2Seq |
| **擅长任务** | 理解（分类、NER、抽取） | 生成（对话、补全、创作） | 翻译、摘要、问答 |
| **代表模型** | BERT, RoBERTa, DeBERTa | GPT, LLaMA, Mistral | T5, BART |
| **推理方式** | 一次性编码 | 自回归逐个生成 | 编码后自回归生成 |
| **当前生态** | 被 Decoder-Only 挤压 | 绝对主流 | 专用任务仍有价值 |

---

## Encoder-Only 模型详解

### BERT 系列

| 模型 | 参数量 | 创新点 |
|------|--------|--------|
| **BERT** | 110M / 340M | 双向 MLM 预训练 + NSP |
| **RoBERTa** | 125M / 355M | 动态掩码、更大 batch、去除 NSP |
| **ALBERT** | 12M / 18M | 跨层参数共享、分解嵌入矩阵 |
| **ELECTRA** | 14M / 110M | GAN 式训练：生成器替换 Token，判别器判断 |
| **DeBERTa** | 100M–1.5B | 解耦注意力（内容 + 位置）+ 增强掩码解码器 |
| **ModernBERT** | 139M / 395M | 2024 年重新设计，支持 8K Context，FlashAttention |

### 为什么 Encoder-Only 在 Agent 时代边缘化？

1. **无法自回归生成**：天然不能用于对话和推理链
2. **任务碎片化**：每个下游任务需要添加分类头微调
3. **Decoder-Only 的通用性**：一套模型同时做理解与生成

但仍有两个存在价值：
- **Embedding 模型**：BERT 系仍是文本嵌入的主流选择
- **轻量分类**：在固定标签集上的分类任务中，Encoder-Only 更快更省

---

## Decoder-Only 模型演进

### 关键代际

| 代际 | 时间 | 代表模型 | 标志性创新 |
|------|------|---------|-----------|
| **1.0** | 2018–2020 | GPT, GPT-2 | 证明了 LM 的 Zero-shot 能力 |
| **2.0** | 2020–2022 | GPT-3, Codex | 涌现能力、In-Context Learning |
| **3.0** | 2023 | LLaMA, Mistral | 开源追上闭源、高效架构设计 |
| **3.5** | 2024 | Mixtral, DeepSeek-V2 | MoE 架构、Multi-head Latent Attention |
| **4.0** | 2025 | DeepSeek-V3, Qwen3 | 超大规模 MoE、推理时缩放 |

### 核心架构组件差异

| 组件 | GPT-3/4 | LLaMA 2/3 | Mistral | Qwen2.5 |
|------|---------|----------|---------|---------|
| **激活函数** | GeLU | SiLU (SwiGLU) | SiLU (SwiGLU) | SiLU (SwiGLU) |
| **位置编码** | Learned | RoPE | RoPE | RoPE |
| **归一化** | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| **注意力** | MHA | GQA | GQA / SWA | GQA |
| **FFN 结构** | FFN | SwiGLU FFN | SwiGLU FFN | SwiGLU FFN |
| **MoE** | 推测有 | 无 | Mixtral 有 MoE | 部分模型有 MoE |

---

## 注意力机制完整分类

### 1. 标准缩放点积注意力

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

复杂度：$O(n^2 \cdot d)$，其中 $n$ 为序列长度，$d$ 为头维度。

### 2. 多头注意力 (MHA)

```python
# 标准 MHA：每个头独立计算完整的注意力
head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)
output = Concat(head_1, ..., head_h) @ W^O
```

### 3. 因果注意力 (Causal Attention)

Decoder-Only 的核心掩码方式。对注意力矩阵的上三角区域置 $-\infty$，确保位置 $t$ 只能看到 $1 \dots t$：

```python
mask[i, j] = -inf if j > i else 0
attention_weights = softmax(QK^T / sqrt(d_k) + mask)
```

### 4. 交叉注意力 (Cross-Attention)

Encoder-Decoder 架构的精髓。Q 来自 Decoder（当前要生成的），K/V 来自 Encoder（源序列的完整编码）：

```
Decoder 的每一层：
  Self-Attention (因果掩码) → Cross-Attention (Q=decoder, K/V=encoder) → FFN
```

在 LMM 中，Cross-Attention 被用于视觉 Token 与文本 Token 的交互。

### 5. GQA / MQA（分组查询 / 多查询注意力）

现代 Decoder-Only 模型的核心优化。减少 KV 头的数量，在效果和速度间取得平衡。

```
MHA： H 个 Q 头，H 个 KV 头  (1:1，最大表达力，最大显存)
GQA： H 个 Q 头，G 个 KV 头  (H/G 共享，平衡方案)
MQA： H 个 Q 头，1 个 KV 头  (H:1，最小显存，轻微质量损失)
```

| 方案 | Q 头数 | KV 头数 | 显存 | 质量 |
|------|--------|---------|------|------|
| MHA | H | H | 高 | 基准 |
| GQA | H | G (如 4, 8) | 中 | 接近 MHA |
| MQA | H | 1 | 低 | 轻微下降 |

::: tip 主流选择
LLaMA 2 70B 用 GQA (G=8)，Mistral 用 GQA + 滑动窗口注意力。纯 MQA 在最新的模型中已经不太常见，GQA 是当前最优平衡点。
:::

### 6. 滑动窗口注意力 (SWA)

Mistral 的关键创新。每个 Token 只关注其前后固定窗口内的 Token：

```python
# 复杂度从 O(n²) 降到 O(n × W)，W 为窗口大小
attention_mask[i, j] = 0 if |i - j| <= window_size else -inf
```

**优势**：
- 推理复杂度与序列长度线性增长
- 搭配环形缓冲区可缓存 KV，减少重复计算
- 长文档场景下收益尤为显著

### 7. 稀疏注意力

使注意力矩阵呈现结构化稀疏，减少计算量：

| 方案 | 模式 | 代表 |
|------|------|------|
| **Longformer** | 滑动窗口 + 全局 Token + 随机 | Longformer |
| **BigBird** | 随机 + 窗口 + 全局 | BigBird |
| **Sparse Transformer** | 跨步注意力 + 固定模式 | Sparse Transformer |

### 8. FlashAttention 系列

不改变注意力计算的数学结果，而是改变计算顺序以匹配 GPU 内存层级。

| 版本 | 核心优化 | 适用场景 |
|------|---------|---------|
| **FlashAttention 1** | Tiling + Recomputation，避免具体化完整注意力矩阵到 HBM | 训练 + 推理 |
| **FlashAttention 2** | 减少非矩阵乘操作、优化线程块调度、更好的并行策略 | 训练 + 推理 |
| **FlashAttention 3** | 利用 Hopper 架构的 FP8 和异步执行，对 H100 专项优化 | H100 推理 |

```
传统注意力：
  QK^T → [n×n 矩阵写入 HBM] → Softmax → ×V → 读回
  问题：n×n 矩阵读写 HBM 是主要瓶颈

FlashAttention：
  分块计算 Softmax（Tiling）→ 每个 Block 仅在 SRAM 中计算
  → 不存储完整的 n×n 中间矩阵
  → 显存从 O(n²) 降至 O(n)
```

### 9. MLA (Multi-head Latent Attention)

DeepSeek-V2/V3 的核心创新。将 KV 压缩到低秩潜在空间来大幅减少 KV Cache：

```
传统：KV 维度 = H × d_head（如 128 × 128 = 16384）
MLA： 压缩后 KV 维度 = latent_dim（如 512）+ 上投影矩阵
```

**效果**：KV Cache 减少 93%，推理吞吐大幅提升。

### 10. 线性注意力

将标准 Softmax 注意力的 $O(n^2)$ 降至 $O(n)$：

| 方案 | 思路 | 代表 |
|------|------|------|
| **Linear Transformer** | 将 Softmax 替换为核函数特征映射 | Linear Transformer |
| **Mamba / Mamba-2** | 将注意力替换为状态空间模型 (SSM) | Mamba, Jamba |
| **RWKV** | 将注意力替换为 RNN 式的线性递归 | RWKV |

```
标准注意力：O(n² · d)      — 全局交互，无法逃避的二次复杂度
线性注意力：O(n · d²)      — 核技巧：先算 K^T V 再乘以 Q
Mamba/SSM： O(n · d × state) — 完全丢弃注意力，用结构化状态空间
```

---

## FlashAttention 实现要点

### 在线 Softmax (Tiling)

传统 Softmax 需要全量输入才能计算。FlashAttention 的关键数学洞察 —— Softmax 可以通过维护 a max 和 sum 来增量计算：

```python
# 分块 Softmax 的伪代码
for block in Q_blocks:
    m_i = max(m_{i-1}, row_max(block))           # 更新最大值
    sum_i = exp(m_{i-1} - m_i) * sum_{i-1} + ... # 修正求和
    # 最终 softmax = exp(block - m_final) / sum_final
```

### GPU 内存层级

```
HBM (High Bandwidth Memory) — 80GB, 慢 (1.5 TB/s)
    ↕
SRAM (Shared Memory) — 20MB, 快 (19 TB/s)
```

FlashAttention 的策略：**在 SRAM 中完成所有计算，只将最终结果写回 HBM**。

::: tip 实战建议
在生产环境中，启用 FlashAttention 通常是免费的性能提升。对于 vLLM 等推理框架，FlashAttention 已经是默认使用。自建推理服务时，安装 `flash-attn` 包即可获得约 2-3× 的注意力计算加速。
:::

---

## 注意力机制选型速查

```
序列 < 4K，追求最高精度   → 标准 MHA + FlashAttention
序列 > 8K，低延迟要求     → GQA + FlashAttention-2
序列 > 32K，KV Cache 受限 → MLA (DeepSeek 方案) 或 MQA + SWA
长文档预填充优化          → FlashAttention-3 (H100)
极致长序列 + 边缘设备     → 线性注意力 / Mamba
LMM 多模态交互            → Cross-Attention (Encoder-Decoder)
```
