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

## 注意力机制详细公式与代码

以下用 **batch_size = B, 序列长度 = S (或 n), 头数 = H, 头维度 = d, 总维度 D = H × d**。

---

### 1. 标准缩放点积注意力

**数学定义**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}}\right) V
$$

**推导**：

点积 $QK^T$ 的每个元素是 $d$ 维向量的内积。当 $d$ 很大时，内积方差 $\approx d$（假设 Q,K 独立标准正态）。Softmax 对大值极度敏感，除以 $\sqrt{d}$ 将方差拉回 1，防止梯度消失。

**完整 PyTorch 实现**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (B, H, S, d)  — 已经过线性投影并 reshape 为多头
    返回:    (B, H, S, d)
    """
    d_k = Q.size(-1)
    # (B, H, S, d) × (B, H, d, S) → (B, H, S, S)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)    # (B, H, S, S)
    output = torch.matmul(attn_weights, V)       # (B, H, S, d)
    return output, attn_weights
```

**复杂度分析**：矩阵乘法 $QK^T$ 产生 $(B, H, S, S)$ 中间张量，计算量 $O(B \cdot H \cdot S^2 \cdot d)$，显存 $O(B \cdot H \cdot S^2)$。这正是长序列推理的瓶颈所在。

---

### 2. 多头注意力 (MHA)

**数学定义**：

$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

**直觉**：不同头关注不同子空间 —— 有的头关注句法（主谓关系），有的关注语义（指代消解），有的关注位置（相邻词模式）。

**完整 PyTorch Module**：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # QKV 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """(B, S, D) → (B, H, S, d)"""
        B, S, _ = x.shape
        x = x.view(B, S, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """(B, H, S, d) → (B, S, D)"""
        B, _, S, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, S, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. 线性投影
        Q = self.W_Q(Q)  # (B, S, D)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. 拆分为多头
        Q = self.split_heads(Q)  # (B, H, S, d)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 缩放点积注意力
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 4. 合并多头 + 输出投影
        output = self.combine_heads(attn_output)  # (B, S, D)
        return self.W_O(output), attn_weights
```

---

### 3. 因果注意力 (Causal Mask)

**原理**：确保位置 $t$ 只能 attend 到 $1 \dots t$（包括自身），实现自回归生成的 Teacher Forcing。

**数学定义**：

$$
\text{CausalAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}} + M_{\text{causal}}\right) V
$$

其中 $M_{\text{causal}}[i,j] = -\infty$ 当 $j > i$，否则 $0$。

**掩码生成 + 可视化**：

```python
def causal_mask(seq_len):
    """
    生成因果掩码
    返回 (1, 1, seq_len, seq_len) — 适配 (B, H, S, S)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, 0.0)
    return mask.unsqueeze(0).unsqueeze(0)

# 可视化因果掩码 (seq_len=5)
#     ┌──────────────────────┐
#  0  │  0  -∞  -∞  -∞  -∞  │  ← token_0 只能看到自己
#  1  │  0   0  -∞  -∞  -∞  │
#  2  │  0   0   0  -∞  -∞  │
#  3  │  0   0   0   0  -∞  │
#  4  │  0   0   0   0   0  │
#     └──────────────────────┘

def causal_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    scores = scores + causal_mask(Q.size(-2)).to(Q.device)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)
```

**KV Cache 推理**：

```python
# 自回归推理时，每步只算新 token 的 Q，复用历史 KV
def decode_step(x_new, past_k, past_v, Q_proj, K_proj, V_proj):
    # x_new: (B, 1, D) — 只输入新 token
    q = Q_proj(x_new)                  # (B, 1, D)
    k = K_proj(x_new)                  # (B, 1, D)
    v = V_proj(x_new)

    # 追加到历史 KV
    k = torch.cat([past_k, k], dim=1)  # (B, S_past+1, D)
    v = torch.cat([past_v, v], dim=1)

    # 注意力（无需因果掩码，因为 Q 只有1个 token，天然因果）
    return scaled_dot_product_attention(q, k, v), k, v
```

---

### 4. 交叉注意力 (Cross-Attention)

**数学定义**：

$$
\text{CrossAttn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\!\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d}}\right) V_{\text{enc}}
$$

$Q$ 来自 Decoder（查询「我需要什么信息」），$K,V$ 来自 Encoder（提供「源序列的全部知识」）。

**完整实现**：

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_model)  # Decoder → Q
        self.W_K = nn.Linear(d_model, d_model)  # Encoder → K
        self.W_V = nn.Linear(d_model, d_model)  # Encoder → V
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, dec_hidden, enc_output, enc_mask=None):
        """
        dec_hidden: (B, S_dec, D) — Decoder 当前状态
        enc_output: (B, S_enc, D) — Encoder 完整输出
        """
        Q = self.split_heads(self.W_Q(dec_hidden))  # (B, H, S_dec, d)
        K = self.split_heads(self.W_K(enc_output))  # (B, H, S_enc, d)
        V = self.split_heads(self.W_V(enc_output))

        # Q 和 K/V 可以来自不同序列长度
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if enc_mask is not None:
            # enc_mask: (B, S_enc), True 表示 padding
            scores = scores.masked_fill(enc_mask[:, None, None, :], float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return self.combine_heads(out), attn
```

**Decoder Layer 完整结构**：

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, causal_mask, enc_mask):
        # 1. 因果自注意力
        x = self.norm1(x + self.self_attn(x, x, x, causal_mask)[0])
        # 2. 交叉注意力
        x = self.norm2(x + self.cross_attn(x, enc_output, enc_mask)[0])
        # 3. FFN
        x = self.norm3(x + self.ffn(x))
        return x
```

---

### 5. GQA / MQA

**核心思想**：减少 KV 头的数量来降低 KV Cache 显存。Q 头保持 $H$ 个，KV 头只有 $G$ 个（$G \ll H$）。

**数学本质**：

$$
\text{head}_i = 
\begin{cases}
\text{Attention}(Q_i, K_i, V_i) & \text{MHA (G = H)} \\[4pt]
\text{Attention}(Q_i, K_{\lfloor iG/H \rfloor}, V_{\lfloor iG/H \rfloor}) & \text{GQA (1 < G < H)} \\[4pt]
\text{Attention}(Q_i, K_0, V_0) & \text{MQA (G = 1)}
\end{cases}
$$

**分组映射关系**：

```
H=8, G=2 时的映射:
  Q_0, Q_1, Q_2, Q_3 → KV_0  (前 4 个 Q 头共享第 0 组 KV)
  Q_4, Q_5, Q_6, Q_7 → KV_1  (后 4 个 Q 头共享第 1 组 KV)
```

**完整 PyTorch 实现**：

```python
class GroupedQueryAttention(nn.Module):
    """
    n_heads:    Q 头数 (如 32)
    n_kv_heads: KV 头数 (如 8)，GQA；若为 1 则为 MQA；若等于 n_heads 则为 MHA
    """
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads   # 每个 KV 头被多少个 Q 头共享
        self.d_k = d_model // n_heads

        # Q 投影: 全尺寸 (H × d)
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        # K, V 投影: 缩小的尺寸 (G × d)，节省参数量和 KV Cache
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(n_heads * self.d_k, d_model, bias=False)

        print(f"KV Cache 节省比例: {(n_heads - n_kv_heads) / n_heads * 100:.1f}%")

    def forward(self, x, past_kv=None, use_cache=False):
        B, S, _ = x.shape

        # 投影
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 关键步骤：将 KV 头从 G 份复制到 H 份
        # repeat_interleave: [0,1,2,3] × rep=4 → [0,0,0,0, 1,1,1,1, ...]
        K = K.repeat_interleave(self.n_rep, dim=1)  # (B, G, S, d) → (B, H, S, d)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # 标准注意力
        d_k = self.d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # 合并头
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out)
```

**参数量对比 (d_model=4096, H=32)**：

| 方案 | G | Q 参数 | KV 参数 | KV Cache/token |
|------|---|--------|---------|----------------|
| MHA | 32 | 4096×4096 | 2 × 4096×4096 | 32 × 128 = 4096 |
| GQA | 8 | 4096×4096 | 2 × 4096×1024 | 8 × 128 = 1024 |
| MQA | 1 | 4096×4096 | 2 × 4096×128 | 1 × 128 = 128 |

::: tip 主流选择
LLaMA 2 70B 用 GQA (G=8)，Mistral 用 GQA + 滑动窗口注意力。纯 MQA 在最新模型中已经不太常见，GQA 是当前最优平衡点。
:::

---

### 6. 滑动窗口注意力 (SWA)

Mistral 7B 的关键创新。每个 Token 只看前后 $W$ 个邻居。

**数学定义**：

$$
\text{SWA}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}} + M_{\text{sw}}\right) V
$$

$$
M_{\text{sw}}[i,j] = 
\begin{cases}
0, & |i - j| \leq W \\
-\infty, & \text{otherwise}
\end{cases}
$$

**复杂度**：$O(S \times W)$ 而非 $O(S^2)$，$W$ 通常为 4096。

**掩码与实现**：

```python
def sliding_window_mask(seq_len, window_size):
    """生成 SWA 掩码"""
    idx = torch.arange(seq_len)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()  # (S, S)
    mask = dist <= window_size
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=4096):
        super().__init__()
        self.window_size = window_size
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads=8)

    def forward(self, x):
        B, S, _ = x.shape
        Q = self.attn.W_Q(x).view(B, S, self.attn.n_heads, -1).transpose(1, 2)
        K = self.attn.W_K(x).view(B, S, self.attn.n_kv_heads, -1).transpose(1, 2)
        V = self.attn.W_V(x).view(B, S, self.attn.n_kv_heads, -1).transpose(1, 2)

        K = K.repeat_interleave(self.attn.n_rep, dim=1)
        V = V.repeat_interleave(self.attn.n_rep, dim=1)

        d_k = self.attn.d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        # 仅保留窗口内
        mask = sliding_window_mask(S, self.window_size).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.attn.W_O(out)
```

**环形缓冲区 (Ring Buffer) 加速**：

```python
class SWAWithRingBuffer:
    """
    每步推理只更新滑动窗口内的 KV，而非全部。
    """
    def __init__(self, max_window=4096):
        self.k_cache = None  # (B, H, max_window, d)
        self.v_cache = None
        self.write_idx = 0
        self.max_window = max_window

    def step(self, q_new, k_new, v_new):
        """q_new: (B, H, 1, d), k_new/v_new: (B, H, 1, d)"""
        if self.k_cache is None:
            self.k_cache = torch.zeros(q_new.shape[0], q_new.shape[1],
                                        self.max_window, q_new.shape[-1])
            self.v_cache = torch.zeros_like(self.k_cache)

        # 环形写入，覆写最旧的位置
        pos = self.write_idx % self.max_window
        self.k_cache[:, :, pos:pos+1, :] = k_new
        self.v_cache[:, :, pos:pos+1, :] = v_new
        self.write_idx += 1

        # 取有效窗口
        valid_len = min(self.write_idx, self.max_window)
        scores = torch.matmul(q_new, self.k_cache[:, :, :valid_len].transpose(-2, -1))
        scores = scores / (q_new.shape[-1] ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, self.v_cache[:, :, :valid_len])
```

---

### 7. 稀疏注意力

**Longformer 的三种模式**：

```
位置 0:  [G][G][W][W][ ][ ][ ][ ][ ][R]   G = Global (全局)
位置 1:  [G][G][W][W][ ][ ][ ][ ][ ][ ]   W = Window (窗口)
位置 2:  [ ][W][W][W][W][ ][ ][ ][ ][ ]   R = Random (随机)
位置 3:  [ ][W][W][W][W][W][ ][ ][ ][ ]
...
```

**Longformer 实现**：

```python
class LongformerAttention(nn.Module):
    """
    组合三种注意力模式：
    - 滑动窗口: 每个 token 关注邻近 W 个
    - 全局注意力: 指定位置的 token 关注所有 token，也被所有 token 关注
    - 随机: 少量随机连接提供长程信息通路
    """
    def __init__(self, d_model, n_heads, window_size=512, global_tokens=1):
        super().__init__()
        self.window = window_size
        self.global_tokens = global_tokens  # 前 N 个 token 为全局 token (如 CLS)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def _make_sparse_mask(self, seq_len):
        """构建 Longformer 稀疏注意力的 mask 矩阵"""
        # 滑动窗口部分
        row = torch.arange(seq_len).unsqueeze(1)
        col = torch.arange(seq_len).unsqueeze(0)
        window_mask = (row - col).abs() <= self.window

        # 全局 token (前 global_tokens 个) — 全可见
        global_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        global_mask[:self.global_tokens, :] = True       # 全局 tokens 看所有
        global_mask[:, :self.global_tokens] = True       # 所有 token 看全局 tokens

        return (window_mask | global_mask).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, S, _ = x.shape
        Q = self.split_heads(self.W_Q(x))
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        mask = self._make_sparse_mask(S).to(x.device)
        scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return self.combine_heads(out)
```

---

### 8. FlashAttention 系列

**核心洞察**：不改变数学结果，只改变计算顺序。传统实现将 $S \times S$ 注意力矩阵写回 HBM（慢），FlashAttention 在 SRAM（快）中分块完成计算。

**GPU 内存层级**：

```
HBM — 80GB, 1.5 TB/s  ← 完整注意力矩阵寄居于此（慢）
  ↕ 数据传输
SRAM — 20MB, 19 TB/s  ← FlashAttention 实际计算发生于此（快）
```

**在线 Softmax 算法** — FlashAttention 的核心数学组件：

标准 softmax 需要两遍扫描（先找 max，再 exp sum），但分块计算时每块只能看到局部数据。解决方案是维护运行中的 max 和 sum，每处理一个新块时修正旧结果。

```python
def online_softmax_forward(blocks):
    """
    分块计算 softmax，等价于全局 softmax 但只需一次遍历。
    blocks: list of (B, H, B_r, d) tensors — 按行分块的 QK^T 结果
    """
    m_prev = None      # 全局最大值（运行中）
    l_prev = None      # 全局 exp 和（运行中）
    O = None           # 累积输出

    for i, S_block in enumerate(blocks):  # S_block: 第 i 个分块， (B, H, B_r, B_c)
        m_curr = S_block.max(dim=-1, keepdim=True).values  # 当前块的最大值

        if m_prev is None:
            m_prev = m_curr
            l_prev = torch.exp(S_block - m_curr).sum(dim=-1, keepdim=True)
            O = torch.exp(S_block - m_curr)  # O 暂存未归一化的 P
        else:
            # 修正因子：因为全局最大值更新了，旧 exp 需要重新标定
            correction = torch.exp(m_prev - m_curr)
            l_new = torch.exp(S_block - m_curr).sum(dim=-1, keepdim=True)
            l_prev = correction * l_prev + l_new

            # 修正旧输出
            O = correction * O + torch.exp(S_block - m_curr)
            m_prev = m_curr

    # 最终归一化
    O = O / l_prev
    return O
```

**FlashAttention 完整算法伪代码**：

```python
def flash_attention(Q, K, V, B_r=128, B_c=128):
    """
    分块计算精确注意力 (Tiling)
    Q, K, V: (B, H, N, d)
    B_r: 输出分块大小  B_c: K/V 分块大小
    """
    B, H, N, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    # 输出和 softmax 辅助变量
    O = torch.zeros_like(Q)
    L = torch.zeros(B, H, N, 1, device=Q.device)  # row-wise sum of exp
    m = torch.full((B, H, N, 1), float('-inf'), device=Q.device)  # row-wise max

    # 在外循环上分块 Q (按行)
    for i in range(0, N, B_r):
        Q_i = Q[:, :, i:i+B_r, :]                 # 加载 Q 块到 SRAM
        O_i = torch.zeros_like(Q_i)
        m_i = m[:, :, i:i+B_r, :]
        L_i = L[:, :, i:i+B_r, :]

        # 在内循环上分块 K, V (按行)
        for j in range(0, N, B_c):
            K_j = K[:, :, j:j+B_c, :]             # 加载 K 块
            V_j = V[:, :, j:j+B_c, :]             # 加载 V 块

            # 计算局部 S = Q_i @ K_j^T
            S_ij = scale * torch.matmul(Q_i, K_j.transpose(-2, -1))

            # 在线 softmax 更新
            m_ij = S_ij.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m_i, m_ij)

            # 修正权重
            P_ij = torch.exp(S_ij - m_new)
            correction = torch.exp(m_i - m_new)

            O_i = correction * O_i + torch.matmul(P_ij, V_j)
            L_i = correction * L_i + P_ij.sum(dim=-1, keepdim=True)
            m_i = m_new

        # 写回 HBM
        O[:, :, i:i+B_r, :] = O_i / L_i
        L[:, :, i:i+B_r, :] = L_i
        m[:, :, i:i+B_r, :] = m_i

    return O
```

**版本演进**：

| 版本 | 核心优化 | 适用 |
|------|---------|------|
| **FlashAttention 1** | Tiling + Recomputation | 训练 + 推理 |
| **FlashAttention 2** | 减少非 matmul 操作、优化 warp 调度、增加并行度 | 训练 + 推理 |
| **FlashAttention 3** | 利用 Hopper FP8、TMA 异步拷贝、WGMMA 指令 | H100 推理 |

::: tip 实战建议
生产环境中安装 `flash-attn` Python 包即可自动获得 2-3× 加速。vLLM 等推理框架已将 FlashAttention 设为默认。自建服务时如遇安装困难，可降级使用 PyTorch 2.0+ 内置的 `F.scaled_dot_product_attention`，它在支持时会自动调用 FlashAttention 后端。
:::

---

### 9. MLA (Multi-head Latent Attention)

DeepSeek-V2/V3 的核心创新。将 KV 投影到一个低秩潜在空间（latent space），再从潜在空间上投回注意力维度。

**数学原理**：

传统 KV 存储占显存主导。MLA 用两步压缩：

$$
c_t^{KV} = W^{DKV} \cdot h_t \quad \text{(降维: D → d_c)}
$$

$$
k_t = W^{UK} \cdot c_t^{KV} \quad \text{(升维: d_c → D)}
$$

$$
v_t = W^{UV} \cdot c_t^{KV} \quad \text{(升维: d_c → D)}
$$

推理时只缓存压缩后的 $c_t^{KV}$（维度 $d_c \ll D$），使用时再上投影。

**完整实现**：

```python
class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2 的 MLA 简化实现。
    d_latent << d_model  →  KV Cache 大幅压缩。
    """
    def __init__(self, d_model=5120, n_heads=128, d_latent=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent = d_latent  # 压缩后的潜在维度
        self.d_k = d_model // n_heads

        # Q 投影 (标准)
        self.W_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)

        # KV 压缩：d_model → d_latent
        self.W_DKV = nn.Linear(d_model, d_latent, bias=False)
        # KV 解压缩：d_latent → n_heads × d_k  (共享一个上投影)
        self.W_UK = nn.Linear(d_latent, n_heads * self.d_k, bias=False)
        self.W_UV = nn.Linear(d_latent, n_heads * self.d_k, bias=False)

        self.W_O = nn.Linear(n_heads * self.d_k, d_model, bias=False)

        kv_original = n_heads * self.d_k
        kv_compressed = d_latent
        print(f"KV Cache 压缩比: {kv_compressed}/{kv_original} = "
              f"{kv_compressed/kv_original:.2%} "
              f"(节省 {(1 - kv_compressed/kv_original)*100:.1f}%)")

    def forward(self, x, past_c_kv=None, use_cache=False):
        B, S, _ = x.shape

        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # KV 压缩到潜在空间
        c_kv = self.W_DKV(x)  # (B, S, d_latent)

        if use_cache and past_c_kv is not None:
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)

        # 从潜在空间上投影
        K = self.W_UK(c_kv).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_UV(c_kv).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        d_k = self.d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out)
```

**MLA vs GQA 对比**：

| 方案 | KV Cache/token | 额外计算 | 质量 |
|------|---------------|---------|------|
| MHA | H × d (大) | 无 | 基准 |
| GQA | G × d (中) | 无 (仅结构变化) | 接近 MHA |
| MLA | d_latent (极小) | 上投影矩阵乘法 (少量) | 接近 MHA |

MLA 典型配置：d_model=5120, H=128, d=40, d_latent=512 → 压缩前 5120，压缩后 512，**节省 90%**。

---

### 10. 线性注意力

**核技巧 (Kernel Trick)**：将 softmax 替换为可分解的核函数，改变计算顺序。

传统注意力的瓶颈在于必须先算 $QK^T$ 产生 $S \times S$ 矩阵。线性注意力的洞察：

$$
\text{Attention}(Q, K, V) = \frac{\phi(Q) \left( \phi(K)^T V \right)}{\phi(Q) \left( \phi(K)^T \mathbf{1} \right)}
$$

先算 $\phi(K)^T V$（代价 $O(S \cdot d^2)$），再乘以 $\phi(Q)$（代价 $O(S \cdot d^2)$），总复杂度 $O(S \cdot d^2)$ 而非 $O(S^2 \cdot d)$。

当 $S \gg d$ 时（长序列推理的常态），这是巨大的收益。

**完整实现**：

```python
class LinearAttention(nn.Module):
    """
    线性注意力：将 O(S²·d) 降至 O(S·d²)。
    适合长序列 (S >> d) 场景。
    """
    def __init__(self, d_model, n_heads, eps=1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.eps = eps

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # 关键：核函数 φ (ELU + 1 保证非负)
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # 线性注意力核心：先算 K^T V (O(S·d²))
        # K: (B,H,S,d), V: (B,H,S,d) → K^T·V: (B,H,d,d)
        KV = torch.matmul(K.transpose(-2, -1), V)

        # 归一化项：K^T·1 → (B,H,d,1)
        K_sum = K.sum(dim=2, keepdim=True).transpose(-2, -1)  # (B,H,d,1)

        # 除法形式：φ(Q) · (φ(K)^T V) / (φ(Q) · (φ(K)^T 1))
        num = torch.matmul(Q, KV)                               # (B,H,S,d)
        den = torch.matmul(Q, K_sum) + self.eps               # (B,H,S,1)
        out = num / den

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out)
```

**因果版本的线性注意力**：因果约束下同样可以保持 $O(S \cdot d^2)$ 复杂度，通过递推累积状态：

```python
def causal_linear_attention(Q, K, V):
    """
    因果 + 线性: 递推维护累积的 KV 状态
    O(S·d²) 而非 O(S²·d)
    """
    B, H, S, d = Q.shape
    Q = F.elu(Q) + 1
    K = F.elu(K) + 1

    outputs = []
    kv_state = torch.zeros(B, H, d, d, device=Q.device)  # (B,H,d,d)
    k_state  = torch.zeros(B, H, d, 1, device=Q.device)  # (B,H,d,1)

    for t in range(S):
        q_t = Q[:, :, t:t+1, :]     # (B, H, 1, d)
        k_t = K[:, :, t:t+1, :]     # (B, H, 1, d)
        v_t = V[:, :, t:t+1, :]

        # 递推更新累积状态
        kv_state = kv_state + torch.matmul(k_t.transpose(-2, -1), v_t)
        k_state  = k_state  + k_t.transpose(-2, -1)

        num = torch.matmul(q_t, kv_state)
        den = torch.matmul(q_t, k_state) + 1e-6
        outputs.append(num / den)

    return torch.cat(outputs, dim=2)
```

**线性注意力 vs Mamba vs RWKV**：

| 方案 | 思路 | 复杂度 |
|------|------|--------|
| **Linear Transformer** | Softmax → 核函数特征映射 | $O(S \cdot d^2)$ |
| **Mamba / Mamba-2** | 注意力 → 结构化状态空间模型 (SSM) | $O(S \cdot d \cdot N)$ |
| **RWKV** | 注意力 → RNN 式的线性递归 | $O(S \cdot d^2)$ |

---

## 注意力机制选型速查

```
序列 < 4K，追求最高精度   → 标准 MHA + FlashAttention
序列 > 8K，低延迟要求     → GQA + FlashAttention-2
序列 > 32K，KV Cache 受限 → MLA (DeepSeek 方案) 或 GQA + SWA
长文档预填充优化          → FlashAttention-3 (H100)
极致长序列 + 边缘设备     → 线性注意力 / Mamba
LMM 多模态交互            → Cross-Attention (Encoder-Decoder)
KV Cache 显存紧张         → MQA (牺牲少量质量) → GQA → MLA (最佳)
训练大 batch 加速         → FlashAttention-2
安全性要求高 (无随机性)   → 确定性注意力 + 因果掩码
```
