# LLM 核心原理

> Token 机制、嵌入、自回归建模、Context 窗口、采样控制与训练流程的完整拆解

---

## 1. Token 与 Tokenization

LLM 不直接处理文本，而是处理 **Token**（词元）。Tokenization 是将文本切分为 Token 序列的过程，决定了模型的「词汇量」和每段文本的「编码密度」。

### 1.1 BPE 分词算法

BPE（Byte Pair Encoding）是 GPT 系列使用的核心分词算法。

**算法流程**：

```
输入文本 → 字符级拆分 → 统计所有相邻 token pair 频率
    → 合并频率最高的 pair → 重复直到达到目标词表大小
```

**完整实现**：

```python
from collections import Counter

def train_bpe(texts, vocab_size=50000):
    """
    从头训练 BPE tokenizer。
    texts: list of strings (训练语料)
    vocab_size: 目标词表大小
    """
    # 1. 初始词表：所有 Unicode 字节 + 特殊 token
    vocab = {i: bytes([i]) for i in range(256)}

    # 2. 把文本转为字节序列
    corpus = [text.encode('utf-8') for text in texts]

    # 3. 字符级拆分
    splits = {i: [bytes([b]) for b in data] for i, data in enumerate(corpus)}

    # 4. 迭代合并
    next_id = 256
    while next_id < vocab_size:
        # 统计所有相邻 pair 的出现频率
        pair_counts = Counter()
        for split in splits.values():
            for j in range(len(split) - 1):
                pair_counts[(split[j], split[j+1])] += 1

        if not pair_counts:
            break

        best_pair = pair_counts.most_common(1)[0][0]

        # 合并
        vocab[next_id] = best_pair[0] + best_pair[1]

        for split in splits.values():
            i = 0
            while i < len(split) - 1:
                if (split[i], split[i+1]) == best_pair:
                    split[i:i+2] = [vocab[next_id]]
                i += 1

        next_id += 1

    return vocab
```

**编码与解码示例**：

```python
# 使用 tiktoken (OpenAI 的 tokenizer)
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 使用的编码

# 编码
text = "Hello, how are you? 你好世界！"
tokens = enc.encode(text)
print(tokens)
# → [9906, 11, 1268, 527, 499, 30, 57668, 53901, 3574, 244, 98220, 6447] (GPT-4 风格)

# 解码
decoded = enc.decode(tokens)
print(decoded)  # → "Hello, how are you? 你好世界！"

# 压缩率对比
print(f"字符数: {len(text)}, Token 数: {len(tokens)}, 压缩率: {len(text)/len(tokens):.1f}")
# 英文: ~4 字符/token, 中文: ~1.5 字符/token
```

### 1.2 主流分词算法对比

| 算法 | 代表模型 | 合并策略 | 特点 |
|------|---------|---------|------|
| **BPE** | GPT-2/3/4, LLaMA | 频率最高优先 | 从字符开始，逐步构建子词 |
| **WordPiece** | BERT | 似然增益最大化 | 选择使训练语料概率增加最多的 pair |
| **SentencePiece** | LLaMA, T5, Gemma | BPE / Unigram | 语言无关，以 Unicode 码点为单位 |
| **Unigram** | XLNet, ALBERT | 概率最大化 | 从大词表逐步削减小概率条目 |

### 1.3 特殊 Token

```
[BOS] / <s>         — 序列开始 (Beginning of Sequence)
[EOS] / </s>        — 序列结束 (End of Sequence)
[PAD]               — 填充 (Padding, batch 内对齐)
[UNK]               — 未知 (Unknown)
[MASK]              — 掩码 (BERT 预训练用)
<|user|> / <|assistant|> — 对话角色分隔 (ChatML 格式)
<|tool_call|>       — 工具调用 (Tool Call 标记)
<|im_start|> / <|im_end|> — Qwen 系列的对话边界标记
```

**Token 预算对 Agent 的影响**：一次 ReAct 循环约消耗 2K-10K tokens。在 128K Context 窗口中，理论可以执行约 10-50 轮 Agent 循环，实际需给重要信息留余量。

---

## 2. 嵌入层

### 2.1 Token Embedding

Token ID 通过嵌入矩阵映射为稠密向量：

$$
\mathbf{e}_i = E[t_i], \quad E \in \mathbb{R}^{V \times D}
$$

其中 $V$ = 词表大小（50K–256K），$D$ = 模型维度（512–8192）。

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Xavier 初始化 (对训练稳定性很重要)
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)

    def forward(self, token_ids):
        # token_ids: (B, S)
        # 输出: (B, S, D)
        return self.embedding(token_ids) * (self.d_model ** 0.5)
        # 乘以 sqrt(d_model) 保持方差稳定 (Transformer 惯例)
```

### 2.2 RoPE 旋转位置编码

RoPE（Rotary Position Embedding）是现代 LLM 最广泛使用的位置编码方案，通过旋转矩阵将位置信息注入注意力计算。

**数学定义**：

给定位置 $pos$ 和维度对 $(2i, 2i+1)$，定义旋转频率：

$$
\theta_i = \text{base}^{-2i/d}, \quad \text{base} = 10000 \text{ (LLaMA 使用 1000000)}
$$

位置 $pos$ 的旋转矩阵作用于向量 $(x_{2i}, x_{2i+1})$：

$$
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} =
\begin{pmatrix}
\cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\
\sin(pos \cdot \theta_i) &  \cos(pos \cdot \theta_i)
\end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

**RoPE 与相对位置的关系**：

两个位置 $m, n$ 的 dot-product attention 分数经 RoPE 后仅依赖相对距离 $(m-n)$：

$$
(R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
$$

这意味着 RoPE 天然编码相对位置，且具备位置插值（外推）能力。

**完整实现**：

```python
def precompute_rope_freqs(dim, max_seq_len, base=10000):
    """预计算所有位置的 cos/sin 旋转值"""
    # theta_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()    # (S,)

    # 外积: (S, 1) × (1, d/2) → (S, d/2)
    freqs = torch.outer(positions, theta)

    cos_cached = torch.cos(freqs)  # (S, d/2)
    sin_cached = torch.sin(freqs)
    return cos_cached, sin_cached


def apply_rotary_emb(x, cos, sin):
    """
    x:   (B, H, S, d) — Q 或 K
    cos/sin: (S, d/2)
    """
    d = x.shape[-1]

    # 将 x 按维度对拆分: (x_0, x_1), (x_2, x_3), ...
    x_reshaped = x.float().reshape(*x.shape[:-1], d // 2, 2)
    x_even = x_reshaped[..., 0]  # (B, H, S, d/2)
    x_odd  = x_reshaped[..., 1]

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, d/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # 旋转
    roped_even = x_even * cos - x_odd * sin
    roped_odd  = x_even * sin + x_odd * cos

    # 合并回原形状
    roped = torch.stack([roped_even, roped_odd], dim=-1)
    return roped.flatten(start_dim=3).to(x.dtype)


class RoPEAttention(nn.Module):
    """带 RoPE 的 GQA 注意力"""
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len=8192):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.W_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(n_heads * self.d_k, d_model, bias=False)

        # 预计算 RoPE 频率
        cos, sin = precompute_rope_freqs(self.d_k, max_seq_len, base=10000)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x, positions=None):
        B, S, _ = x.shape
        if positions is None:
            positions = torch.arange(S, device=x.device)

        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 应用 RoPE 到 Q 和 K (V 不需要)
        rope_cos = self.cos[positions]
        rope_sin = self.sin[positions]
        Q = apply_rotary_emb(Q, rope_cos, rope_sin)
        K = apply_rotary_emb(K, rope_cos, rope_sin)

        # GQA 扩展
        if self.n_heads > self.n_kv_heads:
            n_rep = self.n_heads // self.n_kv_heads
            K = K.repeat_interleave(n_rep, dim=1)
            V = V.repeat_interleave(n_rep, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out)
```

---

## 3. 自回归语言建模

### 3.1 核心目标

LLM 的训练目标是最小化下一个 Token 的负对数似然。

**自回归分解**：将序列 $x = (x_1, x_2, \dots, x_T)$ 的概率分解为条件概率的乘积：

$$
P(x) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

**损失函数（交叉熵）**：

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

其中 $P(x_t \mid x_{<t})$ 由模型最后一层的 softmax 给出。

### 3.2 训练流程架构

```
┌─────────────────────────────────────────────────────┐
│                   训练 Pipeline                       │
│                                                     │
│  Raw Text                                           │
│    │                                                │
│    ▼                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │Tokenizer │───▶│Embedding │───▶│Transformer    │  │
│  │          │    │+ RoPE    │    │Layers × N    │  │
│  └──────────┘    └──────────┘    │              │  │
│                                  │Self-Attn     │  │
│  "The cat"                       │  ↓           │  │
│  → [464, 3797]                   │FFN (SwiGLU)  │  │
│                                  │  ↓           │  │
│                                  │RMSNorm       │  │
│                                  └──────┬───────┘  │
│                                         │          │
│                                         ▼          │
│                                  ┌──────────────┐  │
│                                  │LM Head       │  │
│                                  │(D → V)       │  │
│                                  │Softmax       │  │
│                                  └──────┬───────┘  │
│                                         │          │
│  Target: "sat"                          ▼          │
│  → [6495]                    P(·|"The cat")        │
│                              → CrossEntropy(target) │
└─────────────────────────────────────────────────────┘
```

**Teacher Forcing**：训练时使用真实的前缀（ground truth），而非模型自己的预测。这避免了误差累积，使训练稳定。

### 3.3 完整损失实现

```python
class CausalLM(nn.Module):
    """自回归语言模型训练"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.RMSNorm(d_model)         # 最终 LayerNorm (LLaMA 风格)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying: 嵌入矩阵和 LM Head 共享权重 (减少参数，提升效果)
        self.lm_head.weight = self.token_embed.embedding.weight

    def forward(self, token_ids, targets=None):
        """
        token_ids: (B, S) — 输入 token IDs
        targets:   (B, S) — 目标 token IDs (shifted by 1)
        """
        x = self.token_embed(token_ids)  # (B, S, D)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)         # (B, S, V)

        if targets is not None:
            # 交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100          # 忽略 padding 位置
            )
            return logits, loss

        return logits

# 数据准备
def prepare_batch(texts, tokenizer, max_len):
    """自动构造 input/target (target 是 input 右移一位)"""
    tokens = tokenizer.encode(texts)
    tokens = tokens[:max_len + 1]

    input_ids = tokens[:-1]      # "The cat sat on"
    target_ids = tokens[1:]      # "cat sat on the"

    return torch.tensor(input_ids), torch.tensor(target_ids)
```

### 3.4 因果掩码 + 注意力可视化

```
序列 "The cat sat on the mat" → Token IDs [464, 3797, 6495, 319, 262, 11877]

注意力矩阵 (6×6, 因果掩码):
         The  cat  sat   on  the  mat
  The  [  ■    ·    ·    ·    ·    ·  ]  只有自己
  cat  [  ■    ■    ·    ·    ·    ·  ]  看前2个
  sat  [  ■    ■    ■    ·    ·    ·  ]
   on  [  ■    ■    ■    ■    ·    ·  ]
  the  [  ■    ■    ■    ■    ■    ·  ]
  mat  [  ■    ■    ■    ■    ■    ■  ]  全可见 (自身及前面全部)

每个 token 对应一行，只能看到它左侧 (含自身) 的列。
```

---

## 4. Context 窗口与扩展技术

### 4.1 Context 长度演进

```
2018  GPT-1:   512 tokens  ──────────
2019  GPT-2:  1024 tokens  ────────────────
2020  GPT-3:  2048 tokens  ────────────────────────────────
2023  GPT-4:  8192 → 128K tokens
2024  Claude 3: 200K tokens
2024  Gemini 1.5 Pro: 1M tokens (最终目标 10M)
2025  LLaMA 4: 实测 10M tokens (Google Titans 架构)
```

### 4.2 位置插值 (Position Interpolation)

最直接的长上下文扩展方法。将预训练的位置索引线性压缩到原始训练范围内。

**数学**：训练时模型见过位置 $[0, L_{\text{train}})$。推理时想要 $L_{\text{test}} = \alpha \cdot L_{\text{train}}$。插值将位置缩放 $\alpha$ 倍：

$$
\text{pos}_{\text{new}} = \text{pos}_{\text{orig}} / \alpha
$$

对 RoPE 的影响 — 缩放频率：

$$
\theta_i' = \theta_i / \alpha = \text{base}^{-2i/d} / \alpha
$$

```python
def position_interpolation(cos, sin, scale_factor):
    """
    位置插值: 将位置索引压缩 scale_factor 倍
    cos, sin: 原始形状 (S, d/2)
    返回: 插值后的 (S*scale_factor, d/2)
    """
    new_len = int(cos.shape[0] * scale_factor)
    orig_indices = torch.linspace(0, cos.shape[0] - 1, new_len)

    # 线性插值 cos/sin 值
    cos_new = F.interpolate(
        cos.T.unsqueeze(0), size=new_len, mode='linear', align_corners=True
    ).squeeze(0).T
    sin_new = F.interpolate(
        sin.T.unsqueeze(0), size=new_len, mode='linear', align_corners=True
    ).squeeze(0).T
    return cos_new, sin_new
```

### 4.3 NTK-aware Scaling

Neural Tangent Kernel 视角下的非线性缩放。核心思想：高频维度（低 $i$）对位置更敏感，需要更小的缩放；低频维度（高 $i$）对位置不敏感，可以更大缩放。

**实现**：不改变位置索引，而是调整 RoPE 的 base 参数：

$$
\text{base}_{\text{new}} = \text{base} \cdot \alpha^{d/(d-2)}
$$

```python
def ntk_aware_rope(base_old, scale_factor, dim):
    """NTK-aware base 调整"""
    alpha = scale_factor ** (dim / (dim - 2))
    return base_old * alpha

# 示例: LLaMA base=10000, 从 4K 扩到 32K
# alpha = 8^(128/(128-2)) ≈ 8^1.016 ≈ 8.27
# base_new = 10000 * 8.27 = 82700
```

### 4.4 YaRN 综合方案

YaRN = NTK-aware Scaling + Temperature Tuning。在 NTK 的基础上，对不同频率维度应用不同的温度系数：

$$
\lambda_i = 1 + (\alpha - 1) \cdot \frac{i}{d/2}
$$

```python
def yarn_rope_freqs(dim, base, scale, original_max_len):
    """YaRN: 对每个频率维度独立缩放"""
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # 频率相关的缩放因子
    ramp = torch.linspace(0, 1, dim // 2)
    scale_per_dim = 1 + (scale - 1) * ramp

    theta = theta / scale_per_dim
    return theta
```

---

## 5. 采样参数完整实现

### 5.1 完整采样 Pipeline

```python
import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=0.7, top_k=50, top_p=0.9,
                      repetition_penalty=1.1, prev_tokens=None, min_p=0.05):
    """
    完整的 LLM 采样流程。
    logits: (B, V) — 模型输出的原始 logits
    """
    # ==== Step 1: Temperature ====
    logits = logits / max(temperature, 1e-8)

    # ==== Step 2: Repetition Penalty ====
    if prev_tokens is not None and repetition_penalty != 1.0:
        for i, prev in enumerate(prev_tokens):
            for token_id in set(prev.tolist()):
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= repetition_penalty
                else:
                    logits[i, token_id] *= repetition_penalty

    # ==== Step 3: Min-P 过滤 (过滤低概率噪声) ====
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        min_threshold = max_prob * min_p
        logits = logits.masked_fill(probs < min_threshold, float('-inf'))

    # ==== Step 4: Top-K ====
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < min_top_k, float('-inf'))

    # ==== Step 5: Top-P (Nucleus) ====
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # 移除累积概率超过 top_p 的 tokens
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # 散射回原索引
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, float('-inf'))

    # ==== Step 6: Softmax + Sample ====
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
    return next_token, probs


# 完整生成循环
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256,
             temperature=0.7, top_k=50, top_p=0.9):
    """自回归文本生成"""
    model.eval()
    input_ids = tokenizer.encode(prompt).to(device).unsqueeze(0)  # (1, S)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        logits = logits[:, -1, :]  # 只取最后一个位置的 logits

        next_token, _ = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prev_tokens=input_ids
        )

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist())
```

### 5.2 各参数效果可视化

```
Temperature 对概率分布的影响 (假设原始 logits 已归一化):

T=0.1:  [0.99, 0.008, 0.002, ...]  ← 尖锐分布，近乎 argmax
T=0.7:  [0.45, 0.25, 0.15, 0.10, 0.05]  ← 平衡分布
T=1.0:  [0.30, 0.22, 0.18, 0.16, 0.14]  ← 原始分布
T=2.0:  [0.18, 0.17, 0.17, 0.16, 0.16, ...]  ← 趋于均匀

Top-P=0.9 示例:
Token (按概率排): A(0.40) B(0.25) C(0.20) D(0.08) E(0.04) F(0.02) G(0.01)
累积概率:         0.40   0.65   0.85   0.93*  ← 到此超过0.9，后面截断
采样池: {A, B, C, D}
```

### 5.3 惩罚机制详解

```python
def apply_repetition_penalty(logits, prev_token_ids, penalty=1.2):
    """
    重复惩罚: 对已出现的 token 降低其 logits。
    penalty > 1: 惩罚重复 (降低概率)
    penalty < 1: 鼓励重复 (提高概率)
    """
    unique_tokens = set(prev_token_ids.tolist())
    for token_id in unique_tokens:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits

def apply_frequency_penalty(logits, prev_token_ids, penalty=0.5):
    """
    频率惩罚: 出现次数越多，惩罚越重。
    与 repetition_penalty 的区别: 考虑频率而非仅出现与否。
    """
    token_counts = Counter(prev_token_ids.tolist())
    for token_id, count in token_counts.items():
        logits[token_id] -= penalty * count
    return logits
```

---

## 6. 预训练

### 6.1 数据 Pipeline

```
┌───────────────────────────────────────────────────────┐
│                    预训练数据流                          │
│                                                       │
│  多源语料收集                                          │
│  (Web, Books, Code, Wikipedia, Papers)                 │
│    │                                                  │
│    ▼                                                  │
│  ┌──────────────┐                                     │
│  │ 质量过滤       │ ← 语言检测、困惑度过滤、去重(MinHash) │
│  └──────┬───────┘                                     │
│         ▼                                             │
│  ┌──────────────┐                                     │
│  │ 敏感内容清洗    │ ← 隐私信息、有害内容、版权材料       │
│  └──────┬───────┘                                     │
│         ▼                                             │
│  ┌──────────────┐                                     │
│  │ 数据配比       │ ← Web:Code:Books = 60:25:15 等     │
│  └──────┬───────┘                                     │
│         ▼                                             │
│  ┌──────────────┐                                     │
│  │ Tokenization  │ → (B, S) 张量流                     │
│  └──────────────┘                                     │
└───────────────────────────────────────────────────────┘
```

### 6.2 分布式训练

| 并行策略 | 原理 | 通信量 | 适用瓶颈 |
|---------|------|--------|---------|
| **数据并行 (DP)** | 每个 GPU 持有完整模型，独立处理不同 batch | 梯度 AllReduce ($O(P)$) | batch 不够大 |
| **模型并行 (TP)** | 单层的权重矩阵切分到多 GPU | 每次前向/反向需通信 ($O(L \times B \times S)$) | 单层太大放不下 |
| **流水线并行 (PP)** | 不同层放在不同 GPU | 仅相邻层间传递激活值 ($O(B)$) | 层数多 |
| **序列并行 (SP)** | 长序列切分到多 GPU | RingAttention 式环形通信 | 序列极长 |
| **ZeRO (3阶段)** | 优化器状态/梯度/参数分片存储 | 按需 Gather/Scatter | 总显存不够 |

**3D 并行**：生产训练通常组合 TP + PP + DP：

```
  GPU 0     GPU 1     GPU 2     GPU 3
┌────────┐┌────────┐┌────────┐┌────────┐
│Layers  ││Layers  ││Layers  ││Layers  │  ← PP 组 1
│ 0-7    ││ 0-7    ││ 0-7    ││ 0-7    │
│(TP切分)││(TP切分)││(TP切分)││(TP切分)│
└────────┘└────────┘└────────┘└────────┘
┌────────┐┌────────┐┌────────┐┌────────┐
│Layers  ││Layers  ││Layers  ││Layers  │  ← PP 组 2
│ 8-15   ││ 8-15   ││ 8-15   ││ 8-15   │
└────────┘└────────┘└────────┘└────────┘
   ↑ TP 维度 (层内)      ↑ DP 维度 (数据)
```

### 6.3 混合精度训练

```python
# 典型训练配置 (PyTorch 风格)
from torch.cuda.amp import autocast, GradScaler

model = CausalLM(vocab_size=128000, d_model=4096, n_layers=32, n_heads=32, max_seq_len=4096)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scaler = GradScaler()  # 梯度缩放器 (防止 FP16 下溢)

for batch in dataloader:
    with autocast(dtype=torch.bfloat16):  # BF16 比 FP16 动态范围更大
        logits, loss = model(batch.input_ids, batch.targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

| 精度 | 指数位 | 尾数位 | 动态范围 | 适用 |
|------|--------|--------|---------|------|
| FP32 | 8 | 23 | ~10^38 | 训练基准 |
| FP16 | 5 | 10 | ~65504 | 推理 (需 loss scaling) |
| BF16 | 8 | 7 | ~10^38 | 训练 + 推理 (推荐) |
| FP8 | 5/4 | 2/3 | ~448 | H100 推理加速 |

---

## 7. 后训练 (Post-Training)

### 7.1 SFT (监督微调)

在高质量 (instruction, response) 对上做微调，使 LLM 学会遵循指令格式。

**数据格式 (ShareGPT / ChatML)**：

```json
{
  "messages": [
    {"role": "system", "content": "你是专业的数据分析师。"},
    {"role": "user", "content": "帮我写 SQL 查上个月的销售额 TOP 10。"},
    {"role": "assistant", "content": "SELECT product_name, SUM(amount) as total\nFROM orders\nWHERE order_date BETWEEN '2025-04-01' AND '2025-04-30'\nGROUP BY product_name\nORDER BY total DESC\nLIMIT 10;"}
  ]
}
```

**SFT 损失**：仅在 Assistant 回复部分计算 loss（mask 掉 User 和 System 部分）：

```python
def sft_loss(logits, labels, mask):
    """
    logits: (B, S, V)
    labels: (B, S)
    mask:   (B, S) — 1 where assistant tokens, 0 otherwise
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask   = mask[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )
    loss_per_token = loss_per_token.view(shift_mask.shape)

    # 只在 assistant token 上计算 loss
    masked_loss = (loss_per_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    return masked_loss
```

### 7.2 RLHF 完整流程

**三阶段训练**：

```
阶段 1: SFT
  基础模型 + 高质量QA对 → SFT 模型 π_SFT

阶段 2: 奖励模型训练
  同一 prompt 下多个 response 的人类偏好 → Bradley-Terry 模型 → RM(·,·)

阶段 3: PPO 优化
  π_SFT + RM → PPO 策略优化 → π_RLHF
```

### 7.3 奖励模型

**Bradley-Terry 偏好模型**：

两个回答 $y_w$ (win) 和 $y_l$ (loss) 相比，$y_w$ 更好的概率：

$$
P(y_w \succ y_l \mid x) = \frac{e^{R(x, y_w)}}{e^{R(x, y_w)} + e^{R(x, y_l)}} = \sigma(R(x, y_w) - R(x, y_l))
$$

**奖励模型损失**：

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma(R(x, y_w) - R(x, y_l))\right]
$$

```python
def reward_model_loss(rm, batch):
    """
    rm: reward model, returns scalar score per (prompt, response)
    batch: (prompt, chosen_response, rejected_response)
    """
    r_chosen  = rm(batch['prompt'], batch['chosen'])
    r_rejected = rm(batch['prompt'], batch['rejected'])

    # Bradley-Terry loss
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss
```

### 7.4 PPO 策略优化

PPO 利用奖励模型的打分在线优化策略。核心是 **KL 散度约束** 防止策略偏离太远：

$$
r_{\text{total}} = r_{\text{RM}} - \beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}})
$$

$$
\text{KL} = \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}
$$

**PPO Clipped Objective**：

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[ \min\left(
r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是概率比，$A_t$ 是优势函数。

```python
def ppo_step(policy, ref_policy, reward_model, prompts, beta=0.1, eps=0.2):
    """
    单步 PPO 优化。
    beta: KL 惩罚系数
    eps:  clipping 范围
    """
    # 1. 从当前策略采样 response
    responses, old_logprobs = policy.sample(prompts)

    # 2. 计算奖励和 KL
    rm_scores = reward_model(prompts, responses)
    ref_logprobs = ref_policy.logprobs(prompts, responses)
    kl_penalty = (old_logprobs - ref_logprobs).mean()  # 近似 KL 散度
    rewards = rm_scores - beta * kl_penalty

    # 3. 计算优势 (GAE)
    advantages = compute_gae(rewards)

    # 4. PPO clipped loss
    curr_logprobs = policy.logprobs(prompts, responses)
    ratio = (curr_logprobs - old_logprobs.detach()).exp()

    loss1 = ratio * advantages
    loss2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    loss = -torch.min(loss1, loss2).mean()

    return loss
```

### 7.5 DPO — 无需奖励模型的直接对齐

DPO 的核心洞察：最优策略 $\pi^*$ 与奖励函数 $R$ 之间存在闭式解：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} R(x, y)\right)
$$

反解出 $R$，代入 Bradley-Terry 得到直接优化目标：

$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}\left[ \log \sigma\!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right) \right]
$$

**DPO 完整实现**：

```python
def dpo_loss(policy, ref_policy, batch, beta=0.1):
    """
    直接偏好优化 — 无需训练奖励模型。
    batch: {prompt, chosen, rejected}
    """
    # 分别计算 chosen 和 rejected 的 log-probabilities
    chosen_logps  = policy.logprobs(batch['prompt'], batch['chosen'])
    rejected_logps = policy.logprobs(batch['prompt'], batch['rejected'])

    # 参考模型的 log-probabilities (冻结)
    with torch.no_grad():
        ref_chosen_logps  = ref_policy.logprobs(batch['prompt'], batch['chosen'])
        ref_rejected_logps = ref_policy.logprobs(batch['prompt'], batch['rejected'])

    # DPO 核心：隐式奖励差
    chosen_ratio  = chosen_logps  - ref_chosen_logps
    rejected_ratio = rejected_logps - ref_rejected_logps

    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**PPO vs DPO 对比**：

| 维度 | PPO | DPO |
|------|-----|-----|
| 奖励模型 | 需要单独训练 | 不需要 |
| 训练稳定性 | 复杂（4个模型同时运行） | 简单（2个模型，离线数据） |
| 数据需求 | 大量在线采样 | 预先收集的偏好对 |
| 理论保证 | 策略梯度收敛 | 等价于 Bradley-Terry 下的最优策略 |
| 适用场景 | 大规模商业训练 | 学术研究、快速实验 |

---

## 关键数值速查

| 参数 | LLaMA 2 7B | LLaMA 3 8B | Mistral 7B | Qwen2.5 7B | DeepSeek-V2 |
|------|-----------|-----------|-----------|-----------|-------------|
| $D$ (d_model) | 4096 | 4096 | 4096 | 3584 | 5120 |
| $H$ (n_heads) | 32 | 32 | 32 | 28 | 128 |
| $d_k$ (head_dim) | 128 | 128 | 128 | 128 | 128 |
| $n_{\text{layers}}$ | 32 | 32 | 32 | 28 | 60 |
| $V$ (vocab) | 32K | 128K | 32K | 152K | 100K |
| RoPE base | 10000 | 500000 | 1000000 | 1000000 | 10000 |
| KV heads (GQA) | 32 (off) | 8 | 8 | 4 | 1 (MLA) |
| FFN 类型 | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| Context 长度 | 4K | 8K | 32K | 128K | 128K |

::: tip 工程实践
构建 Agent 系统时，对 LLM 的理解深度的分水岭在于：能否精准预测 Prompt 的 Token 消耗，以及能否通过采样参数的组合调控模型的「确定性 vs 创造性」平衡。
:::
