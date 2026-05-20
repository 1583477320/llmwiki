# LMM 多模态感知与对齐

> Vision Transformer 与跨模态对齐 —— 让模型「看见」世界

## 从 LLM 到 LMM

大型多模态模型（Large Multimodal Model, LMM）将 LLM 的文本理解能力扩展至视觉、音频等多模态输入，使模型能够「看见」并理解图像、视频内容。

### 核心架构范式

绝大多数 LMM 采用「视觉编码器 + 连接器 + LLM」的架构：

```
图像 → [Vision Encoder] → [Connector/Projector] → [LLM] → 文本输出
```

| 模型 | 视觉编码器 | 连接器 | LLM 基座 |
|------|-----------|--------|---------|
| LLaVA | CLIP ViT-L | MLP 投影 | LLaMA/Vicuna |
| Qwen-VL | ViT-G | Cross-Attention Resampler | Qwen |
| InternVL | InternViT | 像素洗牌投影 | InternLM |
| GPT-4V | 闭源 | 闭源 | GPT-4 |

---

## Vision Transformer (ViT)

### 核心思想

将图像的 Patch 序列类比为 NLP 中的 Token 序列：

1. 图像被切分为固定大小的 Patches（如 $16\times16$）
2. 每个 Patch 被展开为向量，经线性投影得到 Patch Embedding
3. 加入位置编码后送入标准 Transformer

### 对比 CNN

| 特性 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 局部性、平移等变性（强） | 弱，完全由数据学习 |
| 数据需求 | 中等 | 大规模数据 |
| 计算复杂度 | $O(k^2 n)$ | $O(n^2)$（注意力） |

---

## 跨模态对齐原理

跨模态对齐的目标是让视觉特征和文本特征在同一个语义空间中具有一致的表示。

### CLIP 对比学习

CLIP 是 LMM 最核心的对齐基础模型，通过对比学习将图像-文本对映射到共享嵌入空间：

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i}\left[\log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)} + \log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ji}/\tau)}\right]
$$

其中 $s_{ij}$ 是第 $i$ 张图像与第 $j$ 段文本的余弦相似度。

### 连接器设计

| 连接器类型 | 描述 | 代表模型 |
|-----------|------|---------|
| Linear / MLP | 最简单，直接投影 | LLaVA-1.5 |
| Q-Former | 可学习的 Query 向量压缩视觉 Token | BLIP-2 |
| Cross-Attention Resampler | 用可学习 Query 交叉关注视觉特征 | Qwen-VL |
| Pixel Shuffle | 像素重排减少 Token 数 | InternVL 2 |

::: tip 关键设计权衡
连接器需要在**视觉信息保留**与**Token 压缩效率**之间取得平衡。视觉特征通常产生数百至数千个 Token，而 LLM 的 Context 窗口昂贵，因此高效的视觉 Token 压缩是 LMM 工程化的核心挑战。
:::

### 高分辨率策略

高分辨率对 OCR、图表理解等任务至关重要，常用方法：

- **Dynamic Resolution**：将图像切分为多个子图分别编码后拼接
- **AnyRes**：支持任意分辨率输入，自适应分块
- **Tile-based Encoding**：按固定尺寸 Tile 切分，适合文档场景

---

## 多模态训练策略

### 两阶段训练

1. **预对齐阶段**：冻结视觉编码器和 LLM，仅训练连接器，在大量图文对上快速对齐模态
2. **指令微调阶段**：解冻 LLM（或全部参数），在多模态指令数据上微调，提升指令遵循与推理能力
