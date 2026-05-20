# vLLM 高性能推理优化

> PagedAttention 与多模态量化调优 —— 让模型推理飞起来

## 为什么需要 vLLM？

LLM 推理的瓶颈不在于计算能力，而在于**显存管理**。标准 Transformer 推理的 KV Cache 管理存在严重的显存碎片化问题，导致实际吞吐量远低于理论峰值。

### 传统问题

- **KV Cache 膨胀**：每个 Token 的 Key/Value 向量需要缓存，长序列下显存爆炸
- **显存碎片**：预分配固定大小导致大量空闲显存无法利用
- **请求并发低**：碎片化显存限制了 Batch Size

## PagedAttention 核心原理

vLLM 的核心创新是 **PagedAttention** —— 借鉴操作系统虚拟内存的分页思想管理 KV Cache。

### 类比操作系统

| 操作系统 | vLLM |
|---------|------|
| 物理内存 → 页框 (Page Frame) | GPU 显存 → KV Block |
| 虚拟内存 → 页表 | 逻辑 KV Cache → Block Table |
| 进程分配连续虚拟地址 | 请求分配连续逻辑 KV Cache |
| 实际物理内存可以不连续 | 实际 KV Block 可以不连续 |
| 避免外部碎片 | 避免 KV Cache 碎片 |

### 工作流程

```
1. 请求到达，分配逻辑 KV Cache
2. Block Table 将逻辑位置映射到物理 KV Block
3. Prefill 阶段：一次处理整个 Prompt，生成 KV Cache
4. Decode 阶段：逐 Token 生成，追加到对应 Block
5. Block 写满则分配新 Block
6. 请求完成后释放所有 Block
```

::: tip 性能收益
PagedAttention 将 KV Cache 浪费从模型内存的 60-80% 降至 4% 以下，吞吐量提升约 **24×**（相比于 HuggingFace Transformers）。
:::

## 连续批处理

vLLM 的 Continuous Batching 允许请求动态加入和离开批处理：

- 传统静态批处理：所有请求需同时完成才能处理下一批
- 连续批处理：完成的请求立即移出，新请求立即加入
- 迭代级调度：每个 Decode Step 都可以调整批处理组合

## 多模态量化调优

### 量化基础

| 精度 | 每参数比特 | 适用场景 |
|------|-----------|---------|
| FP32 | 32 | 训练（全精度） |
| FP16 | 16 | 推理基准 |
| BF16 | 16 | 训练（更好的动态范围） |
| INT8 | 8 | 推理加速 |
| INT4 / NF4 | 4 | 边缘部署 |

### AWQ 量化

vLLM 主流使用 AWQ（Activation-aware Weight Quantization）：

- **核心洞察**：并非所有权重同样重要 — 少量「显著通道」对模型质量影响更大
- **策略**：对显著通道使用 Per-Channel 缩放保护，其余通道激进量化

### GPTQ 量化

逐层最优量化，通过校准数据集求解最优量化参数。

### 多模态模型量化特殊考量

多模态 LMM 的量化比纯文本 LLM 更复杂：

1. **视觉编码器量化**
   - ViT 的激活值分布与 LLM 不同
   - 视觉 Patch 间的注意力模式对量化更敏感
   - 建议：视觉编码器使用 FP16/BF16，仅量化 LLM 部分

2. **连接器量化**
   - MLP 连接器参数量通常很小，建议保持 FP16
   - Cross-Attention Resampler 的注意力计算对精度敏感

3. **图像 Token 数量影响**
   - 高分辨率输入 → 数千个视觉 Token → KV Cache 暴增
   - PagedAttention 在此场景收益最大

## vLLM 部署最佳实践

### 关键参数

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --quantization awq
```

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `gpu-memory-utilization` | GPU 显存使用率上限 | 0.85–0.95 |
| `max-num-seqs` | 最大并发序列数 | 根据显存和 max-model-len 计算 |
| `enable-prefix-caching` | 复用相同 Prefix 的 KV Cache | 推荐开启 |
| `max-model-len` | 支持的最大序列长度 | 根据业务需求设置 |
| `quantization` | 量化方法 | 低显存场景用 awq |

### 投机解码

使用 Draft Model 快速生成候选 Token，Target Model 批量验证，在不降低质量的前提下提升 Decode 速度：

```
Draft Model (小、快) → 生成 K 个候选 Token
Target Model (大、准) → 一次前向验证所有候选 → 接受/拒绝
```
