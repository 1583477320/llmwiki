---
layout: home

hero:
  name: "AgentWiki"
  text: "从深度学习到 LMM + Agent 落地"
  tagline: 系统化沉淀神经网络、Transformer、大语言模型、Agent 架构与工程实践知识
  image:
    src: /logo.svg
    alt: AgentWiki
  actions:
    - theme: brand
      text: 开始学习
      link: /01-deep-learning/nn-foundations
    - theme: alt
      text: GitHub
      link: https://github.com/1583477320/llmwiki

features:
  - icon: 🧬
    title: 深度学习基础
    details: 反向传播、梯度下降、优化器、损失函数与正则化 —— 神经网络训练的数学本质。
    link: /01-deep-learning/nn-foundations/

  - icon: 🔮
    title: Transformer 架构
    details: 自注意力机制、多头注意力、位置编码、FlashAttention、GQA、MLA 等前沿注意力变体详解。
    link: /02-transformers/transformer-mechanics/

  - icon: 🤖
    title: 大语言模型
    details: Token 机制、Context 窗口、采样策略、RLHF/DPO 对齐，以及多模态模型的对齐与感知原理。
    link: /03-large-models/llm-core/

  - icon: 🧠
    title: Agent 认知与编排
    details: ReAct 控制流、Plan-and-Execute、多 Agent 协作、结构化 Tool Call、记忆系统与多模态 RAG。
    link: /04-agent-architectures/react-flow/

  - icon: 🚀
    title: Agent 产品实战
    details: Claude Code、Codex CLI、Aider、Cursor、OpenClaw 等主流产品架构解析、部署配置与使用指南。
    link: /06-agent-products/overview/

  - icon: ⚙️
    title: 工程落地
    details: vLLM 推理优化、PagedAttention、模型量化、投机解码，以及 Agent 防死循环与桌面自动化安全。
    link: /05-engineering-tools/vllm-inference/

---
