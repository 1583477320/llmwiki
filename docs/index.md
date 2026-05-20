---
layout: home

hero:
  name: AgentWiki
  text: 深度演进到 LMM + Agent
  tagline: 从神经网络基础到多模态 Agent 工程落地的系统化知识库
  image:
    src: /logo.svg
    alt: AgentWiki
  actions:
    - theme: brand
      text: 📖 开始系统学习
      link: /01-deep-learning/nn-foundations
    - theme: alt
      text: ⚡ GitHub
      link: https://github.com

features:
  - icon: 🧬
    title: 深度学习核心基础
    details: 从反向传播与优化器本质出发，理解神经网络训练的数学直觉，掌握损失函数设计、梯度优化与正则化等核心概念。
    link: /01-deep-learning/nn-foundations/

  - icon: 🔮
    title: Transformer 架构与注意力
    details: 深入拆解自注意力机制的数学原理、多头注意力的工程实现、位置编码的演进，以及 FlashAttention、GQA、MLA 等前沿注意力变体。
    link: /02-transformers/transformer-mechanics/

  - icon: 🤖
    title: 大语言模型与多模态
    details: 掌握 LLM Token 机制、Context 窗口与采样策略控制。理解 Vision Transformer 如何在嵌入空间完成跨模态对齐，实现图文统一建模。
    link: /03-large-models/llm-core/

  - icon: 🧠
    title: Agent 认知与编排系统
    details: 构建 ReAct 感知-思考-行动状态机循环。基于 JSON Schema 强约束实现结构化 Tool Call，并设计多模态 RAG 驱动的长期记忆架构。
    link: /04-agent-architectures/react-flow/

  - icon: 🚀
    title: 主流 Agent 产品实战
    details: 深入 Claude Code、Codex CLI、Aider、Cursor、OpenClaw 等主流 Agent 产品的架构原理、部署配置与使用技巧，掌握选型与落地能力。
    link: /06-agent-products/overview/

  - icon: ⚙️
    title: 生产环境工程落地
    details: 深入 vLLM PagedAttention 推理优化与量化调优。实践 OpenClaw 桌面自动化的动作空间设计，以及 Agent 动作死循环的反思容错机制。
    link: /05-engineering-tools/vllm-inference/

---
