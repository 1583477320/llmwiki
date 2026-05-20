import { defineConfig } from 'vitepress'

const BASE = process.env.BASE_PATH || '/'

export default defineConfig({
  title: 'AgentWiki',
  description: '深度学习到 LMM + Agent 落地知识库',
  base: BASE,

  themeConfig: {
    logo: '🧠',

    search: {
      provider: 'local'
    },

    nav: [
      { text: '首页', link: '/' },
      { text: '知识库', link: '/01-deep-learning/nn-foundations' }
    ],

    sidebar: [
      {
        text: '🧬 深度学习核心基础',
        collapsed: false,
        items: [
          { text: '神经网络与梯度下降', link: '/01-deep-learning/nn-foundations' }
        ]
      },
      {
        text: '🔮 Transformer 架构与注意力',
        collapsed: false,
        items: [
          { text: 'Transformer 架构详解', link: '/02-transformers/transformer-mechanics' },
          { text: 'Transformer 分类与注意力机制', link: '/02-transformers/transformer-taxonomy' }
        ]
      },
      {
        text: '🤖 大语言模型与多模态',
        collapsed: false,
        items: [
          { text: 'LLM 核心原理', link: '/03-large-models/llm-core' },
          { text: 'LMM 多模态感知与对齐', link: '/03-large-models/multimodal-alignment' }
        ]
      },
      {
        text: '🧠 Agent 认知与编排',
        collapsed: false,
        items: [
          { text: 'Agent 类型与使用范式', link: '/04-agent-architectures/agent-types' },
          { text: 'ReAct 状态机控制流', link: '/04-agent-architectures/react-flow' },
          { text: '结构化输出与 Tool Call', link: '/04-agent-architectures/structured-json' },
          { text: '记忆系统与多模态 RAG', link: '/04-agent-architectures/memory-rag' }
        ]
      },
      {
        text: '🚀 主流 Agent 产品',
        collapsed: false,
        items: [
          { text: '产品总览与选型', link: '/06-agent-products/overview' },
          { text: 'Claude Code', link: '/06-agent-products/claude-code' },
          { text: 'Codex CLI', link: '/06-agent-products/codex-cli' },
          { text: 'Aider', link: '/06-agent-products/aider' },
          { text: 'Cursor', link: '/06-agent-products/cursor' },
          { text: 'OpenClaw', link: '/06-agent-products/openclaw' },
          { text: '更多 Agent 产品', link: '/06-agent-products/other-agents' }
        ]
      },
      {
        text: '⚙️ 工程落地与实践',
        collapsed: false,
        items: [
          { text: 'vLLM 高性能推理优化', link: '/05-engineering-tools/vllm-inference' },
          { text: 'OpenClaw 桌面自动化', link: '/05-engineering-tools/openclaw-automation' },
          { text: '防死循环反思机制', link: '/05-engineering-tools/loop-prevention' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/1583477320/llmwiki' }
    ],

    footer: {
      message: 'Powered by VitePress',
      copyright: '© 2026 AgentWiki — 硬核工程知识库'
    },

    outline: {
      level: [2, 3],
      label: '页面导航'
    },

    docFooter: {
      prev: '← 上一节',
      next: '下一节 →'
    }
  },

  markdown: {
    lineNumbers: true
  }
})
