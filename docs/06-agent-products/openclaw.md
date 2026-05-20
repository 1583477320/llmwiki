# OpenClaw

> 开源桌面自动化 Agent —— LMM 驱动的 GUI 操作引擎

## 产品定位

OpenClaw 是一个开源的桌面自动化 Agent 框架，利用 LMM（多模态大模型）理解屏幕内容并控制鼠标键盘执行操作。它是目前 Agent 从纯文本交互走向真实 GUI 环境操作的代表项目。

## 架构原理

```
┌─────────────────────────────────────────────┐
│              OpenClaw Agent                  │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │           Planner                     │   │
│  │   任务分解 → 子目标 → 动作序列          │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │           LMM Engine                  │   │
│  │   屏幕理解 → UI 元素定位 → 决策         │   │
│  │   (Claude / GPT-4V / Qwen-VL)        │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │         Action Executor               │   │
│  │   鼠标控制 │ 键盘输入 │ 截图 │ 系统API  │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │         Verifier                      │   │
│  │   动作前后截图对比 │ 断言检查 │ 重试     │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## 部署与安装

```bash
# 克隆仓库
git clone https://github.com/openclaw/openclaw
cd openclaw

# 安装
pip install -e .

# 系统依赖（Linux）
sudo apt-get install xdotool x11-utils scrot

# 系统依赖（macOS）
brew install cliclick

# 配置模型
export MODEL_BACKEND="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."

# 启动服务
openclaw serve --port 8765
```

### Docker 部署（推荐）

```bash
# 拉取镜像
docker pull openclaw/openclaw:latest

# 运行（带 GUI 访问）
docker run -it \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  openclaw/openclaw:latest
```

## 动作空间

OpenClaw 定义了完整的桌面操作原语：

### 鼠标操作

```json
{"type": "click",        "x": 400, "y": 300}
{"type": "double_click", "x": 400, "y": 300}
{"type": "right_click",  "x": 400, "y": 300}
{"type": "drag",         "start": [100, 200], "end": [300, 400]}
{"type": "scroll",       "direction": "down", "amount": 5}
{"type": "hover",        "x": 400, "y": 300}
```

### 键盘操作

```json
{"type": "type",    "text": "Hello World"}
{"type": "press",   "key": "enter"}
{"type": "hotkey",  "keys": ["Ctrl", "c"]}
{"type": "hotkey",  "keys": ["Ctrl", "Shift", "n"]}
```

### 系统操作

```json
{"type": "screenshot",         "region": "full"}
{"type": "screenshot",         "region": [0, 0, 800, 600]}
{"type": "open_app",           "name": "Chrome"}
{"type": "close_app",          "name": "Calculator"}
{"type": "wait",               "ms": 2000}
{"type": "get_clipboard"}
{"type": "set_clipboard",      "text": "..."}
```

## 视觉定位策略

如何将"点击登录按钮"变为精确的像素坐标？OpenClaw 支持多种策略：

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **Direct Grounding** | LMM 直接输出边界框坐标 | 明显的 UI 元素 |
| **Set-of-Mark** | 为检测到的 UI 元素标注数字 ID | 复杂页面，密集元素 |
| **OCR Matching** | 通过文字识别找到元素位置 | 文本按钮、菜单 |
| **Accessibility Tree** | 读取系统无障碍 API | 原生应用 |
| **Template Matching** | 提供参考图片进行模板匹配 | 图标、特定按钮 |

### Set-of-Mark 流程

```
1. 屏幕截图 → UI 检测模型 (OmniParser/UGround)
2. 输出: [{id: 1, type: "button", text: "登录", bbox: [320,240,380,270]},
          {id: 2, type: "input", text: "用户名", bbox: [200,180,400,210]}, ...]
3. 在截图上叠加数字标签
4. LMM: "点击 id=1 的元素（登录按钮）"
5. 解析: id=1 → bbox 中心点 [350, 255] → 执行鼠标点击
```

## Python API 使用

```python
from openclaw import Agent, Action

# 创建 Agent
agent = Agent(
    model="claude-sonnet-4-6",
    backend="anthropic",
    screen_size=(1920, 1080)
)

# 单步操作
agent.click(x=350, y=255)
agent.type_text("Hello World")
agent.hotkey("Ctrl", "s")

# 高层次任务
result = agent.run("""
    1. 打开 Firefox 浏览器
    2. 访问 github.com
    3. 搜索 "openclaw"
    4. 截图保存搜索结果
    5. 关闭浏览器
""")

print(result.status)       # "success"
print(result.steps)        # 执行步骤详情
print(result.screenshots)  # 每步截图路径
```

## 安全配置

```yaml
# openclaw.yaml
safety:
  # 高危操作确认
  confirm_before:
    - action: "rm"
    - action: "delete"
    - domain: "*.bank.com"
    - domain: "*.internal.corp"

  # 沙箱
  sandbox:
    enabled: true
    docker_image: "openclaw/sandbox:latest"
    # 仅在沙箱内操作，不影响真实桌面

  # 人机协同
  human_in_loop:
    enabled: true
    on_action_count: 5        # 每 5 步暂停
    on_uncertainty: 0.7       # 置信度 < 70% 暂停
    on_sensitive_action: true  # 敏感操作暂停

  # 录制与回放
  record: true
  replay_dir: ./sessions/
```

## 与编程 Agent 的对比

| 维度 | OpenClaw | Claude Code | Codex CLI |
|------|---------|------------|-----------|
| 操作对象 | 任意 GUI 应用 | 终端 + 代码文件 | 终端 + 代码文件 |
| 感知方式 | 多模态视觉理解 | 文本/Tool 输出 | 文本/Tool 输出 |
| 执行方式 | 模拟键鼠输入 | 调用系统工具 | 调用系统工具 |
| 适用任务 | 操作网页、桌面软件 | 编写代码 | 编写代码 |
| 安全风险 | 高（真实系统操作） | 中 | 低（内置沙箱） |
| 典型场景 | RPA、测试自动化、GUI 操作 | 软件开发 | 软件开发 |

## 实践建议

1. **先用沙箱测试**：在 Docker 中运行，确认动作序列正确后再上真实桌面
2. **启用 human_in_loop**：高风险场景每 5-10 步暂停确认
3. **动作后验证**：每次关键操作后截图比对，不要盲目信任
4. **时间等待要充足**：GUI 操作天然异步，点击后给足渲染时间（2-5 秒）
5. **录制备查**：生产环境中录制所有操作，出问题时可以回放分析
