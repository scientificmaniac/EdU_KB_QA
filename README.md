# 职业生涯规划智能问答系统

> 一个基于**本地知识库 + Ollama 本地大模型**的桌面问答应用。
> 知识库负责给出准确、可控的参考事实，大模型负责把这些事实组织成自然、可执行的建议。

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-182%20passed-brightgreen.svg)](#运行测试)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.com/)

---

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [配置](#配置)
- [命令行参考](#命令行参考)
- [知识库格式](#知识库格式)
- [项目结构](#项目结构)
- [运行测试](#运行测试)
- [打包为 exe](#打包为-exe)
- [常见问题](#常见问题)
- [开发说明](#开发说明)
- [许可证](#许可证)

---

## 特性

| 能力 | 说明 |
|---|---|
| 🧠 **检索增强问答** | 先用本地知识库检索参考资料，再把参考资料注入提示词，回答有据可依 |
| 🔍 **中文语义检索** | 字符 bigram Dice 系数 + 序列相似度混合打分，改写提问（「职业规划怎么做」）也能命中，并用阈值挡住无关条目 |
| 🏷️ **标签体系** | 知识条目支持多标签，可按标签筛选检索范围与浏览知识库 |
| ⚡ **流式输出** | 逐字显示回答，首字延迟从"等整段生成完"降到"几百毫秒" |
| 🗂️ **知识库管理** | 界面上直接添加、导入、导出知识条目 |
| 🛡️ **健壮性** | 原子写入、损坏自动备份、读取故障绝不覆盖原数据、网络失败重试与分类报错 |
| ⚙️ **可配置** | 模型、服务地址、检索阈值、历史轮数等均可在配置文件 / 环境变量 / 命令行中调整 |
| 🧪 **可测试** | 182 个自动化测试，全部用 Mock，不需要真实 Ollama 服务 |

---

## 快速开始

### 1. 环境要求

- **Python 3.9+**（推荐 3.11）
- **tkinter**（Python 官方安装包默认自带；Debian/Ubuntu 需 `sudo apt install python3-tk`）
- **Ollama**（本地大模型运行时，[下载地址](https://ollama.com/download)）

### 2. 获取代码并安装依赖

```bash
git clone https://github.com/scientificmaniac/EdU_KB_QA.git
cd EdU_KB_QA

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 准备本地模型

```bash
ollama serve            # 若未作为后台服务运行
ollama pull llama2      # 或换成 qwen2、gemma2 等
```

### 4. 自检环境

```bash
python app.py --check
```

输出示例：

```
[配置]
  知识库路径   : D:\EDU_KB_Q&A\knowledge_base.json
  Ollama 地址  : http://localhost:11434
  模型         : llama2
  检索条数/阈值: top_n=3, min_score=0.25

[知识库]
  条目数量     : 2
  标签数量     : 6

[Ollama]
  服务状态     : 在线
  已安装模型   : 3 个
    - llama2:latest
    - qwen2:7b
```

### 5. 启动

```bash
python app.py
```

也可以在界面上按 `Return` 直接发送，输入 `exit` 退出。

---

## 配置

配置按 **内置默认值 → `config.json` → 环境变量 → 命令行参数** 的顺序逐层覆盖。

把 `config.example.json` 复制为 `config.json` 即可自定义（`config.json` 已在 `.gitignore` 中，不会误提交）：

```bash
cp config.example.json config.json
```

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `kb_path` | `knowledge_base.json` | 知识库路径，相对路径以项目根目录为基准 |
| `ollama_base_url` | `http://localhost:11434` | Ollama 服务地址 |
| `model` | `llama2` | 模型名称 |
| `connect_timeout` | `5.0` | 建连超时（秒） |
| `request_timeout` | `120.0` | 读取超时（秒） |
| `max_retries` | `2` | 网络类失败的重试次数 |
| `max_history_turns` | `3` | 保留的历史对话轮数，`0` 表示不保留上下文 |
| `top_n` | `3` | 检索最多注入的参考条目数 |
| `min_score` | `0.25` | 检索相似度阈值（`0~1`），调低可提高召回 |
| `stream` | `true` | 是否流式输出 |
| `cache_size` | `128` | 非流式响应缓存条数，`0` 表示禁用 |
| `temperature` | `0.7` | 采样温度 |
| `system_prompt` | 见示例 | 系统提示词 |
| `prompt_template` | 见示例 | 用户提示词模板，占位符：`{question}` `{context}` `{tags}` |

### 环境变量

所有配置项都有对应的环境变量，前缀 `CAREERQA_`：

```bash
# Windows PowerShell
$env:CAREERQA_MODEL = "qwen2"
$env:CAREERQA_OLLAMA_BASE_URL = "http://192.168.1.10:11434"

# macOS / Linux
export CAREERQA_MODEL=qwen2
```

可用变量：`CAREERQA_KB_PATH`、`CAREERQA_OLLAMA_BASE_URL`、`CAREERQA_MODEL`、`CAREERQA_CONNECT_TIMEOUT`、`CAREERQA_REQUEST_TIMEOUT`、`CAREERQA_MAX_RETRIES`、`CAREERQA_MAX_HISTORY_TURNS`、`CAREERQA_TOP_N`、`CAREERQA_MIN_SCORE`、`CAREERQA_STREAM`、`CAREERQA_CACHE_SIZE`、`CAREERQA_TEMPERATURE`

---

## 命令行参考

```bash
python app.py [选项]
python -m career_qa [选项]      # 等价写法
```

| 选项 | 说明 |
|---|---|
| `--config PATH` | 指定配置文件路径（默认 `config.json`） |
| `--kb PATH` | 指定知识库 JSON |
| `--ollama-url URL` | 指定 Ollama 服务地址 |
| `--model NAME` | 指定模型名称 |
| `--top-n N` | 检索返回条数 |
| `--min-score S` | 检索相似度阈值 |
| `--no-stream` | 关闭流式输出 |
| `--check` | 仅做环境自检后退出（退出码 `0` 正常 / `1` 有问题 / `2` 配置错误） |
| `--list-models` | 列出 Ollama 已安装模型 |
| `--version` | 显示版本 |

---

## 知识库格式

`knowledge_base.json` 是一个 JSON 数组，每个元素包含 `问题`、`答案` 和可选的 `标签`：

```json
[
    {
        "问题": "如何进行职业规划?",
        "答案": "职业规划需要明确自身优势、兴趣和价值观，设定长期和短期目标，并制定相应的发展计划。",
        "标签": ["职业规划", "职业发展", "目标设定"]
    },
    {
        "问题": "数据科学家的职业前景如何?",
        "答案": "数据科学家是当前市场上需求旺盛的职业，随着大数据和人工智能的发展，其前景非常广阔。",
        "标签": ["职业前景", "数据科学", "人工智能"]
    }
]
```

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `问题` | ✅ | 条目标题，索引与展示用；也接受英文键 `question` |
| `答案` | ✅ | 注入提示词的参考资料正文；也接受英文键 `answer` |
| `标签` | ❌ | 字符串数组，也接受英文键 `tags`、逗号分隔字符串 |

容错行为：

- 顶层也接受 `{"entries": [...]}` 形式的包装；
- 单条格式错误会被**跳过并告警**，不影响其余条目加载；
- 整个文件 JSON 语法损坏时，会自动重命名为 `knowledge_base.json.corrupt-<时间戳>` 备份，并以空库启动；
- 若是**读取故障**（文件被其他程序独占、权限不足、网络盘掉线），则**绝不移动或改名原文件**，
  只记录告警并以空库启动，同时禁止写回该文件 —— 避免一次读取抖动被误判成「文件损坏」而演变成数据丢失；
- 发生「读取故障 → 拒绝写回」后，排除故障再点界面上的「刷新」即可恢复正常读写；
- 保存采用「临时文件 + 原子替换」，中途失败不会写坏原文件。

---

## 项目结构

```
EDU_KB_Q&A/
├── app.py                      # 兼容入口（python app.py）
├── career_qa/                  # 应用包
│   ├── __init__.py             # 版本与包说明
│   ├── __main__.py             # python -m career_qa
│   ├── cli.py                  # 命令行入口、环境自检
│   ├── config.py               # 配置加载与校验
│   ├── knowledge_base.py       # 知识库读写 + 中文相似度检索
│   ├── llm_client.py           # Ollama HTTP 客户端（流式 / 缓存 / 重试）
│   ├── dialogue.py             # 检索 + 提示词 + 多轮历史编排
│   └── gui.py                  # tkinter 界面
├── tests/                      # 182 个自动化测试（Mock，无需 Ollama）
│   ├── test_config.py
│   ├── test_knowledge_base.py
│   ├── test_llm_client.py
│   ├── test_dialogue.py
│   └── test_gui_smoke.py       # GUI 端到端冒烟（无显示环境自动跳过）
├── knowledge_base.json         # 示例知识库
├── config.example.json         # 配置模板
├── requirements.txt
├── .github/workflows/ci.yml    # GitHub Actions 持续集成
├── 智能问答系统.spec            # PyInstaller 打包配置
├── CHANGELOG.md
└── LICENSE
```

### 数据流

```
用户提问
   │
   ▼
KnowledgeBase.search()          # 标签过滤 + 相似度打分 + 阈值筛选
   │  命中的参考资料
   ▼
DialogueManager.build_prompt()  # 参考资料 + 问题 → 提示词
   │
   ▼
OllamaClient.stream_chat()      # 携带多轮历史请求 /api/chat
   │  逐段文本
   ▼
queue.Queue → 主线程 _pump()    # 线程安全地渲染到界面
```

---

## 运行测试

测试全部使用 Mock，**不需要启动 Ollama**，也不需要联网：

```bash
python -m unittest discover -s tests -t . -v
```

预期输出：

```
Ran 182 tests in ~2.1s

OK
```

也可以只跑某一层：

```bash
python -m unittest tests.test_knowledge_base -v   # 知识库与检索
python -m unittest tests.test_llm_client -v       # HTTP 客户端与流式解析
python -m unittest tests.test_dialogue -v         # 对话编排
python -m unittest tests.test_gui_smoke -v        # GUI 端到端
```

在无图形界面的服务器上，`test_gui_smoke` 会自动跳过。

---

## 打包为 exe

```bash
pip install pyinstaller
pyinstaller 智能问答系统.spec --noconfirm
```

产物位于 `dist/智能问答系统.exe`。`build/` 与 `dist/` 都是可再生成的派生物，已被 `.gitignore` 忽略。

> 注意：打包后仍需本机运行 Ollama 服务；exe 只是客户端。

---

## 常见问题

<details>
<summary><b>提示「无法连接 Ollama 服务」（WinError 10061）</b></summary>

Ollama 没有运行，或地址配错了。

```bash
ollama serve                  # 启动服务
curl http://localhost:11434/api/tags   # 验证连通
```

如果 Ollama 跑在另一台机器上，用 `--ollama-url http://<ip>:11434` 指定，并确认对方监听 `0.0.0.0`（设置 `OLLAMA_HOST=0.0.0.0:11434`）。
</details>

<details>
<summary><b>提示「模型 'x' 不存在」</b></summary>

```bash
ollama pull llama2            # 安装默认模型
python app.py --list-models   # 查看已安装模型
python app.py --model qwen2   # 换用已安装的模型
```
</details>

<details>
<summary><b>回答不引用知识库内容</b></summary>

说明检索没有命中，可以：

1. 调低阈值提高召回：`python app.py --min-score 0.1`
2. 增加注入条数：`python app.py --top-n 5`
3. 给知识条目补充**与提问用词接近**的标签和问题表述
4. 用 `python app.py --check` 确认知识库确实被加载
</details>

<details>
<summary><b>回答偶尔不好 / 想换模型</b></summary>

本地小模型能力有限，可换更大的模型：`python app.py --model qwen2:7b`。
也可以调整 `temperature`（越低越稳定）与 `system_prompt`。
</details>

<details>
<summary><b>界面中文显示为方块</b></summary>

程序会按平台自动选择中文字体（Windows 用 Microsoft YaHei UI）。Linux 上若缺字体：

```bash
sudo apt install fonts-noto-cjk
```
</details>

<details>
<summary><b>知识库文件被写坏了怎么办</b></summary>

程序不会静默覆盖：

- 文件**内容损坏**（JSON 语法错误）时，会把原文件备份为 `knowledge_base.json.corrupt-<时间戳>`，
  修复该备份后改回 `knowledge_base.json` 即可；
- 文件**读取失败**（被占用、权限不足、网络盘掉线）时，程序**不会改动原文件**，
  但会拒绝写回并给出提示；排除故障后点界面上的「刷新」即可恢复。
</details>

---

## 开发说明

### 设计约定

- 核心逻辑与界面解耦：`career_qa` 包内除 `gui.py` 外均不依赖 tkinter，可独立测试与复用；
- 只依赖标准库 + `requests`，检索算法自行实现，不引入分词器；
- 依赖通过构造函数注入（`OllamaClient(session=..., sleep=...)`），便于 Mock；
- 配置对象 `AppConfig` 不可变，用 `with_overrides()` 派生。

### 代码风格

- 全部公开函数带类型标注与中文文档字符串；
- 提交前请确保 `python -m unittest discover -s tests -t .` 全绿。

### Git 提交信息建议

```
<类型>: <简短描述>

feat     新功能
fix      缺陷修复
refactor 重构
docs     文档
test     测试
chore    构建与杂项
```

---

## 许可证

本项目基于 [MIT License](LICENSE) 发布。
