# 贡献指南

感谢你愿意为「职业生涯规划智能问答系统」做出贡献！本文档说明如何搭建开发环境、
需要遵循哪些约定，以及怎样让改动顺利合并。

无论是修一个错别字、往示例知识库补一条内容，还是重构检索算法，都欢迎。

## 目录

- [行为准则](#行为准则)
- [你可以如何贡献](#你可以如何贡献)
- [开发环境搭建](#开发环境搭建)
- [项目结构速览](#项目结构速览)
- [开发工作流](#开发工作流)
- [代码规范](#代码规范)
- [测试规范](#测试规范)
- [提交信息规范](#提交信息规范)
- [常见任务指引](#常见任务指引)
- [文档维护](#文档维护)
- [发布流程](#发布流程)
- [需要帮助？](#需要帮助)

---

## 行为准则

- 尊重每一位参与者，就事论事地讨论技术问题。
- 不接受人身攻击、歧视性言论、骚扰行为。
- 对新手友好：提问没有「蠢问题」，回答时给出上下文而不是一句「去看文档」。

## 你可以如何贡献

| 方式 | 做法 |
|---|---|
| 🐞 报告缺陷 | 开 issue，附上「复现步骤 + 期望结果 + 实际结果 + 环境信息」 |
| 💡 功能建议 | 开 issue 说明使用场景，以及你希望它解决什么问题 |
| 📝 改进文档 | 直接提 PR，修正错别字、补充示例、澄清歧义 |
| 💻 提交代码 | 见下方的[开发工作流](#开发工作流) |
| 📚 扩充示例知识库 | 补充真实的职业生涯规划问答条目 |
| 🌏 翻译 | 目前界面为简体中文，如有多语言需求请先开 issue 讨论 |

### 报告缺陷时请附上

```bash
python app.py --check     # 环境自检输出（配置 / 知识库 / Ollama 连通性）
python --version          # Python 版本
ollama --version          # Ollama 版本
```

再加上：操作系统、使用的模型名（如 `llama2` / `qwen2`）、以及完整的报错文本。

> ⚠️ 提交日志前请先**移除个人隐私信息**（用户名、内网地址、知识库中的敏感内容）。

---

## 开发环境搭建

### 1. 环境要求

- **Python 3.9+**（推荐 3.11；CI 在 3.9 / 3.11 / 3.12 上验证）
- **tkinter**（Python 官方安装包自带；Debian/Ubuntu 需 `sudo apt install python3-tk`）
- **Ollama**（仅端到端联调时需要，**跑测试不需要**）

### 2. 克隆与安装

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

### 3. 确认环境可用

```bash
python -m unittest discover -s tests -t . -v    # 182 个测试应全绿
python app.py --check                           # 环境自检
python app.py                                   # 启动图形界面
```

> 在无图形界面的服务器上，`tests/test_gui_smoke.py` 会自动跳过，其余测试照常运行。
> 想在 Linux 无头环境跑 GUI 测试：`xvfb-run -a python -m unittest discover -s tests -t .`

### 4. 可选：准备本地模型

```bash
ollama serve
ollama pull llama2        # 或换成 qwen2、gemma2 等
```

---

## 项目结构速览

完整的结构与数据流图见 [README](README.md#项目结构)。几条关键约定：

- **`career_qa/` 除 `gui.py` 外都不依赖 tkinter**，必须保持可独立导入与测试；
- 依赖方向是单向的：`cli → gui → dialogue → {knowledge_base, llm_client, config}`，
  **不要反向依赖**，也不要在 `knowledge_base` 里调用大模型；
- 所有对外行为都要有对应的测试。

```
career_qa/
├── cli.py              命令行入口、环境自检、打包后释放内嵌数据
├── config.py           配置加载、类型转换与校验
├── knowledge_base.py   知识库读写（原子写入）+ 中文相似度检索
├── llm_client.py       Ollama HTTP 客户端（NDJSON 流式、LRU 缓存、重试）
├── dialogue.py         检索 + 提示词构造 + 多轮历史裁剪
└── gui.py              tkinter 界面与线程安全消息泵
```

---

## 开发工作流

### 1. 先开 issue（可选但推荐）

较大的改动（新功能、重构、行为变更）建议先开 issue 讨论方向，
避免写完 PR 才发现方案不被接受。

### 2. 从最新 `master` 拉分支

```bash
git checkout master
git pull
git checkout -b feat/support-qwen2
```

分支命名约定：

| 前缀 | 用途 | 示例 |
|---|---|---|
| `feat/` | 新功能 | `feat/vector-search` |
| `fix/` | 缺陷修复 | `fix/stream-close-leak` |
| `docs/` | 文档 | `docs/add-contributing` |
| `refactor/` | 重构（不改变行为） | `refactor/split-gui` |
| `test/` | 只改测试 | `test/add-cli-cases` |
| `chore/` | 构建、依赖、杂项 | `chore/bump-requests` |

### 3. 开发与自测

```bash
python -m unittest discover -s tests -t . -v
python -m compileall -q career_qa app.py tests
python app.py --check
```

**提交 PR 前请确保测试全绿**，CI 也会跑同样的检查。

### 4. 提交并推送

```bash
git add <具体文件>          # 避免 git add -A 误提交构建产物
git commit -m "fix(kb): 读取故障不再改名原文件"
git push -u origin fix/stream-close-leak
```

### 5. 开 Pull Request

PR 描述里请写清楚：

- **要解决的问题**（关联 issue 编号，如 `Closes #12`）
- **改动方式**，以及为什么选这个方案
- **如何验证**（跑了哪些测试、有没有手工验证步骤）
- **行为变更 / 破坏性变更**（如果有，务必显式说明）

#### PR 检查清单

- [ ] `python -m unittest discover -s tests -t .` 全绿
- [ ] 新功能附带测试；修复缺陷附带**回归测试**（能复现修复前的失败）
- [ ] 覆盖了失败路径（异常、超时、写盘失败等），而不只是成功路径
- [ ] 行为变更已更新 `README.md`，用户可见的改动已更新 `CHANGELOG.md`
- [ ] 提交信息符合[规范](#提交信息规范)
- [ ] 没有提交 `build/`、`dist/`、`.venv/`、`config.json`、`__pycache__/`
- [ ] 没有引入新的第三方依赖（如确有必要，请在 PR 里说明理由）

---

## 代码规范

### 通用

- 遵循 **PEP 8**：4 空格缩进，函数与类之间空两行。
- 每个模块以 `from __future__ import annotations` 开头（兼容 Python 3.9 的类型标注）。
- **所有公开函数、方法都要有类型标注**，并写**中文文档字符串**。
- 尽量保持单行不超过 100 字符。
- 命名：模块/函数/变量用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_SNAKE_CASE`，
  私有成员加单下划线前缀。

### 设计约定

1. **核心层不依赖 GUI**。`knowledge_base`、`llm_client`、`dialogue`、`config`
   不得 `import tkinter`。
2. **依赖注入以便测试**。需要与外部交互的对象通过构造函数传入，例如
   `OllamaClient(session=..., sleep=...)`，测试里可替换为伪造实现。
3. **配置集中管理**。不要把模型名、地址、阈值散落在代码里；加进
   [`AppConfig`](career_qa/config.py)，并同步更新 `config.example.json` 与 README 的配置表。
4. **`AppConfig` 是不可变的**，用 `with_overrides()` 派生新配置，不要就地修改。
5. **错误要分类**。向外抛出 `LLMError` / `KnowledgeBaseError` 等语义化异常的子类，
   不要把裸 `requests` 异常泄漏给上层。
6. **不在库代码里 `print()`**。需要输出用 `logging`；CLI 的结果输出除外。
7. **标准库优先**。检索、解析、缓存等能力目前都用标准库实现，
   引入新依赖前请先说明为什么标准库做不到。

### GUI 专属规则（重要）

`gui.py` 里有一条硬性约束：

> **绝不能在后台线程里读写任何 Tk 控件或 `tk.StringVar`。**

正确做法：发起请求前在主线程做数据快照，工作线程只处理纯 Python 数据并通过
`self._queue` 投递消息，渲染统一由主线程的 `_pump()` 完成。
修改交互逻辑时请先读 `ChatGUI._worker` / `_pump` / `_handle_message` 三个方法。

---

## 测试规范

- 测试框架是**标准库 `unittest`**，不引入 pytest。
- **测试不得依赖 Ollama 或网络**：HTTP 一律用 `tests/test_llm_client.py` 里的
  `FakeSession` / `FakeResponse` 伪造。
- 文件系统操作使用 `tempfile.TemporaryDirectory()`，不要污染仓库。
- 一个测试只验证一件事；断言要具体（比较值，而不是只断言「没抛异常」）。
- **失败路径的测试尤其重要**。历史上漏掉的缺陷几乎都在失败路径上：
  读取故障被误判为文件损坏、落盘失败后内存未回滚、响应未关闭导致连接泄漏。
- 修 bug 时，请先写一个**能复现该 bug 的测试**，确认它失败，再修代码让它通过。

```bash
# 全部测试
python -m unittest discover -s tests -t . -v

# 单个模块
python -m unittest tests.test_knowledge_base -v

# 单个用例
python -m unittest tests.test_knowledge_base.ReadFailureTests -v
```

### 性能基线

检索是 O(条目数 × 问题长度) 的字符串比较。改动 `similarity()` / `score()` 后，
请用一万条规模确认没有明显退化（当前约 200 ms/次）：

```python
python -c "
import json, random, tempfile, time
from pathlib import Path
from career_qa.knowledge_base import KnowledgeBase
random.seed(0)
entries = [{'问题': f'第{i}个职业规划问题', '答案': 'x'*80, '标签': [f'T{i%50}']} for i in range(10000)]
p = Path(tempfile.mkdtemp()) / 'kb.json'
p.write_text(json.dumps(entries, ensure_ascii=False), encoding='utf-8')
kb = KnowledgeBase(p)
t = time.perf_counter(); hits = kb.search('职业规划问题', top_n=3, min_score=0.25)
print(f'{(time.perf_counter()-t)*1000:.0f} ms, {len(hits)} hits')
"
```

---

## 提交信息规范

采用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)，
**描述用简体中文**：

```
<类型>(<范围>): <简短描述，不超过 50 字>

<正文：为什么这么改，而不是重复改动内容>

<脚注：BREAKING CHANGE 说明 / Closes #issue>
```

**类型**

| 类型 | 含义 |
|---|---|
| `feat` | 新功能 |
| `fix` | 缺陷修复 |
| `docs` | 文档 |
| `refactor` | 重构（不改变对外行为） |
| `perf` | 性能优化 |
| `test` | 测试 |
| `build` | 构建脚本、依赖 |
| `ci` | CI 配置 |
| `chore` | 其他杂项 |
| `revert` | 回滚 |

**范围**（可选）：`config`、`kb`、`llm`、`dialogue`、`gui`、`cli`、`docs`、`tests`

**示例**

```
fix(llm): 修复流式响应未关闭导致的连接泄漏

stream=True 的 Response 持有连接，必须显式 close() 才会归还连接池。
之前正常读完、中途放弃、以及重试丢弃的 5xx 响应都没有关闭。

新增 ResourceReleaseTests 覆盖上述三条路径。

Closes #18
```

破坏性变更在类型后加 `!`，并在脚注中说明：

```
refactor(kb)!: search() 改为返回 SearchHit 列表

BREAKING CHANGE: search_answers() 已移除，调用方需改用 search()。
```

---

## 常见任务指引

<details>
<summary><b>新增一个命令行选项</b></summary>

1. 如果需要持久化配置，在 [`career_qa/config.py`](career_qa/config.py) 的
   `AppConfig` 中加字段（带默认值与注释），并在 `__post_init__` 里加上校验；
2. 若支持环境变量覆盖，在 `ENV_FIELDS` 中登记映射；
3. 在 `build_parser()` 里加 `parser.add_argument(...)`；
4. 在 `load_config()` 的 `cli_data` 中把参数合并进配置（`None` 表示不覆盖）；
5. 需要类型转换的值，在 `_coerce()` 中处理；
6. 补 `tests/test_config.py` 用例，覆盖「命令行 > 环境变量 > 配置文件」的优先级；
7. 更新 `config.example.json` 与 README 的配置表。

</details>

<details>
<summary><b>调整检索策略或打分公式</b></summary>

1. 改 [`career_qa/knowledge_base.py`](career_qa/knowledge_base.py) 的
   `similarity()`（提问 vs 标题）或 `score()`（叠加标签加分）；
2. **保证返回值落在 `[0, 1]`**，且分数要能区分不同质量的匹配
   （历史上固定给高分导致排序与阈值同时失效）；
3. 注意 `min_score` 的语义是「低于此值视为不相关」；
4. 补 `tests/test_knowledge_base.py` 的 `SearchTests`，至少覆盖：
   完全匹配排第一、改写提问能命中、无关提问被过滤、分数单调递减；
5. 按上一节的方法验证一万条规模下的性能。

</details>

<details>
<summary><b>给知识条目新增字段</b></summary>

1. 在 `Entry` dataclass 中加字段，给出默认值；
2. 同步更新 `Entry.from_dict()`（**保持向后兼容**：旧文件缺该字段时要能读）与 `to_dict()`；
3. 如需筛选，在 `KnowledgeBase` 上加方法，并考虑是否要参与 `score()`；
4. 补测试：正常解析、字段缺失、类型错误；
5. 更新 README 的[知识库格式](README.md#知识库格式)章节。

</details>

<details>
<summary><b>修改界面交互</b></summary>

1. 先读 `ChatGUI._worker` / `_pump` / `_handle_message`，理解消息泵；
2. **不要在后台线程碰 Tk**（见 [GUI 专属规则](#gui-专属规则重要)）；
3. 长耗时操作要设置 `_busy` 状态并禁用按钮，避免并发请求；
4. 补 `tests/test_gui_smoke.py` 用例——它会真实创建窗口并驱动事件循环，
   用 `self.wait_idle(gui)` 等待后台完成；
5. 新增的可见文案请用简体中文，并与现有风格保持一致。

</details>

<details>
<summary><b>报告「程序卡死」类问题</b></summary>

这类问题请额外提供：

- 卡死时界面状态栏的文字（例如是否停在「正在生成回答…」）；
- 是发送后立即卡，还是生成到一半卡住；
- Ollama 是否在同时处理其他请求。

界面线程安全的实现细节在 `gui.py` 的模块文档字符串里有说明。

</details>

---

## 文档维护

- **用户可见的行为变更**必须同时更新 `README.md` 与 `CHANGELOG.md`；
- `CHANGELOG.md` 按版本分段，分类为「修复 / 新增 / 变更（不兼容）/ 工程化」，
  格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
- README 中的命令输出示例、测试数量等**易腐坏的数字**，
  改动后请实际运行命令核对，不要凭记忆写；
- 文档里的图表使用 **Mermaid**（GitHub 原生渲染），提交前请在预览里确认能正常显示。

---

## 发布流程

1. 确认 `master` 上测试全绿、CI 通过；
2. 更新版本号——**唯一来源是** [`career_qa/__init__.py`](career_qa/__init__.py)
   的 `__version__`（`--version` 与包元数据都从这里读取）；
3. 在 `CHANGELOG.md` 顶部新增该版本段落，并把「未发布」的内容归位；
4. 提交：`chore: 发布 vX.Y.Z`；
5. 打标签并推送：

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin master --follow-tags
   ```

6. 在 GitHub 上创建 Release，把 CHANGELOG 中该版本的内容粘过去。

版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)：
不兼容的 API 变更升主版本，向后兼容的新功能升次版本，向后兼容的修复升补丁版本。

---

## 需要帮助？

- 先看 [README 的常见问题](README.md#常见问题)；
- 再搜一搜现有的 issue，避免重复提问；
- 仍然没解决就开 issue，写清楚你的目标、尝试过的做法和实际结果。

感谢你的贡献！🎉
