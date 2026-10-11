# 更新日志

本文件记录本项目的重要变更，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [2.0.0] - 2026-10-11

以「修复缺陷 + 工程化」为主的版本。核心问答能力不变，但修复了两个导致程序无法正常使用的
致命缺陷，并把单文件脚本重构为可测试的应用包。

### 修复（致命）

- **多轮对话必然崩溃**：`OllamaClient.generate_response` 用 `(prompt, tuple(history))` 作缓存键，
  而 `history` 是 `[{"role": ..., "content": ...}]`，`tuple(history)` 仍含 dict，作为字典键
  不可哈希 —— 第二轮对话必定抛 `TypeError: unhashable type: 'dict'`。
  现改为对 `(model, temperature, prompt, history)` 做 JSON 稳定序列化后取键。
  回归测试：`tests/test_llm_client.py::CacheTests::test_multi_turn_history_does_not_crash`。
- **流式输出永远为空**：原实现按 SSE 协议查找 `data: ` 前缀，但 Ollama `/api/chat`
  返回的是 **NDJSON**（每行一个完整 JSON 对象），前缀判断永远为假，流式回答恒为空串。
  现按行 `json.loads` 解析，同时兼容部分网关可能添加的 `data: ` 前缀。
  回归测试：`tests/test_llm_client.py::IterChunksTests`。
- **程序在当前环境无法启动**：`app.py` 顶部 `import pandas as pd` 是死导入（全文件未使用），
  而运行环境并未安装 pandas。已移除该导入，运行时依赖收敛为 `requests` 一个第三方包。

### 修复（稳定性）

- 所有 HTTP 请求补充连接超时与读取超时，并区分连接失败 / 模型不存在 / 服务端错误，
  统一包装为 `LLMError` 子类；网络类失败支持指数退避重试。
- 知识库 JSON **语法损坏**时不再崩溃：自动备份为 `*.corrupt-<时间戳>` 并以空库降级启动；
  同一秒内二次损坏会追加序号，不再覆盖上一次的备份。
- **读取故障不再被误判为文件损坏（重要）**：旧实现把 `OSError` 与 `JSONDecodeError`
  合并处理，任何瞬时读取失败（文件被其他进程独占、权限变化、网络盘掉线）都会把
  **内容完好**的文件改名备份、内存降级为空库，用户随后添加一条知识即造成真实数据丢失。
  现在读取故障绝不移动原文件，并禁止写回该文件，避免一次读取抖动演变成数据丢失。
- 知识库保存改为「临时文件 + `os.replace`」原子写入，中途失败不会写坏原文件，
  且失败时清理写了一半的 `.tmp` 文件，不在用户目录里堆积垃圾。
- 新增 / 修改 / 删除条目在落盘失败时**回滚内存状态**，不再出现「界面提示保存失败、
  但条目已进内存、重试却被判为重复条目」的错乱状态。
- 修复 `allow_duplicate=True` 时新增条目被惰性加载覆盖丢失的问题。
- 修复未加载内存副本就调用 `save()` 会把磁盘内容清空的风险。
- 修复 `import_kb` 后「添加知识」实际写入导入文件、却让用户以为在改原库的语义陷阱，
  现在会明确提示当前知识库路径。
- 修复 GUI 后台线程直接读取 Tk 变量（`selected_tags` 等）的线程安全隐患，
  改为发起请求前快照、结果统一经 `queue.Queue` 回主线程渲染。
- 修复消息泵在单条消息渲染失败时静默停摆、界面永久卡在「正在生成回答…」的问题。
- 修复状态栏在回答结束后覆盖掉标签筛选上下文的问题。
- 修复流式响应从不关闭导致的 HTTP 连接泄漏：正常读完、消费方中途放弃
  （`GeneratorExit`）、解析报错，以及重试时被丢弃的 5xx 响应，现在都会关闭并归还连接。
- 修复检索打分饱和：包含关系原先固定给 0.95 分，叠加标签加分后普遍顶到 1.0，
  排序与 `min_score` 阈值同时失去区分度；现在按「提问长度 / 标题长度」给出连续分数。
- 标签筛选未命中时，提示词会说明「已在标签 X 范围内检索但没有匹配」，
  避免用户误以为知识库中根本没有相关内容。
- 流式解析遇到无法解析的脏行时记录 warning，不再静默吞掉导致回答「缺字」且无从排查。
- `_messages()` 返回历史副本而非内部列表别名，避免调用方意外改写对话上下文。
- 请求进行中禁用「发送」按钮，避免连点造成多路请求并发改写同一份对话历史。
- 移除可变默认参数 `history=[]`；响应缓存改为有上限的 LRU（默认 128 条），避免无界增长。

### 新增

- **中文相似度检索**：字符 bigram Dice 系数 + `difflib` 序列相似度混合打分，
  并叠加标签命中加分与相似度阈值。改写提问（如「职业规划怎么做」）也能命中，
  无关提问不再污染提示词。
- **可配置化**：新增 `AppConfig`（不可变数据类），按
  `默认值 → config.json → 环境变量 → 命令行` 逐层覆盖，并附 `config.example.json` 模板。
- **命令行功能**：`--check` 环境自检、`--list-models` 列出模型、`--kb/--model/--ollama-url/--top-n/--min-score/--no-stream` 等参数、
  `--version`。
- **知识库容错与格式兼容**：支持 `{"entries": [...]}` 包装与英文键 `question/answer/tags`；
  单条格式错误跳过并告警；新增 `update_entry`、`remove_entry`、`tag_counts`、`filter_by_tags`。
- **自动化测试**：182 个用例，覆盖配置、知识库、HTTP 客户端、对话编排与 GUI 端到端，
  全部使用 Mock，**不需要启动 Ollama**；其中包含针对上述每一项缺陷的回归用例
  （含此前零覆盖的失败路径：读取故障、落盘失败、连接释放）。
- **CI**：GitHub Actions 在 Python 3.9 / 3.11 / 3.12 上运行测试。
- **文档**：新增 `README.md`（安装、配置、命令行、知识库格式、FAQ）、
  `CONTRIBUTING.md`（开发环境、分支与提交规范、代码约定、测试要求、常见任务指引、
  发布流程）与本更新日志；README 中的项目结构、数据流与检索流程图改用
  **Mermaid** 绘制，在 GitHub 上可直接渲染。
- **流式界面输出**：接上修好的流式解析，回答逐字显示；新增「清空对话」按钮与状态栏。

### 变更（不兼容）

- `app.py` 从「全部实现」变为兼容入口，核心逻辑迁移到 `career_qa` 包。
  旧的 `from app import KnowledgeBase, OllamaClient, DialogueManager, ChatGUI` 仍然可用。
- `KnowledgeBase.search_answers()` → `KnowledgeBase.search()`，返回带相似度得分的
  `SearchHit` 列表，而非裸 dict。
- `KnowledgeBase.get_tags_stats()` → `KnowledgeBase.tag_counts()`。
- `KnowledgeBase.add_entry()` 默认拒绝重复问题（规范化后比较），
  如需强制添加请传 `allow_duplicate=True`。
- `OllamaClient.generate_response()` 保留并兼容旧签名；新增 `chat()` 与 `stream_chat()`。
- 移除了对 `pandas` 的依赖。

### 工程化

- 新增 `.gitignore`，忽略 `.venv/`、`build/`、`dist/`、`.idea/`、`__pycache__/`、`config.json` 等。
- 新增 `requirements.txt` 与 `LICENSE`（MIT）。
- 从版本控制中移除 `.idea/` 与 `dist/knowledge_base.json`。
- 修正 `智能问答系统.spec`：原先打包的是并不存在的 `knowledge_base.xlsx`，
  现改为打包 `knowledge_base.json` 与 `config.example.json`；
  删除与之重复的 `ChatApp.spec`。
- 修复 PyInstaller 打包后数据文件路径问题：冻结运行时以 exe 所在目录为数据根目录，
  首次启动会把内嵌的示例知识库释放到 exe 旁边，便于用户就地编辑。
- 版本号收敛为**单一来源**：`career_qa/__init__.py` 的 `__version__`，
  `--version` 输出与包元数据都从它读取，不再有两处硬编码不同步的风险。

---

## [1.0.0] - 初始版本

- 单文件 `app.py`：tkinter 界面 + JSON 知识库 + Ollama 问答。
- 标签筛选、添加 / 导入 / 导出知识库。
