# MLE-Library

一个由人类监督、Pi 图书管理员维护的 Machine Learning Engineering 知识库。人用 Obsidian 阅读和与 Pi 对话，AutoResearch 通过只读 MCP 获取证据和综合回答。

## 核心约定

- `raw/` 是不可变原始资料层：PDF、Web Clipper Markdown 和原始图片。
- `wiki/` 是 Pi 在人的监督下维护的知识层，重要结论必须引用原始来源。
- `AGENTS.md` 与 `.agents/skills/` 是 Schema，规定 ingest、query、lint、Obsidian、Dataview、Marp 和 Git 工作流。
- AutoResearch 只能通过只读 MCP 查询，不能直接更新 Wiki。
- 每次正式 ingest、lint 或研究沉淀都形成独立 Git commit；只读查询不改变文件或历史。

详细规则见 [AGENTS.md](AGENTS.md)，知识入口是 [wiki/index.md](wiki/index.md)。

## 依赖

- Pi（当前已按 1.5.2 的命令参数配置）
- Node.js 18 或更新版本
- Bash、ripgrep (`rg`)、Git
- Poppler 的 `pdftotext` 与 `pdftoppm`
- 可选：Obsidian、Dataview 和 Marp 社区插件

安装 Node 依赖：

```bash
npm ci
```

检查环境：

```bash
./bin/mle doctor
```

Pi 沿用当前用户的模型与认证配置。本项目不保存 API Key，也不会修改全局 Pi、Git 或 Obsidian 配置。

## 人类使用方式

从仓库根目录运行：

```bash
./bin/mle
# 或带一个初始问题
./bin/mle chat "帮我整理这个知识库"
```

启动器会定位库根目录，加载 `AGENTS.md` 和本地 Obsidian、Marp、Dataview Skills。无需先输入 `pi` 或手动装配参数。

一次性只读查询：

```bash
./bin/mle ask "这个库目前有哪些关于推理优化的知识？"
```

`ask` 子命令默认关闭 thinking。

实时关键词检索：

```bash
./bin/mle search "continuous batching"
./bin/mle search --scope wiki --limit 5 --json "LoRA"
```

当前检索是直接搜索 Markdown/文本并回读原文，不建立索引，不包含向量检索、查询扩展或语义重排。PDF 需要先提取为 Markdown。

## 摄取资料

先把新资料放进对应目录：

- 论文：`raw/papers/`
- Web Clipper 文章：`raw/articles/`
- 原始附件：`raw/assets/`

然后运行：

```bash
./bin/mle ingest raw/articles/example.md
./bin/mle ingest raw/papers/example.pdf
```

交互模式下，Pi 会讨论重点、更新来源页和相关 Wiki、维护交叉链接、Index 与 Log。你检查 diff 并确认后，它创建一个 `ingest: <来源标题>` 提交。

自动执行、事后审阅：

```bash
./bin/mle ingest --print raw/articles/example.md
```

该模式会完成摄取，但变更留给你审阅，不绕过人工确认和提交规则。

### PDF 文本与视觉阅读

PDF 摄取使用 `pdftotext -layout` 生成 `raw/extracted/<名称>.md`，记录原始路径、SHA-256 和 `## Page N` 页码标记。原始 PDF 与提取件都进入 Git。

当模型支持视觉时，可渲染需要核查的页面：

```bash
./bin/mle render raw/papers/example.pdf 4
```

渲染结果位于 `.cache/pdf/<PDF哈希>/page-4.png`，可重复生成且不进入 Git。适合检查图表、公式、表格和提取异常的页面。没有视觉能力时使用文本提取，并明确保留未核查视觉内容的限制。扫描 PDF 不提供独立 OCR 管线。

## Lint 与 Git

```bash
# 只读报告，不写文件、不提交
./bin/mle lint --dry-run

# 交互巡检、确认修复、更新 Log 并做一次 lint commit
./bin/mle lint
```

Git 提交约定：

- `ingest: <来源标题>`：一份资料及其所有相关更新。
- `lint: <摘要>`：一次正式巡检及接受的修复；无页面修复时也记录巡检结果。
- `query: <摘要>`：用户明确要求沉淀的查询成果。
- 其他维护按逻辑变更分别提交。

Pi 提交前检查工作区，只暂存本次任务文件，不包含 `node_modules/`、`.cache/`、Pi 会话或临时文件，也不会自动 push。

## Obsidian

把仓库根目录作为 Vault 打开。已配置：

- Wikilink；
- 重命名时自动更新链接；
- 附件默认保存到 `raw/assets/`。

入口为 `wiki/index.md`，动态视图位于 `wiki/views/catalog.md`，Marp 文稿放在 `wiki/views/decks/`。Dataview 和 Marp 社区插件需要你在 Obsidian 中自行确认安装和启用；插件只是展示层，Markdown 才是知识本身。

## 只读 MCP

启动：

```bash
./bin/mle mcp
```

这是本地 stdio 服务，不监听网络端口。AutoResearch 的配置片段如下，将路径替换为实际库路径：

```json
{
  "mcpServers": {
    "mle-library": {
      "command": "/absolute/path/to/MLE-Library/bin/mle",
      "args": ["mcp"]
    }
  }
}
```

提供四个工具：

| 工具 | 用途 |
| --- | --- |
| `search_knowledge` | 按 `wiki`、`raw` 或 `all` 实时检索，返回路径、行号、片段和匹配信息 |
| `read_document` | 读取指定行段、PDF 提取件或已有图片，拒绝目录逃逸 |
| `get_context` | 检索并回读原文，按 `max_chars` 返回有上限的证据包 |
| `ask_librarian` | 启动只读、无会话 Pi，综合回答并给出引用 |

MCP 不提供写入、摄取、页面渲染、索引更新或 Git 提交能力。`ask_librarian` 只能使用 `read/grep/find/ls`，禁用扩展和其他 MCP，并默认关闭 thinking；默认超时 120 秒，输出上限 1 MB。

AutoResearch 的研究成果留在它自己的研究区。需要进入正式 Wiki 时，由你通过 Pi 图书管理员审阅、整理和提交。
