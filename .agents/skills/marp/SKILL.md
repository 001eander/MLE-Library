---
name: marp
description: Create, render, and inspect self-contained narrative Marp slide decks from MLE-Library knowledge while keeping source traceability in wiki pages and generated presentations in the wiki. Use when the owner asks for slides, a presentation, a talk, or Marp HTML/PDF/PPTX export.
---

# Marp

Create presentations as Markdown under `wiki/views/decks/` using lowercase kebab-case filenames. Read the relevant wiki pages and sources before outlining the deck, and keep generated render output in `.cache/` unless the owner requests otherwise.

## Deck skeleton

Start each deck with frontmatter and split later slides with a horizontal ruler on its own line:

```markdown
---
marp: true
theme: default
size: 16:9
paginate: true
title: 演示标题
---

<!-- _class: lead -->
<!-- _paginate: false -->

# 演示标题

副标题或所有者

---

## 第一张内容幻灯片

- 每张幻灯片聚焦一个主旨
- 提供简洁的论据
```

Marp Core provides the built-in `default`, `gaia`, and `uncover` themes. Size presets are theme-dependent; `16:9` is the default choice for this project. The `marp` directive enables Marp in VS Code, while the other global and local directives also affect CLI output.

## Directives

Global directives such as `theme`, `size`, `style`, `headingDivider`, `math`, and deck metadata apply to the whole presentation. Frontmatter is equivalent to directives set on the first slide.

Local directives include `paginate`, `header`, `footer`, `class`, `backgroundColor`, `backgroundImage`, and `color`. They apply to the current slide and are inherited by following slides. Add an underscore prefix to limit a directive to the current slide:

```markdown
<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _backgroundColor: #123456 -->
```

Use `<!-- class: lead -->` without the underscore only when all following slides should inherit that class. Reset inherited headers or footers with an empty value such as `<!-- footer: '' -->`.

`headingDivider: 2` automatically splits before level-1 and level-2 headings; an array such as `headingDivider: [1, 3]` selects exact levels. Avoid combining automatic heading dividers with manual rulers unless the extra split behavior is intentional.

## Authoring rules

- Write slide titles and body copy in concise Simplified Chinese. Preserve source quotations and established technical terms where needed for precision.
- 默认面向没有读过本 repo 的读者，用连贯讲解组织 deck，使其脱离仓库单独阅读也能理解问题、机制、证据和适用边界。
- 每张幻灯片聚焦一个主旨。模型机制、结构假设和设计取舍优先于榜单名次，排名只作为支持讲解的证据。
- 不要直接把结果表格搬入幻灯片。优先使用简短讲解、示意图和必要的简单图表。
- 内容页默认不放来源、页码或引用脚注。末尾设置“延伸阅读”，列出相关 wiki 页面；精确来源定位继续保存在被引用的 wiki 页面中。
- Clearly label synthesis, uncertainty, version-sensitive guidance, and hypotheses.
- Use `# <!-- fit --> Heading` sparingly for a heading that must fit on one line.
- Treat speaker notes in HTML comments as optional; do not place essential explanations only in notes.
- Do not assume Obsidian-specific syntax such as Mermaid code blocks will render in Marp. A `mermaid` fence exports as a code block (verified with Marp CLI 4.5.1); use native Markdown or a pre-rendered asset. For the diagram workflow — inline Mermaid in wiki notes, pre-rendered SVG in decks — use the `visualizer` skill.

Marp uses `$...$` for inline math and `$$...$$` for display math. When math is required, declare `math: mathjax` in frontmatter for deterministic rendering; MathJax is the current default, and `math: katex` is the alternative.

Fragmented lists use `*` for bullets or `1)` for ordered items. They appear incrementally only in HTML with the default `bespoke` template; PDF and PPTX render them as normal lists. Do not use them when hidden items would make exported slides misleading.

Most HTML tags are disabled by default. `<style>` and `<br />` are the default exceptions; do not enable arbitrary HTML for untrusted deck content.

## Images and assets

Store new long-lived images in `raw/assets/<deck-slug>/` only with the owner's approval. Never modify original source assets.

Marp extends image alt text with sizing keywords:

```markdown
![w:640](path/to/diagram.png)
![width:800px height:450px](path/to/chart.png)
![bg contain](path/to/background.png)
```

Use `bg cover` to fill the slide or `bg contain` to fit the whole image; `cover` is the default background sizing. Advanced multiple and split backgrounds require inline SVG slide support and are less portable, so avoid them unless the target renderer has been verified.

Browser-based conversion blocks local files by default. When exporting trusted project decks with local assets, use `--allow-local-files` deliberately and inspect for blocked-resource warnings. Never enable it for untrusted Markdown.

## Render and export

Prefer an existing project-pinned Marp CLI when available. For a one-shot render without adding a dependency, use the official CLI through `npx` and record the reported version:

```bash
npx --yes @marp-team/marp-cli@latest --version
npx --yes @marp-team/marp-cli@latest wiki/views/decks/example.md -o .cache/marp/example/example.html
npx --yes @marp-team/marp-cli@latest wiki/views/decks/example.md -o .cache/marp/example/example.pdf
npx --yes @marp-team/marp-cli@latest --allow-local-files wiki/views/decks/example.md -o .cache/marp/example/example.pdf
```

HTML is the default output. The output extension can select HTML, PDF, PPTX, PNG/JPEG, or presenter-note TXT output. PDF, PPTX, and image conversion require a compatible Chrome, Edge, or Firefox installation.

Regular PPTX slides are pre-rendered and not generally editable. `--pptx-editable` is experimental, has lower visual reproducibility, requires LibreOffice Impress, and does not support presenter notes; do not use it by default.

After rendering, inspect the actual output for overflow, clipping, tiny text, broken math, missing images, and readable citations. HTML success alone does not verify PDF/PPTX layout. Do not claim visual verification unless the rendered slides were actually opened and inspected.

After creating or changing a deck, update `wiki/index.md` and `wiki/log.md` as part of the same supervised knowledge change. Keep `.cache/` render artifacts out of commits.

## Official references

Based on the Marp documentation and Marp CLI 4.5.1 / Marp Core 4.4.0:

- [How to write slides](https://marp.app/docs/guide/how-to-write-slides) and [directives](https://marp.app/docs/guide/directives)
- [Heading divider](https://marp.app/docs/guide/heading-divider), [fragmented lists](https://marp.app/docs/guide/fragmented-list), and [fitting headers](https://marp.app/docs/guide/fitting-header)
- [Math typesetting](https://marp.app/docs/guide/math-typesetting)
- [Marpit image syntax](https://marpit.marp.app/image-syntax)
- [Marp CLI usage, export formats, browser requirements, and local-file security](https://github.com/marp-team/marp-cli#basic-usage)
