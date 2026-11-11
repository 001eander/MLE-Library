---
name: marp
description: Create Marp slide decks from MLE-Library knowledge while preserving citations and keeping generated presentations in the wiki. Use when the owner asks for slides, a presentation, or a talk.
---

# Marp

Create presentations as Markdown under `wiki/views/decks/` using lowercase kebab-case filenames.

Start each deck with:

```yaml
---
marp: true
theme: default
paginate: true
size: 16:9
title: Presentation Title
---
```

## Authoring rules

- Read the relevant wiki pages and sources before outlining the deck.
- Keep one main idea per slide and prefer concise bullets, diagrams, and tables over paragraphs.
- Use `<!-- _class: lead -->` for a title or section slide only when useful.
- Put source citations on the slides where claims appear and include a final sources slide with links or paths.
- Clearly label synthesis, uncertainty, version-sensitive guidance, and hypotheses.
- Store new long-lived images in `raw/assets/<deck-slug>/` only with the owner's approval. Never modify original source assets.
- Treat speaker notes in HTML comments as optional; do not hide essential citations only in notes.

After creating a deck, update `wiki/index.md` and `wiki/log.md` as part of the same supervised knowledge change. Do not claim that Marp rendering was visually verified unless the rendered slides were actually inspected.
