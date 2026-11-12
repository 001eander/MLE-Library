---
name: visualizer
description: Design diagrams for MLE-Library — inline Mermaid in Obsidian wiki notes, pre-rendered SVG for Marp decks. Use when asked to draw, diagram, or visually explain a model structure, data flow, or comparison, and when a figure must be embedded in a deck.
---

# Visualizer

Diagrams are presentation, never the source of truth. A figure may restate a mechanism that is already documented in a wiki page; it must not introduce claims the page does not support. Label synthesis and uncertainty the same way the page does.

## Two rendering contexts

| Where | Mermaid support | What to do |
| --- | --- | --- |
| Obsidian wiki page (`wiki/**/*.md`) | native `mermaid` code fence | write the block inline; no asset, no pre-render |
| Marp deck (`wiki/views/decks/*.md`) | none — the fence renders as a code block | pre-render to SVG and embed with an explicit width |
| Deck exported to PDF/PPTX | same as the deck | pre-render; local images additionally require `--allow-local-files` |

Verified 2026-11-12 with Marp CLI 4.5.1 / Marp Core 4.4.0: a `mermaid` fence exported to PNG stays a syntax-highlighted code block. Obsidian renders the same fence normally, so only the deck path needs pre-rendering.

## Decide before drawing

- One idea per figure: a flow, a structure, or a comparison. Not all three.
- Keep 4–8 nodes. Past that, split the figure or move detail into the surrounding text.
- If a Markdown table carries the content just as well, use the table.
- State in the caption what the figure shows and which parts are synthesis.

## Legibility budget for slides

On-slide label size ≈ Mermaid base font (14 px) × displayed width ÷ diagram natural width. The natural width comes from the layout, not from the export format, so adding detail shrinks the text.

At a 1140 px embed width:

| Diagram natural width | On-slide label |
| --- | --- |
| 552 px | ~29 px |
| 820 px | ~19 px |
| 1168 px | ~14 px |

Keep natural width under about 850 px for a 1140 px slide. Read the real value from the SVG `viewBox`, not from the PNG pixel size.

Ways to stay inside the budget:

- Short labels, one to four tokens; put formulas in the adjacent slide.
- Merge chained stages into one box (`Mₙ = t-softmax(Fₙ, t)` as a single node).
- Avoid seven or more ranks in an `LR` flow; a long chain always forces small text.
- Move a terminal node into the caption when the figure is already wide.

## Authoring Mermaid

- Start with `flowchart LR` or `flowchart TB`; dagre is the layout engine.
- Quote every label (`A["X"]`); use `<br/>` for line breaks and `-->|"label"|` for edge labels.
- Colour with `classDef` plus `:::name`, and keep to two or three fills. Say what the colours mean in the caption.
- Mermaid is not math-aware. Write `Hₙ₋₁` or `H_n-1`; LaTeX inside a node will not render.
- Unicode subscripts render, but consistency varies by theme and font. Check the exported figure.
- A cluster's `direction` is not reliably honoured in dagre layouts. Do not depend on `subgraph … direction LR` to place nodes.

## Pre-render pipeline

1. Source: `raw/assets/<deck-slug>/<figure>.mmd`.
2. Config: `raw/assets/<deck-slug>/cfg.json` with `{"flowchart": {"wrappingWidth": 380, "htmlLabels": true}}`. This stops labels from breaking mid-token.
3. Render, recording the reported CLI version:

```bash
npx --yes @mermaid-js/mermaid-cli@latest -i figure.mmd -o figure.svg -b white -c cfg.json
```

Prefer SVG: it stays crisp at any embed width. Use PNG only when a target renderer rejects SVG; `-s <n>` raises the raster resolution and `--size` upscales the whole image rather than enlarging the text.

4. Embed in the deck with an explicit width: `![w:1140](../../../raw/assets/<deck-slug>/<figure>.svg)`.
5. Render the deck with `--allow-local-files` when it references local assets.

## Verify

- Open the rendered slide or page, not just the HTML.
- Judge legibility at 1:1. Crop a label and compare it with the body text, or compare the diagram width against the numbers above.
- Check for clipped boxes, edge labels lying on top of another line, and boxes wider than the slide.
- Confirm the caption matches the figure, and that no claim appears in the figure alone.

## File discipline

- Deck figures live in `raw/assets/<deck-slug>/` together with their `.mmd` and `cfg.json`, so the figure can be regenerated.
- Record CLI versions and asset SHA-256 in the matching `wiki/log.md` entry.
- `.cache/` renders, temporary crops, and test decks are never committed.
- The wiki page remains the source of truth; the figure is a view of it.

## Alternatives

- Obsidian Canvas (`.canvas`) for exploratory maps that will be rearranged by hand.
- Dataview tables when the content is really metadata.
- Plain Markdown tables when the content is tabular.

## Checklist

1. Context chosen: inline fence for a wiki page, pre-rendered SVG for a deck.
2. Node count and labels inside the legibility budget.
3. Colours defined by `classDef`, meaning explained in the caption.
4. `.mmd` and `cfg.json` stored next to the asset.
5. Rendered output inspected at 1:1, versions and hashes logged.
