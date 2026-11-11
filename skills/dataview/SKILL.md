---
name: dataview
description: Create and maintain Dataview views and consistent YAML frontmatter for MLE-Library. Use for dynamic catalogs, source tables, stale-note reports, and metadata checks.
---

# Dataview

Dataview is a read-only presentation layer. Keep canonical page metadata in YAML and the curated catalog in `wiki/index.md`.

## Metadata

Use the root `AGENTS.md` frontmatter conventions. In particular:

- `type` identifies the page category.
- `created` and `updated` use ISO `YYYY-MM-DD` dates.
- `sources` contains quoted Wikilinks to source pages.
- `verified` and `applies_to` document fast-changing claims.
- `status` distinguishes `draft`, `stable`, and `archived` knowledge.

Do not add fabricated metadata merely to make a query complete. Use `default(field, value)` or `choice(...)` in queries when missing values need a readable fallback.

## Views

Place reusable Dataview pages under `wiki/views/`. Query the `wiki` folder explicitly, for example:

```dataview
TABLE type, updated, status
FROM "wiki"
WHERE type AND type != "index" AND type != "overview"
SORT updated DESC
```

For stale practice or tool pages, filter on `verified` and explain the chosen date threshold. A missing `verified` field means unknown freshness, not automatically stale or current.

Dataview queries do not replace lint citations or factual verification. Do not claim that a query renders correctly in Obsidian unless it was actually tested with Dataview enabled.
