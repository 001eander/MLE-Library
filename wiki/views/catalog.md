---
title: Dynamic Catalog
type: view
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - dataview
---

# Dynamic Catalog

Requires the Dataview community plugin. The curated source of truth remains [[index]].

```dataview
TABLE type, updated, status, default(verified, "unknown") AS Verified
FROM "wiki"
WHERE type AND !contains(list("index", "overview", "view"), type)
SORT updated DESC
```
