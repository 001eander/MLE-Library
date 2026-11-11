---
title: 动态目录
type: view
created: 2026-11-11
updated: 2026-11-11
tags:
  - mle
  - dataview
---

# 动态目录

需要启用 Dataview 社区插件。经过人工整理的事实来源仍是 [[index|知识索引]]。

```dataview
TABLE WITHOUT ID file.link AS "页面", type AS "类型", updated AS "更新日期", status AS "状态", default(verified, "未知") AS "核验日期"
FROM "wiki"
WHERE type AND !contains(list("index", "overview", "view"), type)
SORT updated DESC
```
