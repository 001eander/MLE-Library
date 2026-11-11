---
name: dataview
description: Create and maintain Dataview DQL views and consistent YAML frontmatter for MLE-Library. Use for dynamic catalogs, source tables, stale-note reports, task views, and metadata checks.
---

# Dataview

Dataview is a presentation layer, not the source of truth. Keep canonical page metadata in YAML and the curated catalog in `wiki/index.md`. Prefer DQL over JavaScript for reusable views in this vault. Follow the root `AGENTS.md` language rules: use Simplified Chinese for human-readable view text and query labels while preserving metadata keys, enumerated values, field names, and source locators.

## Indexed metadata

Dataview can query only indexed data, not arbitrary note text. YAML frontmatter fields are available at page level. Inline fields use `Key:: Value`; use `[key:: value]` when annotating a specific list item or task.

Use the root `AGENTS.md` frontmatter conventions:

- `type` identifies the page category.
- `created` and `updated` use ISO `YYYY-MM-DD` dates, which Dataview parses as dates.
- `sources` contains quoted Wikilinks such as `"[[source-slug]]"`; unquoted Wikilinks are invalid YAML.
- `verified` and `applies_to` document fast-changing claims.
- `status` distinguishes `draft`, `stable`, and `archived` knowledge.

Do not invent metadata to make a query complete. `default(field, value)` and `choice(condition, left, right)` change displayed results, not the underlying notes.

Useful implicit page fields include:

| Field | Meaning |
| --- | --- |
| `file.name`, `file.path`, `file.folder` | File name and vault paths |
| `file.link` | Link to the page |
| `file.ctime`, `file.mtime` | Creation and modification timestamps |
| `file.cday`, `file.mday` | Creation and modification dates without time |
| `file.day` | Date from the file name or a `Date` field, when available |
| `file.tags`, `file.etags` | All expanded tags or explicit tags |
| `file.inlinks`, `file.outlinks` | Incoming and outgoing links |
| `file.tasks`, `file.lists` | Indexed tasks and list items |

## Build a DQL query

Use a `dataview` code block. Start with exactly one query type, optionally add fields, then scope the query and refine it:

```dataview
TABLE type, updated, status
FROM "wiki"
WHERE type AND type != "index" AND type != "overview"
SORT updated DESC
LIMIT 20
```

Query types:

- `TABLE`: Page-level rows with calculated columns and optional `AS "Header"` labels.
- `LIST`: Page links with at most one additional expression.
- `TASK`: Task-level results; checking a task in the rendered view also changes the original file.
- `CALENDAR`: One dot per result on the supplied date field; a date expression is required.

`TABLE WITHOUT ID` and `LIST WITHOUT ID` suppress the default file or group column. With `TABLE WITHOUT ID`, include `file.link` explicitly when page links are still needed.

Always consider a `FROM` scope. Folder sources include subfolders and require full vault paths without a trailing slash. Other sources include `#tag`, `[[note]]` for pages linking to a note, and `outgoing([[note]])` for pages linked from a note. Combine sources with `AND`, `OR`, and parentheses.

After the query type and optional `FROM`, data commands run in the order written:

- `WHERE expression` filters results.
- `SORT field ASC|DESC` sorts by one or more fields.
- `GROUP BY field` creates one result per key and collects original results in `rows`.
- `FLATTEN field AS name` expands an array into one result per entry.
- `LIMIT n` caps results; put it after `SORT` when selecting the top or newest n results.

After grouping, use `key` for the group value and `rows` for its members. Field swizzling such as `rows.file.link` collects that field from every member.

## Expressions and dates

- DQL equality is `=`, not `==`; inequality is `!=`.
- Use `date(today)`, `date(now)`, and `dur(90 days)` for date calculations.
- Guard missing values before comparisons. Null comparisons can be surprising; `WHERE verified AND verified < date(today) - dur(90 days)` excludes pages without `verified`.
- Use `typeof(field) = "date"` when a type check is safer than a truthiness check.
- `dateformat(date, "yyyy-MM-dd")` returns text, not a date; do not compare its result directly with a date object.
- Common functions include `default`, `choice`, `contains`, `length`, `flat`, `filter`, `map`, `sum`, and `average`.

Inline DQL such as `` `= this.updated` `` renders exactly one expression. Query types and data commands are not available inline.

Prefer DQL. Use `dataviewjs` only when DQL cannot express the view; Dataview JavaScript can access the file system, so never run untrusted copied scripts.

## MLE-Library patterns

Catalog pages while labeling missing verification data as unknown:

```dataview
TABLE WITHOUT ID file.link AS "页面", type AS "类型", updated AS "更新日期", status AS "状态", default(verified, "未知") AS "核验日期"
FROM "wiki"
WHERE type AND !contains(list("index", "overview", "view"), type)
SORT updated DESC
```

Review fast-changing pages using an explicit 90-day policy. Adjust and explain the threshold when the owner requests a different one:

```dataview
TABLE WITHOUT ID
  file.link AS "页面",
  default(verified, "未知") AS "核验日期",
  choice(!verified, "未知",
    choice(verified < date(today) - dur(90 days), "待复核", "近期已核验")) AS "时效状态"
FROM "wiki/practices" OR "wiki/tools"
WHERE status != "archived"
SORT verified ASC
```

A missing `verified` field means unknown freshness, not automatically stale or current. Do not silently convert it into a date or a factual verification claim.

Summarize pages by category:

```dataview
TABLE length(rows) AS "页面数", rows.file.link AS "文件"
FROM "wiki"
WHERE type
GROUP BY type
```

Place reusable views under `wiki/views/` and update `wiki/index.md` and `wiki/log.md` as part of the same supervised knowledge change. Queries do not replace lint citations or factual verification. Do not claim that a query renders correctly until it has actually been tested in Obsidian with Dataview enabled.

## Official references

Based on the Dataview 0.5.70 documentation:

- [Metadata](https://blacksmithgu.github.io/obsidian-dataview/annotation/add-metadata/) and [implicit page fields](https://blacksmithgu.github.io/obsidian-dataview/annotation/metadata-pages/)
- [Query structure](https://blacksmithgu.github.io/obsidian-dataview/queries/structure/) and [query types](https://blacksmithgu.github.io/obsidian-dataview/queries/query-types/)
- [Data commands](https://blacksmithgu.github.io/obsidian-dataview/queries/data-commands/) and [sources](https://blacksmithgu.github.io/obsidian-dataview/reference/sources/)
- [Expressions](https://blacksmithgu.github.io/obsidian-dataview/reference/expressions/) and [functions](https://blacksmithgu.github.io/obsidian-dataview/reference/functions/)
- [DQL, inline DQL, and JavaScript](https://blacksmithgu.github.io/obsidian-dataview/queries/dql-js-inline/)
