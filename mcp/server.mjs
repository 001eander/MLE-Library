import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { searchKnowledge } from "../lib/search.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TEXT_EXTENSIONS = new Set([".md", ".markdown", ".txt"]);
const IMAGE_TYPES = new Map([
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".gif", "image/gif"],
  [".webp", "image/webp"],
]);

const toolDefinitions = [
  {
    name: "search_knowledge",
    description: "Search current wiki and raw Markdown/text with lexical matching. Results include paths, line numbers, snippets, and matched terms.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", minLength: 1 },
        scope: { type: "string", enum: ["all", "wiki", "raw"], default: "all" },
        limit: { type: "integer", minimum: 1, maximum: 50, default: 8 },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    name: "read_document",
    description: "Read a line range from an allowed Markdown/text file, the extracted text for a raw PDF, or an existing image under raw/assets or .cache/pdf.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", minLength: 1 },
        start_line: { type: "integer", minimum: 1, default: 1 },
        line_count: { type: "integer", minimum: 1, maximum: 1000, default: 200 },
        max_chars: { type: "integer", minimum: 1, maximum: 100000, default: 30000 },
      },
      required: ["path"],
      additionalProperties: false,
    },
  },
  {
    name: "get_context",
    description: "Retrieve a bounded evidence pack by searching and then reading source lines around each match.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", minLength: 1 },
        scope: { type: "string", enum: ["all", "wiki", "raw"], default: "all" },
        limit: { type: "integer", minimum: 1, maximum: 20, default: 6 },
        context_lines: { type: "integer", minimum: 1, maximum: 40, default: 12 },
        max_chars: { type: "integer", minimum: 1000, maximum: 100000, default: 20000 },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    name: "ask_librarian",
    description: "Ask a one-shot Pi librarian to synthesize an answer with citations. Pi has read/search/list tools only and cannot write files, render pages, or commit.",
    inputSchema: {
      type: "object",
      properties: {
        question: { type: "string", minLength: 1 },
        timeout_seconds: { type: "integer", minimum: 1, maximum: 300, default: 120 },
      },
      required: ["question"],
      additionalProperties: false,
    },
  },
];

function jsonResult(value) {
  return {
    content: [{ type: "text", text: JSON.stringify(value, null, 2) }],
    structuredContent: value,
  };
}

function errorResult(error) {
  return {
    isError: true,
    content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
  };
}

function integer(value, fallback, name, minimum, maximum) {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved < minimum || resolved > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`);
  }
  return resolved;
}

function resolveInside(requestedPath) {
  if (typeof requestedPath !== "string" || !requestedPath || requestedPath.includes("\0")) {
    throw new Error("path must be a non-empty library-relative string");
  }
  if (path.isAbsolute(requestedPath)) throw new Error("absolute paths are not allowed");
  const candidate = path.resolve(ROOT, requestedPath);
  const relative = path.relative(ROOT, candidate);
  if (!relative || relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error("path is outside the library");
  }
  const real = fs.realpathSync(candidate);
  const realRelative = path.relative(ROOT, real);
  if (realRelative === ".." || realRelative.startsWith(`..${path.sep}`) || path.isAbsolute(realRelative)) {
    throw new Error("path resolves outside the library");
  }
  if (!fs.statSync(real).isFile()) throw new Error("path is not a file");
  return { absolute: real, relative: realRelative.split(path.sep).join("/") };
}

function assertTextPath(relative) {
  if (!(relative.startsWith("raw/") || relative.startsWith("wiki/"))) {
    throw new Error("text reads are limited to raw/ and wiki/");
  }
  if (!TEXT_EXTENSIONS.has(path.extname(relative).toLowerCase())) {
    throw new Error("only Markdown and plain-text files can be read as text");
  }
}

function readTextSnippet({ absolute, relative }, startLine, lineCount, maxChars) {
  assertTextPath(relative);
  if (fs.statSync(absolute).size > 50 * 1024 * 1024) throw new Error("text file exceeds the 50 MB read limit");
  const lines = fs.readFileSync(absolute, "utf8").split("\n");
  const selected = lines.slice(startLine - 1, startLine - 1 + lineCount);
  let text = selected.join("\n");
  const truncatedByChars = text.length > maxChars;
  if (truncatedByChars) text = text.slice(0, maxChars);
  return {
    path: relative,
    start_line: startLine,
    end_line: selected.length > 0 ? startLine + selected.length - 1 : startLine,
    total_lines: lines.length,
    truncated_by_chars: truncatedByChars,
    text,
  };
}

function readDocument(args) {
  const startLine = integer(args.start_line, 1, "start_line", 1, 1_000_000_000);
  const lineCount = integer(args.line_count, 200, "line_count", 1, 1000);
  const maxChars = integer(args.max_chars, 30000, "max_chars", 1, 100000);
  let document = resolveInside(args.path);
  let originalPdf;
  if (document.relative.startsWith("raw/") && document.relative.toLowerCase().endsWith(".pdf")) {
    originalPdf = document.relative;
    const extracted = `raw/extracted/${path.basename(document.relative, path.extname(document.relative))}.md`;
    try {
      document = resolveInside(extracted);
    } catch {
      throw new Error(`no extracted Markdown found for ${originalPdf}; run mle ingest under human supervision first`);
    }
  }
  const mimeType = IMAGE_TYPES.get(path.extname(document.relative).toLowerCase());
  if (mimeType) {
    if (!(document.relative.startsWith("raw/assets/") || document.relative.startsWith(".cache/pdf/"))) {
      throw new Error("image reads are limited to raw/assets/ and .cache/pdf/");
    }
    if (fs.statSync(document.absolute).size > 20 * 1024 * 1024) throw new Error("image exceeds the 20 MB read limit");
    return {
      content: [{
        type: "image",
        data: fs.readFileSync(document.absolute).toString("base64"),
        mimeType,
      }],
    };
  }
  const result = readTextSnippet(document, startLine, lineCount, maxChars);
  if (originalPdf) result.original_pdf = originalPdf;
  return jsonResult(result);
}

async function getContext(args) {
  const limit = integer(args.limit, 6, "limit", 1, 20);
  const contextLines = integer(args.context_lines, 12, "context_lines", 1, 40);
  const maxChars = integer(args.max_chars, 20000, "max_chars", 1000, 100000);
  const results = await searchKnowledge({ query: args.query, scope: args.scope ?? "all", limit });
  const evidence = [];
  let used = 0;
  let truncated = false;
  for (const result of results) {
    if (used >= maxChars) {
      truncated = true;
      break;
    }
    const firstMatch = result.matches[0]?.line ?? 1;
    const startLine = Math.max(1, firstMatch - 3);
    const snippet = readTextSnippet(resolveInside(result.path), startLine, contextLines, maxChars - used);
    used += snippet.text.length;
    truncated ||= snippet.truncated_by_chars;
    evidence.push({
      path: snippet.path,
      layer: result.layer,
      citation: `${snippet.path}#L${snippet.start_line}-L${snippet.end_line}`,
      matched_terms: result.matched_terms,
      score: result.score,
      start_line: snippet.start_line,
      end_line: snippet.end_line,
      text: snippet.text,
    });
  }
  if (evidence.length < results.length) truncated = true;
  return jsonResult({
    query: args.query,
    retrieval: "ripgrep lexical search with source read-back",
    evidence,
    truncated,
    chars: used,
    max_chars: maxChars,
    warnings: evidence.length === 0 ? ["No matching documents found."] : [],
  });
}

function askLibrarian(args) {
  const timeoutSeconds = integer(args.timeout_seconds, 120, "timeout_seconds", 1, 300);
  const prompt = `Answer the following question using MLE-Library. Read wiki/index.md, search and read full relevant pages, consult raw sources when needed, and cite actual paths with line, section, or page locators. State uncertainty and conflicting evidence. You are strictly read-only: do not modify files, render pages, invoke other MCP servers, or change Git state.\n\nQuestion: ${args.question}`;
  const piArgs = [
    "--print",
    "--no-session",
    "--no-context-files",
    "--append-system-prompt", path.join(ROOT, "AGENTS.md"),
    "--no-skills",
    "--no-extensions",
    "--no-mcp",
    "--tools", "read,grep,find,ls",
    "--",
    prompt,
  ];
  return new Promise((resolve, reject) => {
    const child = spawn("pi", piArgs, { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let outputExceeded = false;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutSeconds * 1000);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      if (stdout.length > 1_000_000) {
        outputExceeded = true;
        child.kill("SIGTERM");
      }
    });
    child.stderr.on("data", (chunk) => {
      if (stderr.length < 100_000) stderr += chunk;
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      if (timedOut) reject(new Error(`Pi librarian timed out after ${timeoutSeconds} seconds`));
      else if (outputExceeded) reject(new Error("Pi librarian exceeded the 1 MB output limit"));
      else if (code !== 0) reject(new Error(`Pi librarian failed (${signal ?? code}): ${stderr.trim()}`));
      else resolve(jsonResult({ answer: stdout.trim(), read_only: true }));
    });
  });
}

const server = new Server(
  { name: "mle-library", version: "0.1.0" },
  {
    capabilities: { tools: {} },
    instructions: "Read-only access to a human-supervised Machine Learning Engineering wiki. Search first, then read full evidence. AutoResearch must not modify this library through other means.",
  },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: toolDefinitions }));
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    const args = request.params.arguments ?? {};
    switch (request.params.name) {
      case "search_knowledge":
        return jsonResult({ results: await searchKnowledge({
          query: args.query,
          scope: args.scope ?? "all",
          limit: integer(args.limit, 8, "limit", 1, 50),
        }) });
      case "read_document":
        return readDocument(args);
      case "get_context":
        return await getContext(args);
      case "ask_librarian":
        return await askLibrarian(args);
      default:
        throw new Error(`unknown tool: ${request.params.name}`);
    }
  } catch (error) {
    return errorResult(error);
  }
});

await server.connect(new StdioServerTransport());
