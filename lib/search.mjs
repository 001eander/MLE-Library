import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function termsFor(query) {
  const terms = [];
  for (const match of query.matchAll(/"([^"]+)"|'([^']+)'|(\S+)/gu)) {
    const term = (match[1] ?? match[2] ?? match[3]).trim();
    if (term && !terms.some((item) => item.toLowerCase() === term.toLowerCase())) {
      terms.push(term);
    }
  }
  if (terms.length === 0) throw new Error("search query must not be empty");
  return terms;
}

function runRipgrep(query, scope) {
  const terms = termsFor(query);
  const pattern = terms.map(escapeRegExp).join("|");
  const roots = scope === "all" ? ["wiki", "raw"] : [scope];
  const args = [
    "--json",
    "--ignore-case",
    "--line-number",
    "--max-count", "50",
    "--max-columns", "800",
    "--max-columns-preview",
    "--glob", "*.md",
    "--glob", "*.markdown",
    "--glob", "*.txt",
    "--regexp", pattern,
    "--",
    ...roots,
  ];
  return new Promise((resolve, reject) => {
    const child = spawn("rg", args, { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      if (stdout.length > 25_000_000) child.kill("SIGTERM");
    });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (code === 0 || code === 1) resolve({ terms, stdout });
      else reject(new Error(`ripgrep failed (${signal ?? code}): ${stderr.trim() || "output limit exceeded"}`));
    });
  });
}

function layerFor(relativePath) {
  if (relativePath.startsWith("raw/")) return "raw";
  if (relativePath.startsWith("wiki/research/")) return "research";
  return "wiki";
}

export async function searchKnowledge({ query, scope = "all", limit = 8 }) {
  if (!["all", "wiki", "raw"].includes(scope)) throw new Error("scope must be all, wiki, or raw");
  const numericLimit = Number(limit);
  if (!Number.isInteger(numericLimit) || numericLimit < 1 || numericLimit > 50) {
    throw new Error("limit must be an integer between 1 and 50");
  }
  const { terms, stdout } = await runRipgrep(query, scope);
  const documents = new Map();
  for (const line of stdout.split("\n")) {
    if (!line) continue;
    const event = JSON.parse(line);
    if (event.type !== "match") continue;
    const absolutePath = event.data.path.text;
    const relativePath = path.relative(ROOT, path.resolve(ROOT, absolutePath)).split(path.sep).join("/");
    const text = event.data.lines.text.trimEnd();
    const lowerText = text.toLowerCase();
    let document = documents.get(relativePath);
    if (!document) {
      document = {
        path: relativePath,
        layer: layerFor(relativePath),
        score: 0,
        matched_terms: new Set(),
        match_count: 0,
        matches: [],
      };
      documents.set(relativePath, document);
    }
    for (const term of terms) {
      if (lowerText.includes(term.toLowerCase())) document.matched_terms.add(term);
    }
    document.match_count += 1;
    if (document.matches.length < 4) {
      document.matches.push({ line: event.data.line_number, text });
    }
  }
  return [...documents.values()]
    .map((document) => {
      const basename = path.basename(document.path).toLowerCase();
      const titleMatches = terms.filter((term) => basename.includes(term.toLowerCase())).length;
      document.score = document.matched_terms.size * 100 + titleMatches * 20 + Math.min(document.match_count, 20) * 5;
      return {
        path: document.path,
        layer: document.layer,
        score: document.score,
        matched_terms: [...document.matched_terms],
        matches: document.matches,
      };
    })
    .sort((a, b) => b.score - a.score || a.path.localeCompare(b.path))
    .slice(0, numericLimit);
}

async function main() {
  const args = process.argv.slice(2);
  let scope = "all";
  let limit = 8;
  let json = false;
  const queryParts = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--scope") {
      scope = args[++index];
    } else if (arg === "--limit") {
      limit = Number(args[++index]);
    } else if (arg === "--json") {
      json = true;
    } else if (arg === "--") {
      queryParts.push(...args.slice(index + 1));
      break;
    } else {
      queryParts.push(arg);
    }
  }
  const results = await searchKnowledge({ query: queryParts.join(" "), scope, limit });
  if (json) {
    process.stdout.write(`${JSON.stringify({ results }, null, 2)}\n`);
    return;
  }
  if (results.length === 0) {
    process.stdout.write("No matches.\n");
    return;
  }
  for (const result of results) {
    process.stdout.write(`${result.path} [${result.layer}] score=${result.score}\n`);
    for (const match of result.matches) {
      process.stdout.write(`  ${match.line}: ${match.text}\n`);
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(`mle search: ${error.message}`);
    process.exitCode = 1;
  });
}
