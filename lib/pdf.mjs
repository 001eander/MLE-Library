import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function relativePath(absolutePath) {
  return path.relative(ROOT, absolutePath).split(path.sep).join("/");
}

function assertRawPdf(source) {
  const resolved = path.resolve(source);
  const relative = relativePath(resolved);
  if (!relative.startsWith("raw/") || relative.includes("../") || !relative.toLowerCase().endsWith(".pdf")) {
    throw new Error("PDF source must be inside raw/ and have a .pdf extension");
  }
  if (!fs.statSync(resolved).isFile()) throw new Error("PDF source is not a file");
  return resolved;
}

export function sha256(file) {
  return createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

export function extractPdf(sourceInput) {
  const source = assertRawPdf(sourceInput);
  const hash = sha256(source);
  const destination = path.join(ROOT, "raw", "extracted", `${path.basename(source, path.extname(source))}.md`);
  const sourceRelative = relativePath(source);
  if (fs.existsSync(destination)) {
    const existing = fs.readFileSync(destination, "utf8");
    const hashMatch = existing.match(/^source_sha256:\s*"?([a-f0-9]{64})"?\s*$/m);
    const pathMatch = existing.match(/^source_path:\s*(.+)$/m);
    if (hashMatch?.[1] === hash && pathMatch?.[1] === JSON.stringify(sourceRelative)) {
      return relativePath(destination);
    }
    throw new Error(`extraction already exists for a different source: ${relativePath(destination)}`);
  }
  const result = spawnSync("pdftotext", ["-layout", source, "-"], {
    encoding: "utf8",
    maxBuffer: 512 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`pdftotext failed: ${result.stderr.trim()}`);
  const pages = result.stdout.split("\f");
  if (pages.length > 1 && pages.at(-1).trim() === "") pages.pop();
  const body = pages.map((page, index) => `## Page ${index + 1}\n\n${page.trim()}\n`).join("\n");
  const frontmatter = [
    "---",
    `source_path: ${JSON.stringify(sourceRelative)}`,
    `source_sha256: ${JSON.stringify(hash)}`,
    'extractor: "pdftotext -layout"',
    "---",
    "",
    "# Extracted PDF Text",
    "",
  ].join("\n");
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, `${frontmatter}${body}`, { encoding: "utf8", flag: "wx" });
  if (result.stdout.trim() === "") {
    console.error("mle pdf: no text extracted; use visual page inspection when the model supports images");
  }
  return relativePath(destination);
}

export function renderPdfPage(sourceInput, pageInput) {
  const source = assertRawPdf(sourceInput);
  const page = Number(pageInput);
  if (!Number.isInteger(page) || page < 1) throw new Error("page must be a positive integer");
  const hash = sha256(source);
  const directory = path.join(ROOT, ".cache", "pdf", hash);
  const destination = path.join(directory, `page-${page}.png`);
  if (!fs.existsSync(destination)) {
    fs.mkdirSync(directory, { recursive: true });
    const prefix = path.join(directory, `page-${page}`);
    const result = spawnSync("pdftoppm", ["-f", String(page), "-l", String(page), "-singlefile", "-png", "-r", "180", source, prefix], {
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024,
    });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`pdftoppm failed: ${result.stderr.trim()}`);
  }
  if (!fs.existsSync(destination)) throw new Error(`render did not produce page ${page}`);
  return relativePath(destination);
}

function main() {
  const [command, source, argument] = process.argv.slice(2);
  if (command === "extract" && source && argument === undefined) {
    console.log(extractPdf(source));
  } else if (command === "render" && source && argument !== undefined) {
    console.log(renderPdfPage(source, argument));
  } else {
    throw new Error("usage: node lib/pdf.mjs extract <raw-pdf> | render <raw-pdf> <page>");
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error) {
    console.error(`mle pdf: ${error.message}`);
    process.exitCode = 1;
  }
}
