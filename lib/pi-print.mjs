import { spawn } from "node:child_process";

const args = ["--print", "--mode", "json", ...process.argv.slice(2)];
const child = spawn("pi", args, { stdio: ["ignore", "pipe", "pipe"] });
let buffer = "";
let finalText = "";
let protocolErrors = "";

function consume(line) {
  if (!line.trim()) return;
  let event;
  try {
    event = JSON.parse(line);
  } catch {
    protocolErrors += `${line}\n`;
    return;
  }
  if (event.type === "message_end" && event.message?.role === "assistant") {
    const text = (event.message.content ?? [])
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("\n");
    if (text.trim()) finalText = text;
  }
}

child.stdout.setEncoding("utf8");
child.stderr.setEncoding("utf8");
child.stdout.on("data", (chunk) => {
  buffer += chunk;
  const lines = buffer.split("\n");
  buffer = lines.pop() ?? "";
  for (const line of lines) consume(line);
});
child.stderr.on("data", (chunk) => { process.stderr.write(chunk); });
child.on("error", (error) => {
  console.error(`mle pi: ${error.message}`);
  process.exitCode = 1;
});
child.on("close", (code, signal) => {
  if (buffer) consume(buffer);
  if (code !== 0) {
    if (protocolErrors) process.stderr.write(protocolErrors);
    process.exitCode = typeof code === "number" ? code : 1;
    return;
  }
  if (signal) {
    console.error(`mle pi: terminated by ${signal}`);
    process.exitCode = 1;
    return;
  }
  if (!finalText.trim()) {
    if (protocolErrors) process.stderr.write(protocolErrors);
    console.error("mle pi: no assistant response received");
    process.exitCode = 1;
    return;
  }
  process.stdout.write(`${finalText.trim()}\n`);
});
