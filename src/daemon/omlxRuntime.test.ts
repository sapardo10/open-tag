// Pure-mapping tests for the oMLX runtime: endpoint resolution, model-list discovery from
// model_settings.json, and chat-completions reply mapping. Run:
// `npx tsx --test src/daemon/omlxRuntime.test.ts`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { discoverOmlxModelsFromSettings, orderOmlxModels, parseOmlxChatReply, resolveOmlxEndpoint, runShellTool } from "./omlxRuntime.js";

function withSettings(body: unknown, fn: (p: string) => void): void {
  const dir = mkdtempSync(path.join(os.tmpdir(), "omlx-test-"));
  const file = path.join(dir, "model_settings.json");
  writeFileSync(file, JSON.stringify(body));
  try { fn(file); } finally { rmSync(dir, { recursive: true, force: true }); }
}

test("endpoint: runtimeConfig wins, settings.json next, env last, hardcoded fallback", () => {
  withSettings({ server: { host: "127.0.0.1", port: 8000 }, auth: { api_key: "omlx-k", secret_key: "s" } }, (file) => {
    assert.deepEqual(resolveOmlxEndpoint({ baseUrl: "http://proxy:9/v1", apiKey: "rc-key" }, {}, file), { baseUrl: "http://proxy:9/v1", apiKey: "rc-key" });
    assert.deepEqual(resolveOmlxEndpoint(null, { OMLX_BASE_URL: "http://env:8000/v1/" }, file), { baseUrl: "http://env:8000/v1", apiKey: "omlx-k" }, "settings.json supplies the key when runtimeConfig/env omit it");
    assert.deepEqual(resolveOmlxEndpoint(null, {}, file), { baseUrl: "http://127.0.0.1:8000/v1", apiKey: "omlx-k" }, "settings.json server/port and api key are honored");
  });
  assert.deepEqual(resolveOmlxEndpoint(null, {}, path.join(os.tmpdir(), "definitely-missing.json")), { baseUrl: "http://localhost:8000/v1" }, "missing config falls back to the omlx serve default port");
});

test("model list: only visible models surface, default marker honored, junk ids rejected", () => {
  withSettings({
    models: {
      "gemma-4-E2B": { is_default: true, is_hidden: false },
      "hidden-Model": { is_default: false, is_hidden: true },
      "Qwen3.8-27B-MLX-4bit": { is_default: false, is_hidden: false },
      "Not A Model": { is_default: false, is_hidden: false }, // whitespace fails the model-id token rule
    },
  }, (file) => {
    const found = discoverOmlxModelsFromSettings(file);
    assert.deepEqual(found.map((m) => m.id), ["gemma-4-E2B", "Qwen3.8-27B-MLX-4bit"]);
    assert.equal(found.find((m) => m.id === "gemma-4-E2B")?.default, true);
    assert.equal(found.find((m) => m.id === "hidden-Model"), undefined, "hidden models must never leak into the picker");
    assert.ok(found.every((m) => m.provider === "omlx"));
  });
  assert.deepEqual(discoverOmlxModelsFromSettings(path.join(os.tmpdir(), "nope.json")), [], "missing settings file → empty list (static fallback on the server)");
  withSettings("{not json" as never, (file) => {
    assert.deepEqual(discoverOmlxModelsFromSettings(file), [], "corrupt JSON degrades to the empty list");
  });
});

test("chat reply: content + reasoning mapped; failures surfaced as errors", () => {
  assert.deepEqual(parseOmlxChatReply({ choices: [{ message: { content: "pong", reasoning_content: "echo" }, finish_reason: "stop" }] }), { text: "pong", reasoning: "echo" });
  assert.deepEqual(parseOmlxChatReply({}), { error: "omlx chat: no choices" });
  assert.deepEqual(parseOmlxChatReply({ error: { message: "model not loaded" } }), { error: "model not loaded" });
  assert.deepEqual(parseOmlxChatReply({ choices: [{ finish_reason: "length" }] }), { error: "omlx chat: truncated" });
  assert.deepEqual(parseOmlxChatReply({ choices: [{ finish_reason: "stop", message: {} }] }), { error: "omlx chat: empty reply" });
});

test("ordering: OPEN_TAG_OMLX_MODEL wins, is added if unlisted, else is_default leads", () => {
  const m = (id: string, d?: boolean) => ({ id, label: id, provider: "omlx", ...(d ? { default: true } : {}) });
  const cat = [m("a", true), m("b")];
  assert.deepEqual(orderOmlxModels(cat).map((x) => x.id), ["a", "b"]);
  assert.deepEqual(orderOmlxModels(cat, "b").map((x) => [x.id, x.default]), [["b", true], ["a", undefined]]);
  assert.deepEqual(orderOmlxModels(cat, "Qwen-x").map((x) => x.id), ["Qwen-x", "a", "b"]);
  assert.deepEqual(orderOmlxModels(cat, "bad id").map((x) => x.id), ["a", "b"]);
});

test("tool calls: shell calls parse (string or object args), others ignored; runShellTool returns output/exit", async () => {
  const reply = parseOmlxChatReply({ choices: [{ message: { content: "", tool_calls: [
    { id: "c1", function: { name: "shell", arguments: "{\"command\":\"echo hi\"}" } },
    { id: "c2", function: { name: "other", arguments: "{}" } },
  ] } }] });
  assert.deepEqual(reply.toolCalls, [{ id: "c1", command: "echo hi" }]);
  assert.equal(await runShellTool("echo hi", os.tmpdir(), process.env), "hi");
  assert.match(await runShellTool("echo bad >&2; exit 3", os.tmpdir(), process.env), /^exit 3\nbad/);
});
