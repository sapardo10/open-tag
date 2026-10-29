// oMLX runtime: talks to the oMLX server's OpenAI-compatible /chat/completions endpoint
// (`omlx serve` / managed server on 127.0.0.1:<settings.json server.port>) instead of spawning
// any CLI — the models run locally, so agents do not depend on Anthropic/OpenAI accounts.
//
// There is no CLI session to resume: conversation history lives in memory only while the
// agent's open-tag session is running. Each start() begins a fresh chat (the standing prompt
// rides on the system role, so there is no envelope hack like reasonix); memory on disk
// (MEMORY.md) is re-read by the daemon as usual, and the wake nudge re-seeds a fresh chat.
import { initialTurnAdmission, protocolAdmission, type ProtocolAdmission, type Runtime, type StartOpts, type RuntimeCallbacks, type RuntimeSession, type TrajectoryEntry } from "./runtime.js";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { homedir } from "node:os";

const MAX = 2000;
const clip = (s: unknown) => String(s ?? "").slice(0, MAX);

// One chat/completions call can run long on 27B-class local models; bound the wait.
const REQUEST_TIMEOUT_MS = 120_000;

// Agents talk to open-tag through shell commands (`open-tag message check/send`), and a bare chat
// model has no hands — so every turn exposes one `shell` function and loops on tool calls.
const MAX_TOOL_STEPS = 16;
const SHELL_TIMEOUT_MS = 60_000;
const TOOL_OUT_MAX = 8000;
const SHELL_TOOL = {
  type: "function",
  function: {
    name: "shell",
    description: "Run one shell command in your working directory and return its combined output. Use it for every `open-tag` command and any file or project work.",
    parameters: { type: "object", properties: { command: { type: "string", description: "The command line to run." } }, required: ["command"] },
  },
} as const;

type ChatMessage = { role: "system" | "user" | "assistant" | "tool"; content: string; tool_calls?: unknown[]; tool_call_id?: string };
export interface OmlxToolCall { id: string; command: string }

/** Wire format of oMLX's OpenAI-compatible server (subset we use). */
export interface OmlxEndpoint {
  baseUrl: string;   // e.g. "http://127.0.0.1:8000/v1"
  apiKey?: string;
}

export interface OmlxChatReply {
  text?: string;
  reasoning?: string;
  toolCalls?: OmlxToolCall[];
  rawToolCalls?: unknown[];
  error?: string;
}

/**
 * Resolve the oMLX server endpoint. Priority: agent runtimeConfig (`baseUrl`/`apiKey`) →
 * `~/.omlx/settings.json` (`server.host`/`server.port`, `auth.api_key`) → env
 * (`OMLX_BASE_URL`/`OMLX_API_KEY`) → hard fallback `http://localhost:8000/v1` (the `omlx serve`
 * example port, so an out-of-the-box oMLX install works with no open-tag configuration).
 * Never throws; an unset/missing config simply means "use the fallback".
 */
export function resolveOmlxEndpoint(
  rc: Record<string, unknown> | null | undefined,
  env: NodeJS.ProcessEnv = process.env,
  settingsPath?: string,
): OmlxEndpoint {
  const raw =
    (typeof rc?.baseUrl === "string" && rc.baseUrl.trim()) ||
    (typeof env.OMLX_BASE_URL === "string" && env.OMLX_BASE_URL.trim()) ||
    omlxSettingsBaseUrl(settingsPath) ||
    "http://localhost:8000/v1";
  const key =
    (typeof rc?.apiKey === "string" && rc.apiKey.trim()) ||
    (typeof env.OMLX_API_KEY === "string" && env.OMLX_API_KEY.trim()) ||
    omlxSettingsApiKey(settingsPath);
  return { baseUrl: raw.trim().replace(/\/+$/, ""), ...(key ? { apiKey: key } : {}) };
}

/** Discover the model ids (plus the default marker) from `~/.omlx/model_settings.json`; empty on absence/corruption. */
export function discoverOmlxModelsFromSettings(
  settingsPath?: string,
): { id: string; label: string; provider: string; default?: boolean }[] {
  let settings: Record<string, any>;
  try {
    settings = JSON.parse(readFileSync(omlxConfigPath(settingsPath), "utf8"));
  } catch {
    return [];
  }
  const out: { id: string; label: string; provider: string; default?: boolean }[] = [];
  const seen = new Set<string>();
  const models = settings?.models;
  if (!models || typeof models !== "object") return [];
  for (const [id, cfg] of Object.entries<Record<string, unknown>>(models)) {
    if (cfg?.is_hidden === true) continue; // whitelist: hidden models must not leak into the picker
    if (!isOmlxModelId(id) || seen.has(id)) continue;
    seen.add(id);
    out.push({
      id,
      label: typeof cfg?.label === "string" && cfg.label ? cfg.label : id,
      provider: "omlx",
      ...(cfg?.is_default === true ? { default: true } : {}),
    });
  }
  return out;
}

type OmlxModel = { id: string; label: string; provider: string; default?: boolean };

/**
 * Order a catalog so the model a new agent should use comes first: `preferred`
 * (`OPEN_TAG_OMLX_MODEL`) wins, else the entry flagged `default`. A preferred id absent from the
 * catalog is still added — the server may serve it without listing it in model_settings.json.
 */
export function orderOmlxModels(models: OmlxModel[], preferred?: string): OmlxModel[] {
  const pref = preferred?.trim();
  const list = models.map((m) => ({ ...m, default: undefined as boolean | undefined }));
  let head = pref ? list.find((m) => m.id === pref) : undefined;
  if (pref && !head && isOmlxModelId(pref)) {
    head = { id: pref, label: pref, provider: "omlx", default: undefined };
    list.push(head);
  }
  head ??= models.find((m) => m.default) && list.find((m) => m.id === models.find((x) => x.default)!.id);
  if (head) head.default = true;
  return [...list.filter((m) => m === head), ...list.filter((m) => m !== head)]
    .map(({ default: d, ...m }) => (d ? { ...m, default: true } : m));
}

/** Ids the running oMLX server reports on `GET /models`; empty when it is down or slow. */
export async function fetchOmlxServedModels(endpoint: OmlxEndpoint, timeoutMs = 3000): Promise<string[]> {
  try {
    const res = await fetch(`${endpoint.baseUrl}/models`, {
      headers: endpoint.apiKey ? { authorization: `Bearer ${endpoint.apiKey}` } : {},
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return [];
    const ids = ((await res.json()) as any)?.data;
    return Array.isArray(ids) ? ids.map((m: any) => m?.id).filter((id: unknown): id is string => typeof id === "string" && isOmlxModelId(id)) : [];
  } catch {
    return [];
  }
}

/** Settings catalog ∪ live server list, preferred/default model first. */
export async function discoverOmlxModels(env: NodeJS.ProcessEnv = process.env): Promise<OmlxModel[]> {
  const fromSettings = discoverOmlxModelsFromSettings();
  const known = new Set(fromSettings.map((m) => m.id));
  const served = (await fetchOmlxServedModels(resolveOmlxEndpoint(null, env))).filter((id) => !known.has(id));
  const all = [...fromSettings, ...served.map((id) => ({ id, label: id, provider: "omlx" }))];
  return orderOmlxModels(all, env.OPEN_TAG_OMLX_MODEL);
}

/** A model id must be an identifier token (mirrors listModels' `isModelId`) — never prose or paths. */
function isOmlxModelId(s: string): boolean {
  return /^[A-Za-z][A-Za-z0-9\-_./]*$/.test(s);
}

/** Run one model-requested command; never rejects — failures come back as text for the model to react to. */
export function runShellTool(command: string, cwd: string, env: NodeJS.ProcessEnv, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve) => {
    execFile("/bin/sh", ["-c", command], { cwd, env, timeout: SHELL_TIMEOUT_MS, maxBuffer: 1024 * 1024, signal }, (err, stdout, stderr) => {
      const out = `${stdout}${stderr ? `\n${stderr}` : ""}`.trim();
      const code = err ? ((err as any).code ?? "error") : 0;
      resolve(clip8k(code === 0 ? (out || "(no output)") : `exit ${code}\n${out || err?.message || ""}`));
    });
  });
}
const clip8k = (s: string) => (s.length > TOOL_OUT_MAX ? `${s.slice(0, TOOL_OUT_MAX)}\n…[truncated]` : s);

/** `parseOmlxChatReply` maps one chat/completions JSON body to trajectory entries; pure + unit-tested. */
export function parseOmlxChatReply(json: unknown): OmlxChatReply {
  const choice = (json as any)?.choices?.[0];
  if (!choice) return { error: String((json as any)?.error?.message ?? (json as any)?.error ?? "omlx chat: no choices") };
  const msg = choice?.message;
  const text = typeof msg?.content === "string" ? msg.content : undefined;
  const reasoning = typeof msg?.reasoning_content === "string" ? msg.reasoning_content : undefined;
  const raw: any[] = Array.isArray(msg?.tool_calls) ? msg.tool_calls : [];
  const toolCalls: OmlxToolCall[] = [];
  for (const c of raw) {
    let args: any = c?.function?.arguments;
    if (typeof args === "string") { try { args = JSON.parse(args); } catch { args = {}; } }
    if (c?.function?.name === "shell" && typeof args?.command === "string" && args.command.trim()) {
      toolCalls.push({ id: String(c.id ?? `call_${toolCalls.length}`), command: args.command });
    }
  }
  if (toolCalls.length) return { ...(text ? { text } : {}), ...(reasoning ? { reasoning } : {}), toolCalls, rawToolCalls: raw };
  if (!text && !reasoning) return { error: choice?.finish_reason === "length" ? "omlx chat: truncated" : "omlx chat: empty reply" };
  return { ...(text ? { text } : {}), ...(reasoning ? { reasoning } : {}) };
}

function omlxSettingsBaseUrl(settingsPath?: string): string | undefined {
  try {
    const settings = JSON.parse(readFileSync(omlxConfigPath(settingsPath), "utf8"));
    const server = settings?.server;
    if (server?.host && server?.port) return `http://${server.host}:${server.port}/v1`;
  } catch { /* fall back below */ }
  return undefined;
}

function omlxSettingsApiKey(settingsPath?: string): string | undefined {
  try {
    const settings = JSON.parse(readFileSync(omlxConfigPath(settingsPath), "utf8"));
    const key = settings?.auth?.api_key;
    return typeof key === "string" && key ? key : undefined;
  } catch { return undefined; }
}

function omlxConfigPath(settingsPath?: string): string {
  return settingsPath ?? path.join(homedir(), ".omlx", "settings.json");
}

interface OmlxInput {
  text: string;
  initial: boolean;
  admission: ProtocolAdmission;
}

/**
 * OmlxChat owns one agent's in-memory chat: a serial queue (mirrors opencode/cursor's one-shot
 * pattern, but HTTP instead of a spawned CLI) of chat/completions calls.
 */
class OmlxChat {
  private queue: OmlxInput[] = [];
  private turnBusy = false;
  private stopped = false;
  private activeInput: OmlxInput | null = null;
  private inFlight: AbortController | null = null;
  private everSucceeded = false;
  private exitReported = false;
  private readonly history: ChatMessage[];

  constructor(
    private readonly endpoint: OmlxEndpoint,
    private readonly model: string,
    private readonly cb: RuntimeCallbacks,
    systemPrompt: string,
    private readonly cwd: string,
    private readonly env: NodeJS.ProcessEnv,
  ) {
    this.history = [{ role: "system", content: systemPrompt }];
    this.admission = initialTurnAdmission(cb);
  }

  private readonly admission: ReturnType<typeof initialTurnAdmission>;

  enqueue(text: string, initial = false): Promise<void> {
    const input: OmlxInput = { text, initial, admission: protocolAdmission() };
    if (this.stopped) input.admission.reject(new Error("omlx stopped before input admission"));
    else { this.queue.push(input); this.pump(); }
    return input.admission.promise;
  }

  private pump(): void {
    if (this.stopped || this.turnBusy || this.queue.length === 0) return;
    void this.runTurn(this.queue.shift()!);
  }

  private rejectQueue(error: Error): void {
    for (const input of this.queue.splice(0)) input.admission.reject(error);
  }

  private reportExit(code: number | null): void {
    if (this.exitReported) return;
    this.exitReported = true;
    this.cb.onExit(code);
  }

  private async runTurn(input: OmlxInput): Promise<void> {
    this.activeInput = input;
    this.turnBusy = true;
    this.cb.onActivity("working", "turn");
    // Keep a bounded window (cut only at a user message so tool-call/result pairs stay intact).
    let start = 1;
    if (this.history.length > 41) {
      start = this.history.length - 40;
      while (start < this.history.length && this.history[start]!.role !== "user") start++;
    }
    const turn: ChatMessage[] = [{ role: "user", content: input.text }];
    this.inFlight = new AbortController();
    try {
      for (let step = 0; ; step++) {
        const res = await fetch(`${this.endpoint.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...(this.endpoint.apiKey ? { authorization: `Bearer ${this.endpoint.apiKey}` } : {}),
          },
          body: JSON.stringify({
            model: this.model,
            messages: [this.history[0], ...this.history.slice(start), ...turn],
            tools: [SHELL_TOOL],
            max_tokens: 4096,
            non_stream: true,
          }),
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        });
        const json = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(String((json as any)?.error?.message ?? `omlx chat ${res.status}`));
        const reply = parseOmlxChatReply(json);
        if (reply.error) {
          this.cb.onTrajectory([{ kind: "text", text: `[omlx error] ${clip(reply.error)}` }]);
          this.cb.onActivity("error", clip(reply.error));
          if (!this.everSucceeded && step === 0) {
            input.admission.reject(new Error(reply.error));
            if (input.initial) this.admission.reject(new Error(reply.error));
            this.rejectQueue(new Error(clip(reply.error)));
            this.reportExit(1);
            return;
          }
          input.admission.accept();
          if (input.initial) this.admission.accept();
          this.history.push(...turn);
          this.cb.onAcceptedTurnFailure?.(input);
          this.finishTurn();
          return;
        }
        if (input.initial) this.admission.accept();
        input.admission.accept();
        if (reply.reasoning) this.cb.onTrajectory([{ kind: "thinking", text: clip(reply.reasoning) }]);
        if (reply.text) this.cb.onTrajectory([{ kind: "text", text: clip(reply.text) }]);
        turn.push({ role: "assistant", content: reply.text ?? "", ...(reply.rawToolCalls ? { tool_calls: reply.rawToolCalls } : {}) });
        if (!reply.toolCalls) break; // final answer
        if (step + 1 >= MAX_TOOL_STEPS) {
          this.cb.log.warn("omlx tool loop capped", { steps: MAX_TOOL_STEPS });
          turn.pop();
          break;
        }
        this.cb.onActivity("working", "tool");
        for (const call of reply.toolCalls) {
          this.cb.onTrajectory([{ kind: "tool", toolName: "shell", toolInput: clip(call.command) }]);
          const out = await runShellTool(call.command, this.cwd, this.env, this.inFlight.signal);
          turn.push({ role: "tool", tool_call_id: call.id, content: out });
        }
      }
      this.history.push(...turn);
      this.everSucceeded = true;
      this.cb.onActivity("online", "");
      this.finishTurn();
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));
      this.cb.log.warn("omlx turn failed", { detail: error.message });
      if (input.initial) this.admission.reject(error);
      input.admission.reject(error);
      if (!this.everSucceeded) {
        this.rejectQueue(error);
        this.reportExit(1);
        return;
      }
      this.cb.onActivity("offline", clip(error.message));
      this.finishTurn(); // later-turn failure: stay alive so the next message can retry
    }
  }

  private finishTurn(): void {
    if (this.activeInput) this.activeInput = null;
    this.turnBusy = false;
    this.inFlight = null;
    this.pump();
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    const error = new Error("omlx stopped before input admission");
    this.activeInput?.admission.reject(error);
    this.activeInput = null;
    this.rejectQueue(error);
    this.inFlight?.abort();
    this.inFlight = null;
    this.reportExit(0);
  }
}

export const omlxRuntime: Runtime = {
  name: "omlx",
  experimental: true,
  start(opts: StartOpts, cb: RuntimeCallbacks): RuntimeSession {
    const endpoint = resolveOmlxEndpoint(opts.runtimeConfig, opts.env);
    const model = opts.model && opts.model !== "default"
      ? opts.model
      : orderOmlxModels(discoverOmlxModelsFromSettings(), opts.env?.OPEN_TAG_OMLX_MODEL ?? process.env.OPEN_TAG_OMLX_MODEL)[0]?.id;
    const chat = new OmlxChat(endpoint, model ?? "", cb, opts.systemPrompt, opts.cwd, opts.env);
    if (!model) {
      // Fail loud (mirrors codex's "thread/start returned no threadId"): an agent without a
      // model on this machine must not silently chat with a nonsense model id — stop before
      // the initial prompt is enqueued so start fails with "omlx stopped before admission".
      cb.log.error("omlx no model", { detail: "no models configured on this machine" });
      chat.stop();
    }
    chat.enqueue(opts.initialPrompt, true).catch(() => {});
    return { deliver: (text) => chat.enqueue(text), stop: () => chat.stop() };
  },
};
