export type User = { id: string; email: string; role: string; created_at: string };

export type SessionFile = {
  id: string;
  name: string;
  kind: "table" | "document";
  source: "csv" | "xlsx" | "pdf" | "image";
  /** Rows for a table, pages for a document. */
  size: number;
  /** Pages (or the image) read with OCR. */
  ocr_pages: number;
  /** Pages OCR could not read reliably, read by the vision model instead. */
  vision_pages: number;
  expires_at: string;
};

export type ChatSession = { id: string; title: string; created_at: string; files: SessionFile[] };

export type ExportFormat = "pdf" | "docx" | "xlsx";

export type ChartSpec = { kind: "bar" | "line"; x: string; y: string[]; dashed?: string[] };

export type TableEvent = {
  type: "table";
  columns: string[];
  types: string[];
  rows: unknown[][];
  truncated: boolean;
};

export type AgentEvent =
  | { type: "intent"; value: "sql" | "python" | "sentiment" | "forecast" | "profile" | "doc_qa" | "doc_summary" | "doc_analysis" | "compose" | "chat" }
  | { type: "code"; language: "sql" | "python"; value: string; attempt: number }
  | { type: "retry"; error: string }
  | TableEvent
  | { type: "value"; value: unknown }
  | { type: "stdout"; value: string }
  | { type: "chart"; spec: ChartSpec }
  | { type: "samples"; items: { label: string; text: string }[] }
  | { type: "sources"; items: { file?: string; page: number; text: string }[] }
  | { type: "document"; title: string; markdown: string; words: number }
  | { type: "files"; formats: ExportFormat[] }
  | { type: "answer"; text: string }
  | { type: "error"; message: string }
  | { type: "done"; latency_ms: number; message_id?: string };

export type StoredMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  steps_json: AgentEvent[];
  latency_ms: number | null;
  created_at: string;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const BASE = "/api/v1";
const NO_REFRESH = ["/auth/login", "/auth/register", "/auth/refresh"];

async function errorFrom(res: Response): Promise<ApiError> {
  let message = `Terjadi kesalahan (${res.status})`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail) && body.detail[0]?.msg) message = body.detail[0].msg;
  } catch {
    // body was not JSON; keep the generic message
  }
  return new ApiError(res.status, message);
}

async function send(path: string, init: RequestInit): Promise<Response> {
  const headers = init.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init.headers };
  const res = await fetch(`${BASE}${path}`, { credentials: "include", ...init, headers });
  if (res.status !== 401 || NO_REFRESH.includes(path)) return res;

  // Access tokens live 15 minutes; refresh once and replay the request.
  const refreshed = await fetch(`${BASE}/auth/refresh`, { method: "POST", credentials: "include" });
  return refreshed.ok ? fetch(`${BASE}${path}`, { credentials: "include", ...init, headers }) : res;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await send(path, init);
  if (!res.ok) throw await errorFrom(res);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  me: () => request<User>("/auth/me"),
  login: (email: string, password: string) =>
    request<User>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  register: (email: string, password: string) =>
    request<User>("/auth/register", { method: "POST", body: JSON.stringify({ email, password }) }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),

  sessions: () => request<ChatSession[]>("/chat/sessions"),
  createSession: () => request<ChatSession>("/chat/sessions", { method: "POST" }),
  session: (id: string) => request<ChatSession & { messages: StoredMessage[] }>(`/chat/sessions/${id}`),
  deleteSession: (id: string) => request<void>(`/chat/sessions/${id}`, { method: "DELETE" }),
  attachFile: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<SessionFile>(`/chat/sessions/${sessionId}/files`, { method: "POST", body: form });
  },
  removeFile: (fileId: string) => request<void>(`/datasets/${fileId}`, { method: "DELETE" }),
};

/** POST a question and call `onEvent` for each server-sent agent event as it arrives. */
export async function streamMessage(sessionId: string, content: string, onEvent: (event: AgentEvent) => void) {
  const res = await send(`/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
  if (!res.ok || !res.body) throw await errorFrom(res);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as AgentEvent);
      }
    }
  }
}

/** Fetch one stored answer as a file and hand it to the browser as a download. */
export async function downloadExport(messageId: string, format: ExportFormat) {
  const res = await send(`/chat/messages/${messageId}/export?format=${format}`, { method: "GET" });
  if (!res.ok) throw await errorFrom(res);
  const name = /filename="([^"]+)"/.exec(res.headers.get("Content-Disposition") ?? "")?.[1] ?? `hasil.${format}`;
  const url = URL.createObjectURL(await res.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}
