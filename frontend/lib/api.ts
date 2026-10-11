export type User = { id: string; email: string; role: string; created_at: string };

export type Dataset = {
  id: string;
  original_filename: string;
  row_count: number;
  created_at: string;
  expires_at: string;
};

export type ChatSession = { id: string; dataset_id: string; title: string; created_at: string };

export type ChartSpec = { kind: "bar" | "line"; x: string; y: string[]; dashed?: string[] };

export type TableEvent = {
  type: "table";
  columns: string[];
  types: string[];
  rows: unknown[][];
  truncated: boolean;
};

export type AgentEvent =
  | { type: "intent"; value: "sql" | "python" | "sentiment" | "forecast" | "chat" }
  | { type: "code"; language: "sql" | "python"; value: string; attempt: number }
  | { type: "retry"; error: string }
  | TableEvent
  | { type: "value"; value: unknown }
  | { type: "stdout"; value: string }
  | { type: "chart"; spec: ChartSpec }
  | { type: "samples"; items: { label: string; text: string }[] }
  | { type: "answer"; text: string }
  | { type: "error"; message: string }
  | { type: "done"; latency_ms: number };

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

  datasets: () => request<Dataset[]>("/datasets"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Dataset>("/datasets", { method: "POST", body: form });
  },

  sessions: () => request<ChatSession[]>("/chat/sessions"),
  createSession: (datasetId: string) =>
    request<ChatSession>("/chat/sessions", { method: "POST", body: JSON.stringify({ dataset_id: datasetId }) }),
  session: (id: string) => request<ChatSession & { messages: StoredMessage[] }>(`/chat/sessions/${id}`),
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
