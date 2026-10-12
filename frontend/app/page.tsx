"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { AssistantMessage } from "@/components/AssistantMessage";
import {
  api,
  ApiError,
  streamMessage,
  type AgentEvent,
  type ChatSession,
  type SessionFile,
  type User,
} from "@/lib/api";

type UiMessage = {
  key: string;
  role: "user" | "assistant";
  content: string;
  steps: AgentEvent[];
  pending: boolean;
  messageId?: string;
};

const DEFAULT_TITLE = "Chat baru";
const ACCEPT = ".csv,.xlsx,.pdf,.png,.jpg,.jpeg,.webp";

const TABLE_SUGGESTIONS = [
  "Ringkas isi dataset ini",
  "Tampilkan tren per bulan",
  "Apa 5 kategori dengan nilai terbesar?",
  "Kolom angka apa yang paling berkorelasi?",
];

const DOCUMENT_SUGGESTIONS = [
  "Ringkas dokumen ini",
  "Apa poin-poin pentingnya?",
  "Apa saran perbaikan untuk dokumen ini?",
  "Adakah tanggal atau tenggat yang disebutkan?",
];

const CAPABILITIES = [
  { title: "Data tabel", text: "CSV atau Excel: total, peringkat, tren, grafik, prediksi, dan sentimen." },
  { title: "Dokumen PDF", text: "Tanya isi dokumen, minta ringkasan, atau minta saran perbaikan." },
  { title: "Gambar & hasil scan", text: "Foto dokumen dan PDF hasil scan dibaca dengan OCR, lalu bisa ditanyakan." },
];

function fileInfo(file: SessionFile): string {
  const count = file.size.toLocaleString("id-ID");
  if (file.kind === "table") return `${count} baris`;
  // "AI" marks pages OCR could not read reliably, which were read by the vision model instead.
  if (file.source === "image") return file.vision_pages > 0 ? "gambar · dibaca AI" : "gambar · OCR";
  const read = [file.ocr_pages > 0 ? `OCR ${file.ocr_pages}` : "", file.vision_pages > 0 ? `AI ${file.vision_pages}` : ""];
  return [`${count} hal`, ...read.filter(Boolean)].join(" · ");
}

function sessionInfo(session: ChatSession): string {
  return session.files.length === 0 ? "Tanpa file" : `${session.files.length} file`;
}

function FileIcon({ file }: { file: Pick<SessionFile, "kind" | "source"> }) {
  const path =
    file.kind === "table"
      ? "M4 5h16v14H4zM4 10h16M4 15h16M10 5v14"
      : file.source === "image"
        ? "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M9 9h.01"
        : "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M9 13h6M9 17h6";
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="shrink-0">
      <path d={path} />
    </svg>
  );
}

function Icon({ d, size = 16 }: { d: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

const TRASH = "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6";
const CLIP = "M21 11.5l-8.6 8.6a5.5 5.5 0 0 1-7.8-7.8l8.6-8.6a3.7 3.7 0 0 1 5.2 5.2l-8.6 8.6a1.8 1.8 0 0 1-2.6-2.6l7.9-7.9";

export default function Home() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const messageCounter = useRef(0);
  // Set while this tab itself fills a session it just created, so the history fetch does not clear it.
  const skipLoadFor = useRef<string | null>(null);

  useEffect(() => {
    api
      .me()
      .then(async (me) => {
        setUser(me);
        const list = await api.sessions();
        setSessions(list);
        if (list[0]) setActiveId(list[0].id);
      })
      .catch(() => router.replace("/login"));
  }, [router]);

  useEffect(() => {
    if (!activeId) return;
    if (skipLoadFor.current === activeId) {
      skipLoadFor.current = null;
      return;
    }
    api
      .session(activeId)
      .then((detail) =>
        setMessages(
          detail.messages.map((m) => ({
            key: m.id,
            role: m.role,
            content: m.content,
            steps: m.steps_json,
            pending: false,
            messageId: m.role === "assistant" ? m.id : undefined,
          })),
        ),
      )
      .catch((err) => setNotice(err instanceof ApiError ? err.message : "Gagal memuat percakapan"));
  }, [activeId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const activeSession = sessions.find((s) => s.id === activeId);
  const files = activeSession?.files ?? [];
  const hasDocument = files.some((f) => f.kind === "document");
  const hasTable = files.some((f) => f.kind === "table");

  const patchSession = useCallback((id: string, fn: (s: ChatSession) => ChatSession) => {
    setSessions((prev) => prev.map((s) => (s.id === id ? fn(s) : s)));
  }, []);

  function openSession(id: string | null) {
    setActiveId(id);
    setMessages([]);
    setNotice(null);
    setMenuOpen(false);
  }

  /** The session to act on: the open one, or a new one created on first use. */
  async function ensureSession(): Promise<string> {
    if (activeId) return activeId;
    const session = await api.createSession();
    setSessions((prev) => [session, ...prev]);
    skipLoadFor.current = session.id;
    setActiveId(session.id);
    return session.id;
  }

  async function newChat() {
    setNotice(null);
    try {
      const session = await api.createSession();
      setSessions((prev) => [session, ...prev]);
      skipLoadFor.current = session.id;
      openSession(session.id);
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Gagal membuat chat baru");
    }
  }

  async function attach(list: File[]) {
    if (list.length === 0 || uploading) return;
    setNotice(null);
    try {
      const sessionId = await ensureSession();
      for (const file of list) {
        setUploading(file.name);
        const added = await api.attachFile(sessionId, file);
        patchSession(sessionId, (s) => ({
          ...s,
          title: s.title === DEFAULT_TITLE ? added.name : s.title,
          files: [...s.files, added],
        }));
      }
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Upload gagal");
    } finally {
      setUploading(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  const updateLast = useCallback((fn: (msg: UiMessage) => UiMessage) => {
    setMessages((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])]);
  }, []);

  async function ask(question: string) {
    if (!question.trim() || busy) return;
    setBusy(true);
    setNotice(null);
    setInput("");
    const stamp = ++messageCounter.current;
    let sessionId: string;
    try {
      sessionId = await ensureSession();
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Gagal memulai chat");
      setBusy(false);
      return;
    }
    patchSession(sessionId, (s) => (s.title === DEFAULT_TITLE ? { ...s, title: question.slice(0, 60) } : s));
    setMessages((prev) => [
      ...prev,
      { key: `u${stamp}`, role: "user", content: question, steps: [], pending: false },
      { key: `a${stamp}`, role: "assistant", content: "", steps: [], pending: true },
    ]);
    try {
      await streamMessage(sessionId, question, (event) => {
        if (event.type === "done") {
          // The reply is saved by now; its id is what the PDF download needs.
          updateLast((msg) => ({ ...msg, messageId: event.message_id }));
          return;
        }
        updateLast((msg) => ({
          ...msg,
          steps: [...msg.steps, event],
          content: event.type === "answer" ? event.text : msg.content,
        }));
      });
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Koneksi terputus";
      updateLast((msg) => ({ ...msg, steps: [...msg.steps, { type: "error", message }] }));
    } finally {
      updateLast((msg) => ({ ...msg, pending: false }));
      setBusy(false);
    }
  }

  async function removeFile(file: SessionFile) {
    if (!activeId || !window.confirm(`Hapus "${file.name}" dari chat ini?`)) return;
    try {
      await api.removeFile(file.id);
      patchSession(activeId, (s) => ({ ...s, files: s.files.filter((f) => f.id !== file.id) }));
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Gagal menghapus file");
    }
  }

  async function removeSession(session: ChatSession) {
    if (!window.confirm(`Hapus chat "${session.title}" beserta semua filenya? Tindakan ini tidak bisa dibatalkan.`)) return;
    try {
      await api.deleteSession(session.id);
      const remaining = sessions.filter((s) => s.id !== session.id);
      setSessions(remaining);
      if (session.id === activeId) openSession(remaining[0]?.id ?? null);
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Gagal menghapus chat");
    }
  }

  async function logout() {
    await api.logout().catch(() => undefined);
    router.replace("/login");
  }

  if (!user) {
    return <main className="flex flex-1 items-center justify-center text-sm text-muted">Memuat...</main>;
  }

  const suggestions = hasDocument ? DOCUMENT_SUGGESTIONS : hasTable ? TABLE_SUGGESTIONS : [];

  return (
    <div className="flex h-dvh">
      <input
        ref={fileInput}
        type="file"
        accept={ACCEPT}
        multiple
        className="hidden"
        onChange={(e) => attach(Array.from(e.target.files ?? []))}
      />
      {menuOpen && (
        <button aria-label="Tutup menu" onClick={() => setMenuOpen(false)} className="fixed inset-0 z-20 bg-black/40 md:hidden" />
      )}
      <aside
        className={`${menuOpen ? "flex" : "hidden"} fixed inset-y-0 left-0 z-30 w-72 max-w-[85vw] shrink-0 flex-col border-r border-border bg-surface md:static md:flex`}
      >
        <div className="p-4">
          <h1 className="font-semibold">Agentic Data Analyst</h1>
          <button
            onClick={newChat}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white"
          >
            <Icon d="M12 5v14M5 12h14" />
            Chat baru
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2" aria-label="Percakapan">
          {sessions.length === 0 && <p className="px-3 py-2 text-xs text-muted">Belum ada percakapan.</p>}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`group mb-1 flex items-center rounded-lg border-l-2 text-sm ${
                s.id === activeId
                  ? "border-accent bg-background"
                  : "border-transparent text-muted hover:bg-background hover:text-foreground"
              }`}
            >
              <button onClick={() => openSession(s.id)} className="min-w-0 flex-1 px-3 py-2 text-left">
                <span className={`block truncate ${s.id === activeId ? "font-medium text-foreground" : ""}`}>{s.title}</span>
                <span className="block truncate text-xs text-muted">{sessionInfo(s)}</span>
              </button>
              <button
                onClick={() => removeSession(s)}
                disabled={busy}
                aria-label={`Hapus chat ${s.title}`}
                title="Hapus chat ini"
                className="mr-1 shrink-0 rounded-md p-2 text-muted hover:text-red-600 focus:opacity-100 disabled:opacity-40 md:opacity-0 md:group-hover:opacity-100 dark:hover:text-red-400"
              >
                <Icon d={TRASH} size={14} />
              </button>
            </div>
          ))}
        </nav>

        <p className="px-4 pb-2 text-xs text-muted">File dan percakapan dihapus otomatis setelah 24 jam.</p>
        <div className="flex items-center justify-between border-t border-border p-4 text-xs text-muted">
          <span className="truncate">{user.email}</span>
          <button onClick={logout} className="hover:text-foreground">
            Keluar
          </button>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="border-b border-border bg-surface px-4 py-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMenuOpen(true)}
              aria-label="Buka menu"
              className="rounded-md p-1.5 text-muted hover:text-foreground md:hidden"
            >
              <Icon d="M4 6h16M4 12h16M4 18h16" size={20} />
            </button>
            <p className="min-w-0 truncate text-sm font-medium">{activeSession?.title ?? "Agentic Data Analyst"}</p>
          </div>
          {(files.length > 0 || uploading) && (
            <ul className="mt-2 flex flex-wrap gap-2" aria-label="File di chat ini">
              {files.map((file) => (
                <li
                  key={file.id}
                  className="flex max-w-full items-center gap-1.5 rounded-full border border-border bg-background py-1 pl-2.5 pr-1 text-xs"
                >
                  <FileIcon file={file} />
                  <span className="max-w-40 truncate font-medium sm:max-w-56">{file.name}</span>
                  <span className="shrink-0 text-muted">{fileInfo(file)}</span>
                  <button
                    onClick={() => removeFile(file)}
                    aria-label={`Hapus ${file.name}`}
                    className="rounded-full p-1 text-muted hover:text-red-600 dark:hover:text-red-400"
                  >
                    <Icon d="M6 6l12 12M18 6L6 18" size={12} />
                  </button>
                </li>
              ))}
              {uploading && (
                <li className="flex items-center gap-2 rounded-full border border-dashed border-border px-2.5 py-1 text-xs text-muted" aria-live="polite">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
                  <span className="max-w-48 truncate">Membaca {uploading}...</span>
                </li>
              )}
            </ul>
          )}
        </header>

        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
            {messages.length === 0 && (
              <div className="py-8 text-center sm:py-14">
                <h2 className="text-xl font-semibold">
                  {files.length > 0 ? "File siap. Mau tanya apa?" : "Tanya data Anda dalam Bahasa Indonesia"}
                </h2>
                <p className="mx-auto mt-2 max-w-md text-sm text-muted">
                  {files.length > 0
                    ? "Anda bisa menambah file lain kapan saja lewat tombol lampiran di bawah."
                    : "Mulai dengan bertanya, atau lampirkan file: tabel, PDF, atau foto dokumen."}
                </p>
                {suggestions.length > 0 ? (
                  <div className="mt-8 grid gap-2 text-left sm:grid-cols-2">
                    {suggestions.map((q) => (
                      <button
                        key={q}
                        onClick={() => ask(q)}
                        className="rounded-lg border border-border bg-surface px-4 py-3 text-left text-sm hover:border-accent"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                ) : (
                  <>
                    <button
                      onClick={() => fileInput.current?.click()}
                      disabled={!!uploading}
                      className="mt-6 inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white disabled:opacity-60"
                    >
                      <Icon d={CLIP} />
                      Lampirkan file
                    </button>
                    <div className="mt-10 grid gap-3 text-left sm:grid-cols-3">
                      {CAPABILITIES.map((item) => (
                        <div key={item.title} className="rounded-xl border border-border bg-surface p-4">
                          <p className="text-sm font-medium">{item.title}</p>
                          <p className="mt-1 text-xs leading-relaxed text-muted">{item.text}</p>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {messages.map((msg) =>
              msg.role === "user" ? (
                <div key={msg.key} className="flex justify-end">
                  <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent px-4 py-2 text-sm text-white">
                    {msg.content}
                  </p>
                </div>
              ) : (
                <div key={msg.key} className="rounded-2xl rounded-tl-md border border-border bg-surface p-4 text-sm shadow-sm sm:p-5">
                  <AssistantMessage steps={msg.steps} pending={msg.pending} messageId={msg.messageId} />
                </div>
              ),
            )}
            <div ref={bottom} />
          </div>
        </div>

        {notice && (
          <p role="alert" className="mx-auto mb-2 w-full max-w-3xl px-4 text-sm text-red-600 dark:text-red-400">
            {notice}
          </p>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
          className="border-t border-border bg-surface p-3 sm:p-4"
        >
          <div className="mx-auto flex max-w-3xl items-center gap-2">
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              disabled={!!uploading}
              aria-label="Lampirkan file"
              title="Lampirkan CSV, Excel, PDF, atau gambar"
              className="shrink-0 rounded-lg border border-border p-2.5 text-muted hover:border-accent hover:text-foreground disabled:opacity-60"
            >
              <Icon d={CLIP} size={18} />
            </button>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPaste={(e) => {
                // A screenshot or copied image pasted into the box is attached like any other file.
                const pasted = Array.from(e.clipboardData.files);
                if (pasted.length > 0) {
                  e.preventDefault();
                  attach(pasted);
                }
              }}
              disabled={busy}
              maxLength={2000}
              placeholder={files.length > 0 ? "Tanyakan sesuatu tentang file ini..." : "Tulis pertanyaan, atau lampirkan file..."}
              aria-label="Pertanyaan"
              className="min-w-0 flex-1 rounded-lg border border-border bg-background px-3 py-2.5 text-sm outline-none focus:border-accent disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              className="shrink-0 rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white disabled:opacity-60"
            >
              Kirim
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
