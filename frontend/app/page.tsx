"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { AssistantMessage } from "@/components/AssistantMessage";
import { api, ApiError, streamMessage, type AgentEvent, type ChatSession, type User } from "@/lib/api";

type UiMessage = {
  key: string;
  role: "user" | "assistant";
  content: string;
  steps: AgentEvent[];
  pending: boolean;
  messageId?: string;
};

const DOCUMENT_SUGGESTIONS = [
  "Ringkas dokumen ini",
  "Apa poin-poin pentingnya?",
  "Apa saran perbaikan untuk dokumen ini?",
  "Adakah tanggal atau tenggat yang disebutkan?",
];

const CAPABILITIES = [
  { title: "Data tabel", text: "CSV atau Excel: total, peringkat, tren, dan grafik dari pertanyaan biasa." },
  { title: "Prediksi & sentimen", text: "Perkiraan ke depan dari data bertanggal, dan sentimen dari kolom ulasan." },
  { title: "Dokumen PDF", text: "Tanya isi dokumen, minta ringkasan, atau minta saran perbaikan." },
];

function sessionInfo(session: ChatSession): string {
  const count = session.size.toLocaleString("id-ID");
  return session.kind === "document" ? `PDF · ${count} halaman` : `Tabel · ${count} baris`;
}

function KindIcon({ kind }: { kind: ChatSession["kind"] }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="shrink-0">
      {kind === "document" ? (
        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M9 13h6M9 17h6" />
      ) : (
        <path d="M4 5h16v14H4zM4 10h16M4 15h16M10 5v14" />
      )}
    </svg>
  );
}

const SUGGESTIONS = [
  "Ringkas isi dataset ini",
  "Tampilkan tren per bulan",
  "Apa 5 kategori dengan nilai terbesar?",
  "Kolom angka apa yang paling berkorelasi?",
];

export default function Home() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const messageCounter = useRef(0);

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
  const activeKind = activeSession?.kind ?? "table";

  async function upload(file: File) {
    setUploading(true);
    setNotice(null);
    try {
      const dataset = await api.upload(file);
      const session = await api.createSession(dataset.id);
      setSessions((prev) => [session, ...prev]);
      setMessages([]);
      setActiveId(session.id);
      setMenuOpen(false);
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Upload gagal");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  const updateLast = useCallback((fn: (msg: UiMessage) => UiMessage) => {
    setMessages((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])]);
  }, []);

  async function ask(question: string) {
    if (!activeId || !question.trim() || busy) return;
    setBusy(true);
    setNotice(null);
    setInput("");
    const stamp = ++messageCounter.current;
    setMessages((prev) => [
      ...prev,
      { key: `u${stamp}`, role: "user", content: question, steps: [], pending: false },
      { key: `a${stamp}`, role: "assistant", content: "", steps: [], pending: true },
    ]);
    try {
      await streamMessage(activeId, question, (event) => {
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

  async function removeDataset(session: ChatSession) {
    if (!window.confirm(`Hapus "${session.title}" beserta percakapannya? Tindakan ini tidak bisa dibatalkan.`)) return;
    setNotice(null);
    try {
      await api.deleteDataset(session.dataset_id);
      // Deleting a dataset removes every conversation that used it.
      const remaining = sessions.filter((s) => s.dataset_id !== session.dataset_id);
      setSessions(remaining);
      if (!remaining.some((s) => s.id === activeId)) {
        setMessages([]);
        setActiveId(remaining[0]?.id ?? null);
      }
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Gagal menghapus data");
    }
  }

  async function logout() {
    await api.logout().catch(() => undefined);
    router.replace("/login");
  }

  if (!user) {
    return <main className="flex flex-1 items-center justify-center text-sm text-muted">Memuat...</main>;
  }

  return (
    <div className="flex h-dvh">
      {menuOpen && (
        <button
          aria-label="Tutup menu"
          onClick={() => setMenuOpen(false)}
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
        />
      )}
      <aside
        className={`${menuOpen ? "flex" : "hidden"} fixed inset-y-0 left-0 z-30 w-72 max-w-[85vw] shrink-0 flex-col border-r border-border bg-surface md:static md:flex`}
      >
        <div className="p-4">
          <h1 className="font-semibold">Agentic Data Analyst</h1>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.xlsx,.pdf"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
          <button
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
            className="mt-4 w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-60"
          >
            {uploading ? "Mengunggah..." : "Unggah CSV / Excel / PDF"}
          </button>
          <p className="mt-2 text-xs text-muted">Maks 50 MB. Data dihapus otomatis setelah 24 jam.</p>
        </div>

        <nav className="flex-1 overflow-y-auto px-2" aria-label="Percakapan">
          {sessions.length === 0 && <p className="px-3 py-2 text-xs text-muted">Belum ada file yang diunggah.</p>}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`group mb-1 flex items-center rounded-lg border-l-2 text-sm ${
                s.id === activeId
                  ? "border-accent bg-background"
                  : "border-transparent text-muted hover:bg-background hover:text-foreground"
              }`}
            >
              <button
                onClick={() => {
                  setActiveId(s.id);
                  setMenuOpen(false);
                }}
                className="flex min-w-0 flex-1 items-center gap-2.5 px-3 py-2 text-left"
              >
                <KindIcon kind={s.kind} />
                <span className="min-w-0">
                  <span className={`block truncate ${s.id === activeId ? "font-medium text-foreground" : ""}`}>{s.title}</span>
                  <span className="block truncate text-xs text-muted">{sessionInfo(s)}</span>
                </span>
              </button>
              <button
                onClick={() => removeDataset(s)}
                disabled={busy}
                aria-label={`Hapus ${s.title}`}
                title="Hapus data ini"
                className="mr-1 shrink-0 rounded-md p-2 text-muted hover:text-red-600 focus:opacity-100 disabled:opacity-40 md:opacity-0 md:group-hover:opacity-100 dark:hover:text-red-400"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6" />
                </svg>
              </button>
            </div>
          ))}
        </nav>

        <div className="flex items-center justify-between border-t border-border p-4 text-xs text-muted">
          <span className="truncate">{user.email}</span>
          <button onClick={logout} className="hover:text-foreground">
            Keluar
          </button>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-border bg-surface px-4 py-3">
          <button
            onClick={() => setMenuOpen(true)}
            aria-label="Buka menu"
            className="rounded-md p-1.5 text-muted hover:text-foreground md:hidden"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          {activeSession ? (
            <div className="flex min-w-0 items-center gap-2.5">
              <KindIcon kind={activeSession.kind} />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">{activeSession.title}</p>
                <p className="text-xs text-muted">{sessionInfo(activeSession)}</p>
              </div>
            </div>
          ) : (
            <p className="text-sm font-medium md:text-muted">Agentic Data Analyst</p>
          )}
        </header>
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
            {!activeId && (
              <div className="py-12 text-center sm:py-20">
                <h2 className="text-xl font-semibold">Tanya data Anda dalam Bahasa Indonesia</h2>
                <p className="mx-auto mt-2 max-w-md text-sm text-muted">
                  Unggah file, lalu ajukan pertanyaan. Jawaban disertai tabel, grafik, dan cara perhitungannya.
                </p>
                <button
                  onClick={() => fileInput.current?.click()}
                  disabled={uploading}
                  className="mt-6 rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white disabled:opacity-60"
                >
                  {uploading ? "Mengunggah..." : "Unggah CSV / Excel / PDF"}
                </button>
                <div className="mt-10 grid gap-3 text-left sm:grid-cols-3">
                  {CAPABILITIES.map((item) => (
                    <div key={item.title} className="rounded-xl border border-border bg-surface p-4">
                      <p className="text-sm font-medium">{item.title}</p>
                      <p className="mt-1 text-xs leading-relaxed text-muted">{item.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeId && messages.length === 0 && (
              <div>
                <p className="mb-3 text-sm text-muted">Coba salah satu pertanyaan ini:</p>
                <div className="grid gap-2 sm:grid-cols-2">
                {(activeKind === "document" ? DOCUMENT_SUGGESTIONS : SUGGESTIONS).map((q) => (
                  <button
                    key={q}
                    onClick={() => ask(q)}
                    className="rounded-lg border border-border bg-surface px-4 py-3 text-left text-sm hover:border-accent"
                  >
                    {q}
                  </button>
                ))}
                </div>
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
          <p role="alert" className="mx-auto mb-2 max-w-3xl px-4 text-sm text-red-600 dark:text-red-400">
            {notice}
          </p>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
          className="border-t border-border bg-surface p-4"
        >
          <div className="mx-auto flex max-w-3xl gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={!activeId || busy}
              maxLength={2000}
              placeholder={
                !activeId
                  ? "Unggah data terlebih dahulu"
                  : activeKind === "document"
                    ? "Tanyakan sesuatu tentang dokumen ini..."
                    : "Tanyakan sesuatu tentang data Anda..."
              }
              aria-label="Pertanyaan"
              className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={!activeId || busy || !input.trim()}
              className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
            >
              Kirim
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
