"use client";

import { DataTable } from "@/components/DataTable";
import { ResultChart } from "@/components/ResultChart";
import type { AgentEvent, TableEvent } from "@/lib/api";

const INTENT_LABEL = {
  sql: "Query SQL",
  python: "Analisis Python",
  sentiment: "Analisis Sentimen",
  chat: "Obrolan",
} as const;

function progressLabel(steps: AgentEvent[]): string {
  const last = steps.at(-1);
  if (!last) return "Memahami pertanyaan...";
  switch (last.type) {
    case "intent":
      if (last.value === "chat") return "Menyusun jawaban...";
      return last.value === "sentiment" ? "Membaca dan menilai teks..." : "Menulis kode analisis...";
    case "code":
      return "Menjalankan kode...";
    case "retry":
      return "Memperbaiki kode...";
    default:
      return "Merangkum hasil...";
  }
}

export function AssistantMessage({ steps, pending }: { steps: AgentEvent[]; pending: boolean }) {
  const intent = steps.find((s) => s.type === "intent");
  const answer = steps.find((s) => s.type === "answer");
  const error = steps.find((s) => s.type === "error");
  const table = steps.findLast((s): s is TableEvent => s.type === "table");
  const chart = steps.findLast((s) => s.type === "chart");
  const value = steps.findLast((s) => s.type === "value");
  const samples = steps.findLast((s) => s.type === "samples");
  const process = steps.filter((s) => s.type === "code" || s.type === "retry" || s.type === "stdout");

  return (
    <div className="space-y-3">
      {intent && (
        <span className="inline-block rounded-full border border-border px-2 py-0.5 text-xs text-muted">
          {INTENT_LABEL[intent.value]}
        </span>
      )}

      {answer ? (
        <p className="whitespace-pre-wrap leading-relaxed">{answer.text}</p>
      ) : error ? (
        <p role="alert" className="text-red-600 dark:text-red-400">
          {error.message}
        </p>
      ) : (
        pending && (
          <p className="flex items-center gap-2 text-sm text-muted" aria-live="polite">
            <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
            {progressLabel(steps)}
          </p>
        )
      )}

      {chart && table && <ResultChart spec={chart.spec} table={table} />}
      {table && <DataTable table={table} />}
      {samples && samples.items.length > 0 && (
        <div>
          <p className="mb-1 text-xs text-muted">Contoh teks</p>
          <ul className="space-y-1.5">
            {samples.items.map((item, i) => (
              <li key={i} className="flex gap-2 rounded-lg border border-border px-3 py-2">
                <span
                  className={`h-fit shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${
                    item.label === "negatif"
                      ? "bg-red-500/10 text-red-700 dark:text-red-400"
                      : "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                  }`}
                >
                  {item.label}
                </span>
                <span className="text-sm">{item.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {value && (
        <pre className="overflow-x-auto rounded-lg border border-border bg-background p-3 text-xs">
          {JSON.stringify(value.value, null, 2)}
        </pre>
      )}

      {process.length > 0 && (
        <details className="rounded-lg border border-border text-sm">
          <summary className="cursor-pointer px-3 py-2 text-muted">
            Lihat proses agent ({process.filter((s) => s.type === "code").length} percobaan kode)
          </summary>
          <div className="space-y-2 border-t border-border p-3">
            {process.map((step, i) =>
              step.type === "code" ? (
                <div key={i}>
                  <p className="mb-1 text-xs text-muted">
                    {step.language.toUpperCase()}, percobaan {step.attempt}
                  </p>
                  <pre className="overflow-x-auto rounded-md bg-background p-2 font-mono text-xs">{step.value}</pre>
                </div>
              ) : step.type === "retry" ? (
                <p key={i} className="rounded-md bg-amber-500/10 px-2 py-1 text-xs text-amber-700 dark:text-amber-400">
                  Gagal: {step.error}
                </p>
              ) : step.type === "stdout" ? (
                <pre key={i} className="overflow-x-auto rounded-md bg-background p-2 font-mono text-xs">
                  {step.value}
                </pre>
              ) : null,
            )}
          </div>
        </details>
      )}
    </div>
  );
}
