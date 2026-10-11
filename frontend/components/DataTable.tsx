import type { TableEvent } from "@/lib/api";

const VISIBLE_ROWS = 50;
const NUMERIC = /INT|DOUBLE|FLOAT|DECIMAL|REAL|NUMERIC/i;

export function formatCell(value: unknown, type?: string): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") return value.toLocaleString("id-ID", { maximumFractionDigits: 2 });
  if (typeof value === "string" && type && /TIMESTAMP|DATE/i.test(type)) return value.slice(0, 10);
  return String(value);
}

export function DataTable({ table }: { table: TableEvent }) {
  const rows = table.rows.slice(0, VISIBLE_ROWS);
  const hidden = table.rows.length - rows.length;

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <div className="max-h-80 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-background text-left text-xs text-muted">
            <tr>
              {table.columns.map((col, i) => (
                <th
                  key={col}
                  className={`px-3 py-2 font-medium ${NUMERIC.test(table.types[i]) ? "text-right" : ""}`}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, r) => (
              <tr key={r} className="border-t border-border">
                {row.map((cell, c) => (
                  <td
                    key={c}
                    className={`px-3 py-1.5 ${NUMERIC.test(table.types[c]) ? "text-right tabular-nums" : ""}`}
                  >
                    {formatCell(cell, table.types[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(hidden > 0 || table.truncated) && (
        <p className="border-t border-border px-3 py-1.5 text-xs text-muted">
          Menampilkan {rows.length} dari {table.rows.length}
          {table.truncated ? "+" : ""} baris
        </p>
      )}
    </div>
  );
}
