"use client";

import { Bar, BarChart, Cell, LabelList, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export type DocChartSpec = {
  kind: "bar" | "donut";
  title: string;
  unit: string;
  fromSource: boolean;
  labels: string[];
  values: number[];
};

// Fixed categorical order, never cycled; the server folds a donut's extra slices into "Lainnya".
const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)"];
const KINDS: Record<string, DocChartSpec["kind"]> = { bar: "bar", batang: "bar", donut: "donut", donat: "donut", pie: "donut" };

const format = (value: number) => value.toLocaleString("id-ID", { maximumFractionDigits: 2 });

/** Same validation as the server: a malformed chart block is dropped, not guessed at. */
export function parseChart(raw: string): DocChartSpec | null {
  try {
    const data = JSON.parse(raw);
    const kind = KINDS[String(data.type ?? "").toLowerCase()];
    const labels: unknown = data.labels;
    const values = Array.isArray(data.values) ? data.values.map(Number) : [];
    if (!kind || !Array.isArray(labels) || labels.length !== values.length || labels.length < 2 || labels.length > 12) return null;
    if (values.some((v: number) => !Number.isFinite(v) || v < 0) || values.reduce((a: number, b: number) => a + b, 0) <= 0) return null;

    let pairs = labels.map((label, i) => ({ label: String(label).slice(0, 40), value: values[i] as number }));
    if (kind === "donut" && pairs.length > COLORS.length) {
      pairs = [...pairs].sort((a, b) => b.value - a.value);
      const rest = pairs.slice(COLORS.length - 1).reduce((sum, p) => sum + p.value, 0);
      pairs = [...pairs.slice(0, COLORS.length - 1), { label: "Lainnya", value: rest }];
    }
    return {
      kind,
      title: String(data.title ?? "").slice(0, 120),
      unit: String(data.unit ?? "").slice(0, 20),
      fromSource: ["dokumen", "document", "file", "sumber"].includes(String(data.source ?? "").toLowerCase()),
      labels: pairs.map((p) => p.label),
      values: pairs.map((p) => p.value),
    };
  } catch {
    return null;
  }
}

const tooltipStyle = {
  background: "var(--chart-surface)",
  border: "1px solid var(--chart-grid)",
  borderRadius: 8,
  color: "var(--foreground)",
  fontSize: 12,
};

export function DocChart({ spec }: { spec: DocChartSpec }) {
  const data = spec.labels.map((label, i) => ({ label, value: spec.values[i] }));
  const total = spec.values.reduce((a, b) => a + b, 0);
  const suffix = spec.unit ? ` ${spec.unit}` : "";
  const share = (value: number) => `${format(Math.round((1000 * value) / total) / 10)}%`;

  return (
    <figure className="rounded-lg border border-border bg-[var(--chart-surface)] p-4">
      {spec.title && <figcaption className="mb-3 text-sm font-medium">{spec.title}</figcaption>}
      {spec.kind === "bar" ? (
        <div style={{ height: 44 * data.length + 16 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 110, bottom: 0, left: 0 }} barCategoryGap="24%">
              <XAxis type="number" hide domain={[0, "dataMax"]} />
              <YAxis
                type="category"
                dataKey="label"
                width={130}
                tickLine={false}
                axisLine={{ stroke: "var(--chart-grid)" }}
                tick={{ fill: "var(--foreground)", fontSize: 12 }}
              />
              <Tooltip
                cursor={{ fill: "var(--chart-grid)", opacity: 0.5 }}
                contentStyle={tooltipStyle}
                formatter={(value) => [`${format(Number(value))}${suffix}`, "Nilai"]}
              />
              <Bar dataKey="value" fill="var(--series-1)" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                <LabelList
                  dataKey="value"
                  position="right"
                  formatter={(value) => `${format(Number(value))}${suffix}`}
                  style={{ fill: "var(--foreground)", fontSize: 12, whiteSpace: "nowrap" }}
                  width={100}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:gap-6">
          <div className="h-44 w-44 shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Tooltip contentStyle={tooltipStyle} formatter={(value, name) => [`${format(Number(value))}${suffix} (${share(Number(value))})`, name]} />
                <Pie
                  data={data}
                  dataKey="value"
                  nameKey="label"
                  innerRadius="55%"
                  outerRadius="100%"
                  startAngle={90}
                  endAngle={-270}
                  stroke="var(--chart-surface)"
                  strokeWidth={2}
                  isAnimationActive={false}
                >
                  {data.map((entry, i) => (
                    <Cell key={entry.label} fill={COLORS[i]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          {/* Every slice is named with its value: identity never rests on colour alone. */}
          <ul className="w-full space-y-1.5 text-sm">
            {data.map((entry, i) => (
              <li key={entry.label} className="flex items-center gap-2">
                <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: COLORS[i] }} />
                <span className="min-w-0 flex-1 truncate">{entry.label}</span>
                <span className="shrink-0 tabular-nums text-muted">
                  {spec.unit === "%" ? share(entry.value) : `${format(entry.value)}${suffix} (${share(entry.value)})`}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="mt-3 text-xs text-muted">
        {spec.fromSource ? "Sumber: dokumen yang dilampirkan" : "Angka perkiraan, perlu diverifikasi"}
      </p>
    </figure>
  );
}
