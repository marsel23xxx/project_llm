"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatCell } from "@/components/DataTable";
import type { ChartSpec, TableEvent } from "@/lib/api";

// Fixed categorical order, never cycled: the backend caps a chart at three series.
const SERIES_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const AXIS_TICK = { fill: "var(--chart-text)", fontSize: 12 };

const compact = (value: number) => value.toLocaleString("id-ID", { notation: "compact", maximumFractionDigits: 1 });

export function ResultChart({ spec, table }: { spec: ChartSpec; table: TableEvent }) {
  const xType = table.types[table.columns.indexOf(spec.x)];
  const data = table.rows.map((row) => {
    const point: Record<string, unknown> = {};
    table.columns.forEach((col, i) => {
      point[col] = col === spec.x ? formatCell(row[i], xType) : row[i];
    });
    return point;
  });
  const showLegend = spec.y.length > 1;

  const shared = (
    <>
      <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
      <XAxis dataKey={spec.x} tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: "var(--chart-grid)" }} minTickGap={16} />
      <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} tickFormatter={compact} width={56} />
      <Tooltip
        cursor={spec.kind === "bar" ? { fill: "var(--chart-grid)", opacity: 0.5 } : { stroke: "var(--chart-text)" }}
        contentStyle={{
          background: "var(--chart-surface)",
          border: "1px solid var(--chart-grid)",
          borderRadius: 8,
          color: "var(--foreground)",
          fontSize: 12,
        }}
        formatter={(value) => (typeof value === "number" ? value.toLocaleString("id-ID") : String(value))}
      />
      {showLegend && (
        <Legend
          wrapperStyle={{ fontSize: 12 }}
          // Labels stay in text ink; the swatch beside each one carries the series color.
          formatter={(value) => <span style={{ color: "var(--chart-text)" }}>{value}</span>}
        />
      )}
    </>
  );

  return (
    <figure className="rounded-lg border border-border bg-[var(--chart-surface)] p-3">
      <figcaption className="mb-2 text-xs text-muted">
        {spec.y.join(", ")} per {spec.x}
      </figcaption>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {spec.kind === "line" ? (
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              {shared}
              {spec.y.map((key, i) => (
                <Line
                  key={key}
                  dataKey={key}
                  stroke={SERIES_COLORS[i]}
                  strokeWidth={2}
                  dot={data.length <= 24 ? { r: 4, strokeWidth: 2, stroke: "var(--chart-surface)", fill: SERIES_COLORS[i] } : false}
                  activeDot={{ r: 5, strokeWidth: 2, stroke: "var(--chart-surface)" }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          ) : (
            <BarChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }} barCategoryGap="24%" barGap={2}>
              {shared}
              {spec.y.map((key, i) => (
                <Bar key={key} dataKey={key} fill={SERIES_COLORS[i]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
