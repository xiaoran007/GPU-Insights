import type { BenchmarkEntry } from "../types/benchmark";
import { formatScore } from "../utils/format";

interface StatsGridProps {
  entries: BenchmarkEntry[];
}

export default function StatsGrid({ entries }: StatsGridProps) {
  const fp32Scores = entries
    .map((e) => e.fp32)
    .filter((v): v is number => typeof v === "number");
  const fp16Scores = entries
    .map((e) => e.fp16)
    .filter((v): v is number => typeof v === "number");
  const vendors = new Set(entries.map((e) => e.vendor));

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <StatCard label="Visible Entries" value={entries.length.toLocaleString()} />
      <StatCard
        label="Top FP32"
        value={fp32Scores.length ? formatScore(Math.max(...fp32Scores)) : "N/A"}
      />
      <StatCard
        label="Top FP16"
        value={fp16Scores.length ? formatScore(Math.max(...fp16Scores)) : "N/A"}
      />
      <StatCard label="Vendors" value={vendors.size.toLocaleString()} />
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="rounded-2xl border border-[var(--color-line)] bg-white p-4 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <p className="m-0 text-sm text-[var(--color-muted)]">{label}</p>
      <p className="m-0 mt-1 font-[var(--font-mono)] text-[clamp(1.4rem,4vw,2rem)] font-bold tabular-nums">
        {value}
      </p>
    </article>
  );
}
