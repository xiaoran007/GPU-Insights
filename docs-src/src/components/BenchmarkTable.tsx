import type { BenchmarkEntry } from "../types/benchmark";
import { formatScore, formatBatchSize } from "../utils/format";

interface BenchmarkTableProps {
  entries: BenchmarkEntry[];
}

export default function BenchmarkTable({ entries }: BenchmarkTableProps) {
  return (
    <section className="rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 font-[var(--font-display)] text-lg">Benchmark Results</h2>
        <p className="m-0 font-[var(--font-mono)] text-sm tabular-nums text-[var(--color-muted)]">
          {entries.length.toLocaleString()} entries
        </p>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[960px] border-collapse">
          <thead>
            <tr>
              {["Device", "Version", "Platform", "FP32", "FP32 Batch", "FP16", "FP16 Batch", "Notes", "Date"].map(
                (h) => (
                  <th
                    key={h}
                    className="sticky top-0 z-10 border-b border-[var(--color-line)] bg-[#f4f8fc] px-2.5 py-2.5 text-left text-xs uppercase tracking-wider text-[#334155]"
                  >
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, i) => (
              <tr key={i} className="hover:bg-[#f7fbff]">
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <div className="font-semibold">{entry.device}</div>
                  {entry.memory && (
                    <small className="text-[var(--color-muted)]">{entry.memory}</small>
                  )}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <VersionBadge version={entry.version} />
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  {entry.platform || "N/A"}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] font-semibold tabular-nums text-[var(--color-brand)]">
                  {formatScore(entry.fp32)}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <BatchTag value={entry.fp32bs} />
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] font-semibold tabular-nums text-[var(--color-accent)]">
                  {formatScore(entry.fp16)}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <BatchTag value={entry.fp16bs} />
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 italic text-[var(--color-muted)]">
                  {entry.note || "N/A"}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  {entry.date || "N/A"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {entries.length === 0 && (
        <p className="mt-3 text-[var(--color-muted)]">
          No benchmark entries match the current filters.
        </p>
      )}
    </section>
  );
}

function VersionBadge({ version }: { version: string }) {
  const isV2 = version === "ver2";
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-bold ${
        isV2
          ? "bg-[var(--color-brand-soft)] text-[var(--color-brand-strong)]"
          : "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
      }`}
    >
      {version.toUpperCase()}
    </span>
  );
}

function BatchTag({ value }: { value: number }) {
  return (
    <span className="inline-block rounded-full border border-[var(--color-line)] bg-[var(--color-surface-soft)] px-2 py-0.5 font-[var(--font-mono)] text-xs tabular-nums">
      {formatBatchSize(value)}
    </span>
  );
}
