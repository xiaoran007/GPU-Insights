import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";
import { useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import type { LlmBenchmarkEntry, LlmInferenceData } from "../types/llmInference";
import { formatTps, toTitleCase } from "../utils/format";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

interface LlmInferencePageProps {
  data: LlmInferenceData;
}

export default function LlmInferencePage({ data }: LlmInferencePageProps) {
  const [selectedCase, setSelectedCase] = useState("all");
  const caseOptions = useMemo(() => buildCaseOptions(data.benchmarks), [data.benchmarks]);
  const entries = [...data.benchmarks]
    .filter((entry) => selectedCase === "all" || entry.caseName === selectedCase)
    .sort((a, b) => (b.tgTps ?? -Infinity) - (a.tgTps ?? -Infinity));
  const successfulEntries = entries.filter((entry) => entry.status !== "failed");
  const topTg = entries
    .map((entry) => entry.tgTps)
    .filter((value): value is number => typeof value === "number");
  const topPp = entries
    .map((entry) => entry.ppTps)
    .filter((value): value is number => typeof value === "number");
  const model = Object.values(data.models)[0];

  return (
    <div className="grid min-w-0 gap-4">
      {model && (
        <section className="min-w-0 rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
          <div className="grid min-w-0 gap-3 md:grid-cols-[1.2fr_1fr] md:items-end">
            <div className="min-w-0">
              <p className="m-0 text-sm font-semibold uppercase tracking-wider text-[var(--color-brand)]">
                Canonical llama.cpp track
              </p>
              <h2 className="m-0 mt-1 font-[var(--font-display)] text-2xl">
                {model.displayName}
              </h2>
              <p className="m-0 mt-1 text-[var(--color-muted)]">
                {model.baseModel} · {model.artifact} · {model.parameters}
              </p>
            </div>
            <code className="block max-w-full overflow-x-auto rounded-lg bg-[var(--color-surface-soft)] px-3 py-2 text-xs text-[#334155]">
              {model.defaultCommand}
            </code>
          </div>
        </section>
      )}

      <CaseSelector
        cases={caseOptions}
        selectedCase={selectedCase}
        onSelect={setSelectedCase}
      />

      <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Visible Entries" value={entries.length.toLocaleString()} />
        <MetricCard label="Successful" value={successfulEntries.length.toLocaleString()} />
        <MetricCard label="Top TG tok/s" value={topTg.length ? formatTps(Math.max(...topTg)) : "N/A"} />
        <MetricCard label="Top PP tok/s" value={topPp.length ? formatTps(Math.max(...topPp)) : "N/A"} />
      </div>

      <LlmThroughputChart entries={successfulEntries} />
      <LlmResultTable entries={entries} />
    </div>
  );
}

function buildCaseOptions(entries: LlmBenchmarkEntry[]) {
  const seen = new Set<string>();
  return entries
    .filter((entry) => {
      if (seen.has(entry.caseName)) return false;
      seen.add(entry.caseName);
      return true;
    })
    .sort((a, b) => a.promptTokens - b.promptTokens || a.generationTokens - b.generationTokens)
    .map((entry) => ({
      name: entry.caseName,
      promptTokens: entry.promptTokens,
      generationTokens: entry.generationTokens,
      count: entries.filter((candidate) => candidate.caseName === entry.caseName).length,
    }));
}

function CaseSelector({
  cases,
  selectedCase,
  onSelect,
}: {
  cases: Array<{ name: string; promptTokens: number; generationTokens: number; count: number }>;
  selectedCase: string;
  onSelect: (caseName: string) => void;
}) {
  const totalCount = cases.reduce((sum, item) => sum + item.count, 0);

  return (
    <section className="grid gap-3 rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 font-[var(--font-display)] text-lg">Cases</h2>
        <p className="m-0 font-[var(--font-mono)] text-sm tabular-nums text-[var(--color-muted)]">
          {totalCount.toLocaleString()} entries
        </p>
      </div>
      <div className="flex flex-wrap gap-2">
        <CaseButton
          active={selectedCase === "all"}
          title="All Cases"
          subtitle={`${totalCount.toLocaleString()} entries`}
          onClick={() => onSelect("all")}
        />
        {cases.map((item) => (
          <CaseButton
            key={item.name}
            active={selectedCase === item.name}
            title={item.name}
            subtitle={`${item.promptTokens.toLocaleString()}p / ${item.generationTokens.toLocaleString()}g · ${item.count.toLocaleString()}`}
            onClick={() => onSelect(item.name)}
          />
        ))}
      </div>
    </section>
  );
}

function CaseButton({
  active,
  title,
  subtitle,
  onClick,
}: {
  active: boolean;
  title: string;
  subtitle: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`max-w-full cursor-pointer rounded-full border px-4 py-2 text-left transition-colors ${
        active
          ? "border-[var(--color-brand)] bg-[var(--color-brand)] text-[#ebfffd]"
          : "border-[var(--color-line)] bg-[var(--color-surface-soft)] text-[var(--color-text)] hover:border-blue-200 hover:bg-blue-50"
      }`}
    >
      <span className="block max-w-[18rem] truncate font-[var(--font-mono)] text-xs font-semibold">
        {title}
      </span>
      <span className={`block text-xs ${active ? "text-[#dffdfa]" : "text-[var(--color-muted)]"}`}>
        {subtitle}
      </span>
    </button>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="min-w-0 rounded-2xl border border-[var(--color-line)] bg-white p-4 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <p className="m-0 text-sm text-[var(--color-muted)]">{label}</p>
      <p className="m-0 mt-1 font-[var(--font-mono)] text-[clamp(1.35rem,4vw,2rem)] font-bold tabular-nums">
        {value}
      </p>
    </article>
  );
}

function LlmThroughputChart({ entries }: { entries: LlmBenchmarkEntry[] }) {
  const top = entries.slice(0, 12);
  if (!top.length) return null;

  const labels = top.map((entry) =>
    entry.device.length > 26 ? entry.device.slice(0, 26) + "..." : entry.device,
  );

  return (
    <article className="min-w-0 rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <h2 className="m-0 font-[var(--font-display)] text-lg">LLM Throughput</h2>
      <p className="mt-1 mb-3 text-sm text-[var(--color-muted)]">
        Token generation and prompt processing for the canonical llama.cpp track
      </p>
      <div className="relative h-80 min-w-0 overflow-hidden">
        <Bar
          data={{
            labels,
            datasets: [
              {
                label: "TG tok/s",
                data: top.map((entry) => entry.tgTps ?? 0),
                yAxisID: "yTg",
                backgroundColor: "rgba(3, 105, 161, 0.72)",
                borderColor: "rgba(3, 105, 161, 1)",
                borderWidth: 1,
              },
              {
                label: "PP tok/s",
                data: top.map((entry) => entry.ppTps ?? 0),
                yAxisID: "yPp",
                backgroundColor: "rgba(15, 118, 110, 0.58)",
                borderColor: "rgba(15, 118, 110, 1)",
                borderWidth: 1,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { ticks: { maxRotation: 35, minRotation: 35 } },
              yTg: {
                beginAtZero: true,
                position: "left",
                title: { display: true, text: "TG tokens/s" },
                ticks: { color: "rgba(3, 105, 161, 1)" },
              },
              yPp: {
                beginAtZero: true,
                position: "right",
                title: { display: true, text: "PP tokens/s" },
                ticks: { color: "rgba(15, 118, 110, 1)" },
                grid: { drawOnChartArea: false },
              },
            },
            plugins: {
              legend: { position: "top" },
              tooltip: {
                callbacks: {
                  title(items) {
                    return top[items[0].dataIndex].device;
                  },
                },
              },
            },
          }}
        />
      </div>
    </article>
  );
}

function LlmResultTable({ entries }: { entries: LlmBenchmarkEntry[] }) {
  return (
    <section className="min-w-0 rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 font-[var(--font-display)] text-lg">LLM Inference Results</h2>
        <p className="m-0 font-[var(--font-mono)] text-sm tabular-nums text-[var(--color-muted)]">
          {entries.length.toLocaleString()} entries
        </p>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[1280px] border-collapse">
          <thead>
            <tr>
              {[
                "Device",
                "Case",
                "Status",
                "Runtime",
                "Backend",
                "Platform",
                "PP tok/s",
                "TG tok/s",
                "Prompt",
                "Gen",
                "Batch",
                "Profile",
                "Notes",
                "Date",
              ].map((heading) => (
                <th
                  key={heading}
                  className="sticky top-0 z-10 border-b border-[var(--color-line)] bg-[#f4f8fc] px-2.5 py-2.5 text-left text-xs uppercase tracking-wider text-[#334155]"
                >
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, index) => (
              <tr key={`${entry.device}-${entry.caseName}-${entry.runtime}-${index}`} className="hover:bg-[#f7fbff]">
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <div className="font-semibold">{formatLlmDeviceLabel(entry)}</div>
                  <small className="text-[var(--color-muted)]">
                    {toTitleCase(entry.vendor)} · {entry.memory || "N/A"}
                  </small>
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <div className="font-[var(--font-mono)] text-xs">{entry.caseName}</div>
                  <small className="text-[var(--color-muted)]">{entry.promptTokens.toLocaleString()}p / {entry.generationTokens.toLocaleString()}g</small>
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <span
                    className={`inline-block rounded-full px-2 py-0.5 text-xs font-bold ${
                      entry.status === "failed"
                        ? "bg-red-50 text-red-700"
                        : "bg-[var(--color-brand-soft)] text-[var(--color-brand-strong)]"
                    }`}
                  >
                    {entry.status.toUpperCase()}
                  </span>
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  <div>{entry.runtime}</div>
                  <small className="text-[var(--color-muted)]">{entry.runtimeVersion || "N/A"}</small>
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  {entry.accelerationBackend || "N/A"}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5">
                  {entry.platform || "N/A"}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] font-semibold tabular-nums text-[var(--color-brand)]">
                  {formatTps(entry.ppTps)}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] font-semibold tabular-nums text-[var(--color-accent)]">
                  {formatTps(entry.tgTps)}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] tabular-nums">
                  {entry.promptTokens.toLocaleString()}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] tabular-nums">
                  {entry.generationTokens.toLocaleString()}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] tabular-nums">
                  {entry.batchSize.toLocaleString()}
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 font-[var(--font-mono)] text-xs tabular-nums">
                  <div>ctx {entry.contextSize ? entry.contextSize.toLocaleString() : "N/A"}</div>
                  <small className="text-[var(--color-muted)]">
                    ub {entry.ubatchSize?.toLocaleString() ?? "N/A"} · KV {entry.cacheTypeK || "?"}/{entry.cacheTypeV || "?"} · FA {entry.flashAttention ? "on" : "off"}
                  </small>
                  <small className="block text-[var(--color-muted)]">
                    split {entry.splitMode || "N/A"}{entry.heterogeneousDevices ? " · heterogeneous" : ""}
                  </small>
                </td>
                <td className="border-b border-[var(--color-line)] px-2.5 py-2.5 italic text-[var(--color-muted)]">
                  {entry.status === "failed" ? entry.error || "Failed" : entry.note || "N/A"}
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
          No LLM inference entries are available.
        </p>
      )}
    </section>
  );
}

function formatLlmDeviceLabel(entry: LlmBenchmarkEntry): string {
  const count = entry.deviceIds?.length ?? 0;
  if (count <= 1 || entry.heterogeneousDevices) {
    return entry.device || "N/A";
  }

  const repeatedPrefix = `${count}x `;
  const baseDevice = entry.device.startsWith(repeatedPrefix)
    ? entry.device.slice(repeatedPrefix.length)
    : entry.device;
  return `${baseDevice} *${count}`;
}
