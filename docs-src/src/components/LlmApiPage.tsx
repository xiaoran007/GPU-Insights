import {
  BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Tooltip,
} from "chart.js";
import { useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import type { LlmApiData, LlmApiEntry, Percentiles } from "../types/llmApi";

ChartJS.register(BarElement, CategoryScale, Legend, LinearScale, Tooltip);

const cardClass = "min-w-0 rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]";

function number(value: number | null | undefined, digits = 1) {
  return value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function p50(value: Percentiles | null, unit: string) {
  return value ? `${number(value.p50)} ${unit}` : "—";
}

function decode(entry: LlmApiEntry) {
  return entry.summary.serverDecodeTps ?? entry.summary.clientDecodeTps;
}

function prefill(entry: LlmApiEntry) {
  return entry.summary.serverPrefillTps?.p50 ?? entry.summary.effectivePrefillTps;
}

export default function LlmApiPage({ data }: { data: LlmApiData }) {
  const [provider, setProvider] = useState("all");
  const providers = useMemo(() => [...new Set(data.benchmarks.map((entry) => entry.provider))].sort(), [data.benchmarks]);
  const entries = data.benchmarks
    .filter((entry) => provider === "all" || entry.provider === provider)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  const ttftValues = entries.map((entry) => entry.summary.ttftMs?.p50).filter((value): value is number => value != null);
  const decodeValues = entries.map((entry) => decode(entry)?.p50).filter((value): value is number => value != null);

  return (
    <div className="grid min-w-0 gap-4">
      <section className={`${cardClass} overflow-hidden bg-gradient-to-br from-white via-[#f3fbfa] to-[#eaf2fc]`}>
        <p className="m-0 text-xs font-bold uppercase tracking-[0.2em] text-[var(--color-brand)]">API benchmark track</p>
        <h2 className="mt-2 mb-2 font-[var(--font-display)] text-2xl font-bold">Measure the service you actually call</h2>
        <p className="m-0 max-w-3xl text-[var(--color-muted)]">
          End-to-end latency includes network and queue time. Engine-reported throughput and client estimates are labeled separately.
          The benchmark runs locally; only reviewed summaries are published here.
        </p>
      </section>

      <section className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Published runs" value={number(entries.length, 0)} />
        <Metric label="Providers" value={number(new Set(entries.map((entry) => entry.provider)).size, 0)} />
        <Metric label="Best median TTFT" value={ttftValues.length ? `${number(Math.min(...ttftValues))} ms` : "—"} />
        <Metric label="Top median decode" value={decodeValues.length ? `${number(Math.max(...decodeValues))} tok/s` : "—"} />
      </section>

      <section className={cardClass}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="m-0 font-[var(--font-display)] text-lg font-semibold">Results</h2>
            <p className="m-0 mt-1 text-sm text-[var(--color-muted)]">Compare runs with similar input size, output size, region, and concurrency.</p>
          </div>
          <label className="text-sm font-semibold text-[var(--color-muted)]">
            Provider
            <select className="ml-2 rounded-lg border border-[var(--color-line)] bg-white px-3 py-2 text-[var(--color-text)]" value={provider} onChange={(event) => setProvider(event.target.value)}>
              <option value="all">All</option>
              {providers.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
        </div>
        {entries.length ? <ResultsTable entries={entries} /> : <EmptyResults />}
      </section>

      {entries.length > 0 && <LatencyChart entries={entries.slice(0, 12)} />}

      <section className={`${cardClass} grid gap-4 md:grid-cols-3`}>
        <Method title="Measured" text="TTFT starts when the client sends a request and ends at the first visible text. Total latency ends when the response stream completes." />
        <Method title="Server reported" text="When vLLM or llama.cpp provides per-request timings, its prefill and decode values are shown as server metrics." />
        <Method title="Estimated" text="Without server timings, prefill uses the slope of TTFT across three input sizes. Decode uses output token usage and the visible stream interval. Missing or weak evidence stays blank." />
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <article className={cardClass}>
    <p className="m-0 text-sm text-[var(--color-muted)]">{label}</p>
    <p className="m-0 mt-2 font-[var(--font-mono)] text-2xl font-bold tabular-nums">{value}</p>
  </article>;
}

function EmptyResults() {
  return <div className="mt-5 rounded-xl border border-dashed border-[var(--color-line)] bg-[var(--color-surface-soft)] p-5">
    <p className="m-0 font-semibold">No API results published yet</p>
    <p className="mt-1 text-sm text-[var(--color-muted)]">Run the benchmark locally, inspect its JSON result, then import a summary into this page.</p>
    <code className="mt-3 block overflow-x-auto rounded-lg bg-white px-3 py-2 text-xs">python -m llm_bench.api.cli --provider openai-chat --model YOUR_MODEL --dry-run</code>
  </div>;
}

function ResultsTable({ entries }: { entries: LlmApiEntry[] }) {
  return <div className="mt-4 overflow-x-auto">
    <table className="w-full min-w-[1440px] border-collapse text-left text-sm">
      <thead className="border-b border-[var(--color-line)] text-xs uppercase tracking-wider text-[var(--color-muted)]">
        <tr>{["Provider / Model", "Endpoint", "Concurrency", "TTFT p50 / p95", "Total p50", "Decode p50", "Prefill", "Requests/s", "Samples", "Input / output", "Date"].map((heading) => <th key={heading} className="px-3 py-3">{heading}</th>)}</tr>
      </thead>
      <tbody>
        {entries.map((entry) => <tr key={entry.runId} className="border-b border-[var(--color-line)] align-top last:border-0">
          <td className="px-3 py-3"><span className="block font-semibold">{entry.model}</span><span className="text-xs text-[var(--color-muted)]">{entry.provider}</span></td>
          <td className="px-3 py-3 font-[var(--font-mono)] text-xs">{entry.endpointHost}</td>
          <td className="px-3 py-3 font-[var(--font-mono)]">{entry.concurrency}</td>
          <td className="px-3 py-3 font-[var(--font-mono)] tabular-nums">{entry.summary.ttftMs ? `${number(entry.summary.ttftMs.p50)} / ${number(entry.summary.ttftMs.p95)} ms` : "—"}</td>
          <td className="px-3 py-3 font-[var(--font-mono)] tabular-nums">{p50(entry.summary.totalMs, "ms")}</td>
          <td className="px-3 py-3 font-[var(--font-mono)] tabular-nums">{p50(decode(entry), "tok/s")}<span className="block text-xs text-[var(--color-muted)]">{entry.summary.serverDecodeTps ? "server" : entry.summary.clientDecodeTps ? "client estimate" : ""}</span></td>
          <td className="px-3 py-3 font-[var(--font-mono)] tabular-nums">{prefill(entry) != null ? `${number(prefill(entry))} tok/s` : "—"}<span className="block text-xs text-[var(--color-muted)]">{entry.summary.serverPrefillTps ? "server" : entry.summary.effectivePrefillTps ? `TTFT slope · R² ${number(entry.summary.prefillFitR2, 2)}` : ""}</span></td>
          <td className="px-3 py-3 font-[var(--font-mono)] tabular-nums">{number(entry.summary.requestThroughputRps)}</td>
          <td className="px-3 py-3">{entry.summary.successes} ok<span className="block text-xs text-[var(--color-muted)]">{entry.summary.failures} failed</span></td>
          <td className="px-3 py-3 font-[var(--font-mono)] tabular-nums">{number(entry.summary.promptTokens?.p50, 0)} / {number(entry.summary.outputTokens?.p50, 0)}<span className="block text-xs text-[var(--color-muted)]">median tokens</span></td>
          <td className="px-3 py-3 whitespace-nowrap">{entry.createdAt.slice(0, 10)}</td>
        </tr>)}
      </tbody>
    </table>
  </div>;
}

function LatencyChart({ entries }: { entries: LlmApiEntry[] }) {
  const rows = entries.filter((entry) => entry.summary.ttftMs && entry.summary.totalMs);
  if (!rows.length) return null;
  return <section className={cardClass}>
    <h2 className="m-0 font-[var(--font-display)] text-lg font-semibold">Median client latency</h2>
    <p className="mt-1 mb-4 text-sm text-[var(--color-muted)]">Most recently published runs · milliseconds</p>
    <div className="h-80 min-w-0">
      <Bar data={{
        labels: rows.map((entry) => `${entry.provider}: ${entry.model}`),
        datasets: [
          { label: "TTFT", data: rows.map((entry) => entry.summary.ttftMs?.p50 ?? 0), backgroundColor: "rgba(15, 118, 110, 0.78)" },
          { label: "Total", data: rows.map((entry) => entry.summary.totalMs?.p50 ?? 0), backgroundColor: "rgba(3, 105, 161, 0.62)" },
        ],
      }} options={{ responsive: true, maintainAspectRatio: false, indexAxis: "y", scales: { x: { beginAtZero: true } } }} />
    </div>
  </section>;
}

function Method({ title, text }: { title: string; text: string }) {
  return <div>
    <h3 className="m-0 text-sm font-bold uppercase tracking-wide text-[var(--color-brand)]">{title}</h3>
    <p className="m-0 mt-2 text-sm leading-relaxed text-[var(--color-muted)]">{text}</p>
  </div>;
}
