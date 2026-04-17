import { useEffect, useState } from "react";
import type {
  NvidiaCoverageData,
  NvidiaEvidence,
  NvidiaGpuSpecRecord,
  NvidiaSpecsData,
  NvidiaTensorThroughputEntry,
} from "../types/nvidiaSpecs";

const GENERATIONS = ["all", "Volta", "Turing", "Ampere", "Ada", "Hopper", "Blackwell"] as const;
const RECORD_TYPES = ["all", "architecture", "sku"] as const;

interface NvidiaSpecsPageProps {
  data: NvidiaSpecsData;
  coverage: NvidiaCoverageData | null;
}

export default function NvidiaSpecsPage({ data, coverage }: NvidiaSpecsPageProps) {
  const [generationFilter, setGenerationFilter] =
    useState<(typeof GENERATIONS)[number]>("all");
  const [recordTypeFilter, setRecordTypeFilter] =
    useState<(typeof RECORD_TYPES)[number]>("all");
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(data.records[0]?.record_id ?? null);

  const visibleRecords = data.records.filter((record) => {
    if (generationFilter !== "all" && record.generation !== generationFilter) {
      return false;
    }
    if (recordTypeFilter !== "all" && record.record_type !== recordTypeFilter) {
      return false;
    }
    return true;
  });

  useEffect(() => {
    if (!visibleRecords.length) {
      setSelectedRecordId(null);
      return;
    }
    if (!visibleRecords.some((record) => record.record_id === selectedRecordId)) {
      setSelectedRecordId(visibleRecords[0].record_id);
    }
  }, [selectedRecordId, visibleRecords]);

  const selectedRecord =
    visibleRecords.find((record) => record.record_id === selectedRecordId) ?? visibleRecords[0] ?? null;

  return (
    <div className="grid gap-4">
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Records" value={String(data.records.length)} hint="Architecture + SKU" />
        <StatCard
          label="Throughput Rows"
          value={String(coverage?.summary.recordsWithThroughput ?? countThroughputRecords(data.records))}
          hint="Official tensor throughput"
        />
        <StatCard
          label="Official Sources"
          value={String(data.metadata.officialSourceDomains.length)}
          hint={data.metadata.officialSourceDomains.join(", ")}
        />
        <StatCard
          label="Generated"
          value={formatShortDate(data.metadata.generatedAt)}
          hint="Specs refresh timestamp"
        />
      </section>

      <section className="rounded-2xl border border-[var(--color-line)] bg-white p-4 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="m-0 text-xs uppercase tracking-[0.22em] text-[var(--color-muted)]">
              NVIDIA Official Reference
            </p>
            <h2 className="m-0 mt-1 font-[var(--font-display)] text-2xl text-slate-950">
              AI-oriented architecture and SKU specs
            </h2>
            <p className="m-0 mt-2 max-w-3xl text-sm text-[var(--color-muted)]">
              This page only surfaces NVIDIA official architecture pages, whitepapers, datasheets,
              and compute capability listings. Missing fields stay null and are called out in the
              coverage summary below.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <FilterGroup
              label="Generation"
              options={GENERATIONS}
              active={generationFilter}
              onChange={(value) => setGenerationFilter(value as (typeof GENERATIONS)[number])}
            />
            <FilterGroup
              label="Record Type"
              options={RECORD_TYPES}
              active={recordTypeFilter}
              onChange={(value) => setRecordTypeFilter(value as (typeof RECORD_TYPES)[number])}
            />
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(340px,0.95fr)]">
        <section className="overflow-hidden rounded-2xl border border-[var(--color-line)] bg-white shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
          <div className="border-b border-[var(--color-line)] px-4 py-3">
            <h3 className="m-0 font-[var(--font-display)] text-lg text-slate-950">
              Spec Records
            </h3>
            <p className="m-0 mt-1 text-sm text-[var(--color-muted)]">
              Sorted by generation, record type, and product name.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse text-sm">
              <thead className="bg-[var(--color-surface-soft)] text-left text-[var(--color-muted)]">
                <tr>
                  <th className="px-4 py-3 font-semibold">Generation</th>
                  <th className="px-4 py-3 font-semibold">Type</th>
                  <th className="px-4 py-3 font-semibold">Product</th>
                  <th className="px-4 py-3 font-semibold">CC</th>
                  <th className="px-4 py-3 font-semibold">Die</th>
                  <th className="px-4 py-3 font-semibold">SM</th>
                  <th className="px-4 py-3 font-semibold">Tensor</th>
                  <th className="px-4 py-3 font-semibold">RT</th>
                </tr>
              </thead>
              <tbody>
                {visibleRecords.map((record) => {
                  const active = record.record_id === selectedRecord?.record_id;
                  return (
                    <tr
                      key={record.record_id}
                      className={`cursor-pointer border-t border-[var(--color-line)] transition-colors ${
                        active ? "bg-cyan-50" : "hover:bg-slate-50"
                      }`}
                      onClick={() => setSelectedRecordId(record.record_id)}
                    >
                      <td className="px-4 py-3 align-top font-medium text-slate-900">
                        {record.generation}
                      </td>
                      <td className="px-4 py-3 align-top">
                        <span
                          className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${
                            record.record_type === "architecture"
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-sky-100 text-sky-800"
                          }`}
                        >
                          {record.record_type}
                        </span>
                      </td>
                      <td className="px-4 py-3 align-top">
                        <p className="m-0 font-semibold text-slate-900">{record.product_name}</p>
                        <p className="m-0 mt-1 text-xs text-[var(--color-muted)]">
                          {formatThroughputPreview(record.official_tensor_throughput)}
                        </p>
                      </td>
                      <td className="px-4 py-3 align-top font-[var(--font-mono)]">
                        {formatNullable(record.compute_capability)}
                      </td>
                      <td className="px-4 py-3 align-top">{formatNullable(record.die_family)}</td>
                      <td className="px-4 py-3 align-top">{formatNullable(record.sm_count)}</td>
                      <td className="px-4 py-3 align-top">
                        {formatNullable(record.tensor_core_count)}
                      </td>
                      <td className="px-4 py-3 align-top">{formatNullable(record.rt_core_count)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="grid gap-4">
          <section className="rounded-2xl border border-[var(--color-line)] bg-white p-4 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
            <h3 className="m-0 font-[var(--font-display)] text-lg text-slate-950">
              Record Details
            </h3>
            {selectedRecord ? (
              <RecordDetails record={selectedRecord} />
            ) : (
              <p className="m-0 mt-3 text-sm text-[var(--color-muted)]">No record matches the current filters.</p>
            )}
          </section>

          {coverage ? <CoveragePanel coverage={coverage} /> : null}
        </aside>
      </div>
    </div>
  );
}

function RecordDetails({ record }: { record: NvidiaGpuSpecRecord }) {
  return (
    <div className="mt-3 grid gap-4">
      <div>
        <p className="m-0 text-xs uppercase tracking-[0.2em] text-[var(--color-muted)]">
          {record.generation} {record.record_type}
        </p>
        <h4 className="m-0 mt-1 font-[var(--font-display)] text-xl text-slate-950">
          {record.product_name}
        </h4>
        <p className="m-0 mt-2 text-sm text-[var(--color-muted)]">
          {record.notes[0] ?? "Evidence-backed fields only; unknown values stay null."}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <SpecItem label="Architecture" value={record.architecture_codename} />
        <SpecItem label="Die Family" value={record.die_family} />
        <SpecItem label="Compute Capability" value={record.compute_capability} />
        <SpecItem label="Tensor Core Gen" value={record.tensor_core_generation} />
        <SpecItem label="GPC / TPC" value={joinNullable([record.gpc_count, record.tpc_count], " / ")} />
        <SpecItem label="SM / Tensor / RT" value={joinNullable([record.sm_count, record.tensor_core_count, record.rt_core_count], " / ")} />
      </div>

      <section className="rounded-2xl bg-[var(--color-surface-soft)] p-3">
        <p className="m-0 text-sm font-semibold text-slate-900">Supported Tensor Data Types</p>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {Object.entries(record.tensor_datatypes).map(([dtype, supported]) => (
            <div key={dtype} className="rounded-xl border border-[var(--color-line)] bg-white px-3 py-2">
              <p className="m-0 font-[var(--font-mono)] text-xs uppercase text-[var(--color-muted)]">
                {dtype}
              </p>
              <p className="m-0 mt-1 font-semibold text-slate-900">
                {supported === true ? "Yes" : "—"}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <p className="m-0 text-sm font-semibold text-slate-900">Official Tensor Throughput</p>
        {record.official_tensor_throughput ? (
          <div className="mt-2 grid gap-2">
            {record.official_tensor_throughput.map((entry) => (
              <div
                key={`${record.record_id}-${entry.label}-${entry.source_id}`}
                className="rounded-xl border border-[var(--color-line)] bg-[var(--color-surface-soft)] px-3 py-2"
              >
                <p className="m-0 text-sm font-semibold text-slate-900">
                  {entry.label}: {entry.value} {entry.unit}
                </p>
                <p className="m-0 mt-1 text-xs text-[var(--color-muted)]">
                  {joinText([entry.dtype?.toUpperCase() ?? null, entry.sparsity], " · ")}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="m-0 mt-2 text-sm text-[var(--color-muted)]">—</p>
        )}
      </section>

      <section>
        <p className="m-0 text-sm font-semibold text-slate-900">Official Sources</p>
        <div className="mt-2 flex flex-col gap-2">
          {record.source_urls.map((url) => (
            <a
              key={url}
              href={url}
              target="_blank"
              rel="noreferrer"
              className="rounded-xl border border-[var(--color-line)] bg-[var(--color-surface-soft)] px-3 py-2 text-sm text-sky-700 hover:text-sky-900"
            >
              {url}
            </a>
          ))}
        </div>
      </section>

      <section>
        <p className="m-0 text-sm font-semibold text-slate-900">Evidence Excerpts</p>
        <div className="mt-2 grid gap-2">
          {flattenEvidence(record.field_evidence).map((item) => (
            <div
              key={`${record.record_id}-${item.field}-${item.evidence.source_id}-${item.evidence.locator}`}
              className="rounded-xl border border-[var(--color-line)] bg-white px-3 py-3"
            >
              <p className="m-0 text-xs font-semibold uppercase tracking-[0.18em] text-[var(--color-muted)]">
                {item.field}
              </p>
              <p className="m-0 mt-1 text-sm font-semibold text-slate-900">
                {item.evidence.title}
              </p>
              <p className="m-0 mt-1 text-xs text-[var(--color-muted)]">{item.evidence.locator}</p>
              <p className="m-0 mt-2 text-sm leading-6 text-slate-700">{item.evidence.excerpt}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3">
        <p className="m-0 text-sm font-semibold text-amber-900">Missing Fields</p>
        <p className="m-0 mt-2 text-sm text-amber-800">{record.missing_fields.join(", ") || "None"}</p>
      </section>
    </div>
  );
}

function CoveragePanel({ coverage }: { coverage: NvidiaCoverageData }) {
  const weakestFields = Object.entries(coverage.coverageByField)
    .sort((left, right) => left[1].ratio - right[1].ratio)
    .slice(0, 8);

  return (
    <section className="rounded-2xl border border-[var(--color-line)] bg-white p-4 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <h3 className="m-0 font-[var(--font-display)] text-lg text-slate-950">Coverage Summary</h3>
      <p className="m-0 mt-1 text-sm text-[var(--color-muted)]">
        Generated {formatLongDate(coverage.generatedAt)}. Lower ratios reflect NVIDIA fields that are
        rarely published for a given SKU class.
      </p>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <StatCard label="Architecture" value={String(coverage.summary.architectureRecords)} hint="Architecture records" />
        <StatCard label="SKUs" value={String(coverage.summary.skuRecords)} hint="Representative products" />
        <StatCard
          label="Throughput"
          value={String(coverage.summary.recordsWithThroughput)}
          hint="Records with official tensor throughput"
        />
        <StatCard
          label="Supplemental"
          value={String(coverage.summary.recordsUsingSupplementalSources)}
          hint="Should stay at zero for official-only mode"
        />
      </div>

      <div className="mt-4 overflow-hidden rounded-2xl border border-[var(--color-line)]">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-[var(--color-surface-soft)] text-left text-[var(--color-muted)]">
            <tr>
              <th className="px-3 py-2 font-semibold">Field</th>
              <th className="px-3 py-2 font-semibold">Coverage</th>
              <th className="px-3 py-2 font-semibold">Ratio</th>
            </tr>
          </thead>
          <tbody>
            {weakestFields.map(([fieldName, field]) => (
              <tr key={fieldName} className="border-t border-[var(--color-line)]">
                <td className="px-3 py-2 font-[var(--font-mono)] text-xs text-slate-800">
                  {fieldName}
                </td>
                <td className="px-3 py-2 text-slate-700">
                  {field.covered}/{field.total}
                </td>
                <td className="px-3 py-2 text-slate-700">{Math.round(field.ratio * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function FilterGroup({
  label,
  options,
  active,
  onChange,
}: {
  label: string;
  options: readonly string[];
  active: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--color-muted)]">
        {label}
      </span>
      {options.map((option) => {
        const activeOption = option === active;
        return (
          <button
            key={option}
            onClick={() => onChange(option)}
            className={`cursor-pointer rounded-full border px-3 py-1.5 text-xs font-semibold transition-colors ${
              activeOption
                ? "border-[var(--color-accent)] bg-[var(--color-accent)] text-white"
                : "border-[var(--color-line)] bg-[var(--color-surface-soft)] text-slate-800 hover:bg-sky-50"
            }`}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}

function StatCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <article className="rounded-2xl border border-[var(--color-line)] bg-white p-4 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <p className="m-0 text-sm text-[var(--color-muted)]">{label}</p>
      <p className="m-0 mt-1 font-[var(--font-mono)] text-[clamp(1.3rem,3vw,1.8rem)] font-bold text-slate-950">
        {value}
      </p>
      <p className="m-0 mt-1 text-xs leading-5 text-[var(--color-muted)]">{hint}</p>
    </article>
  );
}

function SpecItem({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-surface-soft)] px-3 py-2">
      <p className="m-0 text-xs uppercase tracking-[0.18em] text-[var(--color-muted)]">{label}</p>
      <p className="m-0 mt-1 font-semibold text-slate-900">{formatNullable(value)}</p>
    </div>
  );
}

function flattenEvidence(fieldEvidence: Record<string, NvidiaEvidence[]>) {
  return Object.entries(fieldEvidence)
    .flatMap(([field, evidenceList]) => evidenceList.map((evidence) => ({ field, evidence })))
    .sort((left, right) => left.field.localeCompare(right.field));
}

function formatNullable(value: number | string | null) {
  if (value === null || value === "") return "—";
  return String(value);
}

function joinNullable(values: Array<number | string | null>, separator: string) {
  const filtered = values.filter((value) => value !== null && value !== "");
  return filtered.length ? filtered.map(String).join(separator) : null;
}

function joinText(values: Array<string | null>, separator: string) {
  const filtered = values.filter(Boolean);
  return filtered.length ? filtered.join(separator) : "Official label only";
}

function formatShortDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(value));
}

function formatLongDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function countThroughputRecords(records: NvidiaGpuSpecRecord[]) {
  return records.filter((record) => record.official_tensor_throughput?.length).length;
}

function formatThroughputPreview(entries: NvidiaTensorThroughputEntry[] | null) {
  if (!entries?.length) return "No official tensor throughput extracted";
  return entries
    .slice(0, 2)
    .map((entry) => `${entry.label}: ${entry.value} ${entry.unit}`)
    .join(" · ");
}
