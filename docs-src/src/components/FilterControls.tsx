import type { Filters, SortKey } from "../types/benchmark";
import { toTitleCase } from "../utils/format";

interface FilterControlsProps {
  filters: Filters;
  vendors: string[];
  architectures: string[];
  platforms: string[];
  showVersionFilter?: boolean;
  onFilterChange: <K extends keyof Filters>(key: K, value: Filters[K]) => void;
}

export default function FilterControls({
  filters,
  vendors,
  architectures,
  platforms,
  showVersionFilter = false,
  onFilterChange,
}: FilterControlsProps) {
  return (
    <section className="grid gap-4 rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 font-[var(--font-display)] text-lg">Filters</h2>
      </div>

      {showVersionFilter && (
        <div className="flex flex-wrap gap-2">
          {["all", "ver2", "ver1"].map((v) => (
            <button
              key={v}
              onClick={() => onFilterChange("version", v)}
              className={`cursor-pointer rounded-full border px-3.5 py-1.5 text-sm font-semibold transition-colors ${
                filters.version === v
                  ? "border-[var(--color-brand)] bg-[var(--color-brand)] text-[#ebfffd]"
                  : "border-[var(--color-line)] bg-[var(--color-surface-soft)] hover:border-blue-200 hover:bg-blue-50"
              }`}
            >
              {v === "all" ? "All Versions" : v.toUpperCase()}
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <SelectField
          label="Vendor"
          value={filters.vendor}
          options={vendors}
          formatter={toTitleCase}
          onChange={(v) => onFilterChange("vendor", v)}
        />
        <SelectField
          label="Architecture"
          value={filters.architecture}
          options={architectures}
          onChange={(v) => onFilterChange("architecture", v)}
        />
        <SelectField
          label="Platform"
          value={filters.platform}
          options={platforms}
          onChange={(v) => onFilterChange("platform", v)}
        />
        <SelectField
          label="Sort By"
          value={filters.sort}
          options={[
            "fp32-desc",
            "fp32-asc",
            "fp16-desc",
            "fp16-asc",
            "date-desc",
            "date-asc",
            "device",
          ]}
          formatter={sortLabel}
          onChange={(v) => onFilterChange("sort", v as SortKey)}
          allOption={false}
        />
      </div>

      <div className="flex flex-col gap-1 text-sm text-[var(--color-muted)]">
        <span>Search</span>
        <input
          type="search"
          placeholder="Search device, note, platform..."
          value={filters.search}
          onChange={(e) => onFilterChange("search", e.target.value.trim().toLowerCase())}
          className="rounded-lg border border-[var(--color-line)] bg-[#fbfdff] px-2.5 py-2 text-sm"
        />
      </div>
    </section>
  );
}

function sortLabel(key: string): string {
  const labels: Record<string, string> = {
    "fp32-desc": "FP32 Score (High → Low)",
    "fp32-asc": "FP32 Score (Low → High)",
    "fp16-desc": "FP16 Score (High → Low)",
    "fp16-asc": "FP16 Score (Low → High)",
    "date-desc": "Date (Newest)",
    "date-asc": "Date (Oldest)",
    device: "Device Name",
  };
  return labels[key] ?? key;
}

function SelectField({
  label,
  value,
  options,
  formatter = (v: string) => v,
  onChange,
  allOption = true,
}: {
  label: string;
  value: string;
  options: string[];
  formatter?: (v: string) => string;
  onChange: (v: string) => void;
  allOption?: boolean;
}) {
  return (
    <label className="flex flex-col gap-1 text-sm text-[var(--color-muted)]">
      <span>{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-[var(--color-line)] bg-[#fbfdff] px-2 py-2 text-sm text-[#0f172a]"
      >
        {allOption && <option value="all">All {label}s</option>}
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {formatter(opt)}
          </option>
        ))}
      </select>
    </label>
  );
}
