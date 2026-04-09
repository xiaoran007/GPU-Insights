import type { BenchmarkEntry, Filters } from "../types/benchmark";
import { useFilters } from "../hooks/useBenchmarkData";
import FilterControls from "./FilterControls";
import StatsGrid from "./StatsGrid";
import BenchmarkTable from "./BenchmarkTable";
import TopDevicesChart from "./TopDevicesChart";
import VendorChart from "./VendorChart";

interface ArchivePageProps {
  entries: BenchmarkEntry[];
}

const ARCHIVE_MODELS = ["resnet50", "cnn"];

export default function ArchivePage({ entries }: ArchivePageProps) {
  const archiveEntries = entries.filter((e) => ARCHIVE_MODELS.includes(e.model));
  const { filters, updateFilter } = useFilters();
  const filtered = useFilteredArchive(archiveEntries, filters);

  // Merge filter options across both archive models
  const allVendors = [...new Set(archiveEntries.map((e) => e.vendor))].sort();
  const allArchitectures = [...new Set(archiveEntries.map((e) => e.architecture))].sort();
  const allPlatforms = [...new Set(archiveEntries.map((e) => e.platform))].sort();

  return (
    <>
      <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5">
        <h2 className="m-0 font-[var(--font-display)] text-lg text-amber-900">
          📦 Legacy Archive
        </h2>
        <p className="mt-1 mb-0 text-sm text-amber-700">
          Historical ResNet50 and CNN benchmark results. These use the previous benchmark
          methodology and scoring system.
        </p>
      </section>

      <StatsGrid entries={filtered} />

      <FilterControls
        filters={filters}
        vendors={allVendors}
        architectures={allArchitectures}
        platforms={allPlatforms}
        showVersionFilter={true}
        onFilterChange={updateFilter}
      />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <TopDevicesChart entries={filtered} />
        <VendorChart entries={filtered} />
      </div>

      <BenchmarkTable entries={filtered} />
    </>
  );
}

function useFilteredArchive(entries: BenchmarkEntry[], filters: Filters) {
  const { version, vendor, architecture, platform, search, sort } = filters;

  let result = [...entries];

  if (version !== "all") {
    result = result.filter((e) => e.version === version);
  }
  if (vendor !== "all") {
    result = result.filter((e) => e.vendor === vendor);
  }
  if (architecture !== "all") {
    result = result.filter((e) => e.architecture === architecture);
  }
  if (platform !== "all") {
    result = result.filter((e) => e.platform === platform);
  }
  if (search) {
    const q = search.toLowerCase();
    result = result.filter((e) =>
      [e.device, e.platform, e.note, e.architecture, e.vendor, e.memory]
        .join(" ")
        .toLowerCase()
        .includes(q),
    );
  }

  // Sort
  const num = (v: number | null, fb: number) => (typeof v === "number" ? v : fb);
  switch (sort) {
    case "fp32-desc":
      result.sort((a, b) => num(b.fp32, -Infinity) - num(a.fp32, -Infinity));
      break;
    case "fp32-asc":
      result.sort((a, b) => num(a.fp32, Infinity) - num(b.fp32, Infinity));
      break;
    case "fp16-desc":
      result.sort((a, b) => num(b.fp16, -Infinity) - num(a.fp16, -Infinity));
      break;
    case "fp16-asc":
      result.sort((a, b) => num(a.fp16, Infinity) - num(b.fp16, Infinity));
      break;
    case "date-desc": {
      const pk = (d: string) => {
        const p = d.split(".").map(Number);
        return p.length === 3 ? p[0] * 10000 + p[1] * 100 + p[2] : 0;
      };
      result.sort((a, b) => pk(b.date) - pk(a.date));
      break;
    }
    case "date-asc": {
      const pk = (d: string) => {
        const p = d.split(".").map(Number);
        return p.length === 3 ? p[0] * 10000 + p[1] * 100 + p[2] : 0;
      };
      result.sort((a, b) => pk(a.date) - pk(b.date));
      break;
    }
    case "device":
      result.sort((a, b) => a.device.localeCompare(b.device));
      break;
  }

  return result;
}
