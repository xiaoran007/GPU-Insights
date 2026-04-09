import { useEffect, useState, useMemo, useCallback } from "react";
import type {
  BenchmarkData,
  BenchmarkEntry,
  Filters,
  SortKey,
} from "../types/benchmark";

const DATA_URL = `${import.meta.env.BASE_URL}data/benchmark-data.json`;

function parseDateKey(value: string): number {
  const parts = value.split(".").map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) return 0;
  const [year, month, day] = parts;
  return year * 10000 + month * 100 + day;
}

function sortEntries(entries: BenchmarkEntry[], sort: SortKey): BenchmarkEntry[] {
  const sorted = [...entries];
  const num = (v: number | null, fallback: number) =>
    typeof v === "number" ? v : fallback;

  switch (sort) {
    case "fp32-desc":
      return sorted.sort((a, b) => num(b.fp32, -Infinity) - num(a.fp32, -Infinity));
    case "fp32-asc":
      return sorted.sort((a, b) => num(a.fp32, Infinity) - num(b.fp32, Infinity));
    case "fp16-desc":
      return sorted.sort((a, b) => num(b.fp16, -Infinity) - num(a.fp16, -Infinity));
    case "fp16-asc":
      return sorted.sort((a, b) => num(a.fp16, Infinity) - num(b.fp16, Infinity));
    case "date-desc":
      return sorted.sort((a, b) => parseDateKey(b.date) - parseDateKey(a.date));
    case "date-asc":
      return sorted.sort((a, b) => parseDateKey(a.date) - parseDateKey(b.date));
    case "device":
      return sorted.sort((a, b) => a.device.localeCompare(b.device));
    default:
      return sorted;
  }
}

export function useBenchmarkData() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(DATA_URL, { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((payload: BenchmarkData) => {
        setData(payload);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { data, error, loading };
}

export function useFilteredEntries(
  entries: BenchmarkEntry[],
  modelFilter: string,
  filters: Filters,
) {
  const filtered = useMemo(() => {
    let result = entries.filter((e) => e.model === modelFilter);

    if (filters.version !== "all") {
      result = result.filter((e) => e.version === filters.version);
    }
    if (filters.vendor !== "all") {
      result = result.filter((e) => e.vendor === filters.vendor);
    }
    if (filters.architecture !== "all") {
      result = result.filter((e) => e.architecture === filters.architecture);
    }
    if (filters.platform !== "all") {
      result = result.filter((e) => e.platform === filters.platform);
    }
    if (filters.search) {
      const q = filters.search.toLowerCase();
      result = result.filter((e) =>
        [e.device, e.platform, e.note, e.architecture, e.vendor, e.memory]
          .join(" ")
          .toLowerCase()
          .includes(q),
      );
    }

    return sortEntries(result, filters.sort);
  }, [entries, modelFilter, filters]);

  return filtered;
}

export function useFilterOptions(entries: BenchmarkEntry[], modelFilter: string) {
  return useMemo(() => {
    const modelEntries = entries.filter((e) => e.model === modelFilter);
    const vendors = [...new Set(modelEntries.map((e) => e.vendor))].sort();
    const architectures = [...new Set(modelEntries.map((e) => e.architecture))].sort();
    const platforms = [...new Set(modelEntries.map((e) => e.platform))].sort();
    return { vendors, architectures, platforms };
  }, [entries, modelFilter]);
}

const DEFAULT_FILTERS: Filters = {
  vendor: "all",
  architecture: "all",
  platform: "all",
  search: "",
  sort: "fp32-desc",
  version: "all",
};

export function useFilters() {
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);

  const updateFilter = useCallback(
    <K extends keyof Filters>(key: K, value: Filters[K]) => {
      setFilters((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const resetFilters = useCallback(() => setFilters(DEFAULT_FILTERS), []);

  return { filters, updateFilter, resetFilters };
}
