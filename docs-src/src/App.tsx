import { useState } from "react";
import type { ModelKey } from "./types/benchmark";
import {
  useBenchmarkData,
  useFilteredEntries,
  useFilterOptions,
  useFilters,
} from "./hooks/useBenchmarkData";

import Layout from "./components/Layout";
import Hero from "./components/Hero";
import ModelTabs from "./components/ModelTabs";
import ModelProfile from "./components/ModelProfile";
import FilterControls from "./components/FilterControls";
import StatsGrid from "./components/StatsGrid";
import BenchmarkTable from "./components/BenchmarkTable";
import TopDevicesChart from "./components/TopDevicesChart";
import VendorChart from "./components/VendorChart";
import EmptyState from "./components/EmptyState";
import ArchivePage from "./components/ArchivePage";

type TabKey = ModelKey | "archive";

export default function App() {
  const { data, error, loading } = useBenchmarkData();
  const [activeTab, setActiveTab] = useState<TabKey>("vit");
  const { filters, updateFilter, resetFilters } = useFilters();

  // Current model key for data hooks (archive falls back to resnet50 for hook compat)
  const modelFilter = activeTab === "archive" ? "resnet50" : activeTab;
  const entries = data?.benchmarks ?? [];
  const filtered = useFilteredEntries(entries, modelFilter, filters);
  const { vendors, architectures, platforms } = useFilterOptions(entries, modelFilter);

  const modelInfo = activeTab !== "archive" ? data?.models[activeTab] : undefined;

  // Reset filters when switching tabs
  const handleTabChange = (tab: TabKey) => {
    setActiveTab(tab);
    resetFilters();
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex min-h-[60vh] items-center justify-center">
          <p className="animate-pulse text-[var(--color-muted)]">Loading benchmark data…</p>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3">
          <p className="text-lg font-semibold text-red-600">Failed to load data</p>
          <code className="rounded bg-red-50 px-3 py-1 text-sm text-red-700">{error}</code>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <Hero lastUpdated={data?.metadata.lastUpdated ?? "—"} />

      <div className="flex flex-col gap-4">
        <ModelTabs activeTab={activeTab} onTabChange={handleTabChange} />

        {activeTab === "archive" ? (
          <ArchivePage entries={entries} />
        ) : (
          <ModelTabContent
            modelInfo={modelInfo}
            filtered={filtered}
            filters={filters}
            vendors={vendors}
            architectures={architectures}
            platforms={platforms}
            onFilterChange={updateFilter}
          />
        )}
      </div>
    </Layout>
  );
}

/* ── Model tab content (ViT / UNet / DDPM) ── */

interface ModelTabContentProps {
  modelInfo?: import("./types/benchmark").ModelInfo;
  filtered: import("./types/benchmark").BenchmarkEntry[];
  filters: import("./types/benchmark").Filters;
  vendors: string[];
  architectures: string[];
  platforms: string[];
  onFilterChange: <K extends keyof import("./types/benchmark").Filters>(
    key: K,
    value: import("./types/benchmark").Filters[K],
  ) => void;
}

function ModelTabContent({
  modelInfo,
  filtered,
  filters,
  vendors,
  architectures,
  platforms,
  onFilterChange,
}: ModelTabContentProps) {
  if (!modelInfo) return null;

  // Empty state when no data exists for this model
  if (filtered.length === 0 && filters.vendor === "all" && !filters.search) {
    return (
      <>
        <ModelProfile model={modelInfo} />
        <EmptyState modelName={modelInfo.displayName} command={modelInfo.defaultCommand} />
      </>
    );
  }

  return (
    <>
      <ModelProfile model={modelInfo} />
      <StatsGrid entries={filtered} />

      <FilterControls
        filters={filters}
        vendors={vendors}
        architectures={architectures}
        platforms={platforms}
        showVersionFilter={false}
        onFilterChange={onFilterChange}
      />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <TopDevicesChart entries={filtered} />
        <VendorChart entries={filtered} />
      </div>

      <BenchmarkTable entries={filtered} />
    </>
  );
}
