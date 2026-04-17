export interface NvidiaEvidence {
  source_id: string;
  title: string;
  url: string;
  locator: string;
  excerpt: string;
}

export interface NvidiaTensorThroughputEntry {
  label: string;
  dtype: string | null;
  value: number | string;
  unit: string;
  sparsity: string | null;
  source_id: string;
}

export interface NvidiaGpuSpecRecord {
  record_id: string;
  record_type: "architecture" | "sku";
  generation: string;
  product_name: string;
  architecture_codename: string | null;
  die_family: string | null;
  compute_capability: string | null;
  sm_count: number | null;
  tensor_core_count: number | null;
  rt_core_count: number | null;
  gpc_count: number | null;
  tpc_count: number | null;
  sm_per_tpc: number | null;
  tpc_per_gpc: number | null;
  enabled_units_summary: string | null;
  tensor_core_generation: string | null;
  tensor_datatypes: Record<string, boolean | null>;
  official_tensor_throughput: NvidiaTensorThroughputEntry[] | null;
  source_urls: string[];
  field_evidence: Record<string, NvidiaEvidence[]>;
  supplemental_sources: string[];
  notes: string[];
  missing_fields: string[];
}

export interface NvidiaSpecsMetadata {
  generatedAt: string;
  description: string;
  officialSourceDomains: string[];
  nullPolicy: string;
}

export interface NvidiaSpecsData {
  metadata: NvidiaSpecsMetadata;
  records: NvidiaGpuSpecRecord[];
}

export interface NvidiaCoverageField {
  covered: number;
  total: number;
  ratio: number;
}

export interface NvidiaCoverageSummary {
  recordsWithThroughput: number;
  recordsUsingSupplementalSources: number;
  architectureRecords: number;
  skuRecords: number;
}

export interface NvidiaCoverageData {
  generatedAt: string;
  totalRecords: number;
  coverageByField: Record<string, NvidiaCoverageField>;
  summary: NvidiaCoverageSummary;
}
