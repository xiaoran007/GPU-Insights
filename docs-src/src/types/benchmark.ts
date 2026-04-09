export interface ModelInfo {
  name: string;
  displayName: string;
  parameters: string;
  inputSize: string;
  task: string;
  defaultCommand: string;
}

export interface BenchmarkEntry {
  model: string;
  vendor: string;
  architecture: string;
  device: string;
  memory: string;
  platform: string;
  fp32: number | null;
  fp32bs: number;
  fp16: number | null;
  fp16bs: number;
  note: string;
  date: string;
  version: string;
}

export interface BenchmarkMetadata {
  lastUpdated: string;
  version: string;
  description: string;
}

export interface BenchmarkData {
  metadata: BenchmarkMetadata;
  models: Record<string, ModelInfo>;
  benchmarks: BenchmarkEntry[];
}

export type ModelKey = "vit" | "unet" | "ddpm";
export type ArchiveModelKey = "resnet50" | "cnn";
export type SortKey =
  | "fp32-desc"
  | "fp32-asc"
  | "fp16-desc"
  | "fp16-asc"
  | "date-desc"
  | "date-asc"
  | "device";

export interface Filters {
  vendor: string;
  architecture: string;
  platform: string;
  search: string;
  sort: SortKey;
  version: string;
}
