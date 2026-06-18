export interface LlmModelInfo {
  name: string;
  displayName: string;
  baseModel: string;
  artifact: string;
  parameters: string;
  task: string;
  defaultCommand: string;
}

export interface LlmBenchmarkEntry {
  model: string;
  caseName: string;
  caseDescription: string;
  status: "ok" | "failed";
  error: string;
  baseModel: string;
  artifact: string;
  runtime: string;
  runtimeVersion: string;
  accelerationBackend: string;
  vendor: string;
  architecture: string;
  device: string;
  memory: string;
  platform: string;
  promptTokens: number;
  generationTokens: number;
  batchSize: number;
  repetitions: number;
  ppTps: number | null;
  ppStddev: number | null;
  tgTps: number | null;
  tgStddev: number | null;
  nGpuLayers: number;
  threads: number | null;
  backendRaw: string;
  modelSizeBytes: number | null;
  modelParams: number | null;
  note: string;
  date: string;
  rawResult: unknown;
}

export interface LlmInferenceData {
  metadata: {
    lastUpdated: string;
    version: string;
    description: string;
  };
  models: Record<string, LlmModelInfo>;
  benchmarks: LlmBenchmarkEntry[];
}
