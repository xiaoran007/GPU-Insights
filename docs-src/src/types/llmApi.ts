export interface Percentiles {
  p50: number;
  p95: number;
}

export interface LlmApiSummary {
  successes: number;
  failures: number;
  cacheStatus: "unknown" | "uncached" | "cached" | "mixed";
  ttftMs: Percentiles | null;
  totalMs: Percentiles | null;
  clientDecodeTps: Percentiles | null;
  serverDecodeTps: Percentiles | null;
  serverPrefillTps: Percentiles | null;
  effectivePrefillTps: number | null;
  prefillFitR2: number | null;
  requestThroughputRps: number | null;
  promptTokens: Percentiles | null;
  outputTokens: Percentiles | null;
}

export interface LlmApiEntry {
  runId: string;
  createdAt: string;
  provider: string;
  model: string;
  endpointHost: string;
  clientRegion: string;
  promptChars: number[];
  repetitions: number;
  concurrency: number;
  decodeOutputTokens: number;
  streamUsageRequested: boolean;
  summary: LlmApiSummary;
}

export interface LlmApiData {
  metadata: {
    lastUpdated: string | null;
    version: string;
    description: string;
  };
  benchmarks: LlmApiEntry[];
}
