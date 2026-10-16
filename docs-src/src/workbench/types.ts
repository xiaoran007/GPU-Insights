import type { LlmApiSummary } from "../types/llmApi";

export interface Config {
  provider: string;
  model: string;
  base_url: string;
  region: string;
  include_usage: boolean;
  allow_insecure_http: boolean;
}

export interface Bootstrap {
  token: string;
  suite: {
    id: string;
    warmups: number;
    repetitions: number;
    inputSizes: number[];
    outputLimits: number[];
    requests: number;
    promptChars: number;
    outputTokens: number;
  };
  defaults: Config;
  providers: { id: string; baseUrl: string; keyConfigured: boolean }[];
  profiles: Record<string, Config>;
  outputDir: string;
}

export interface Sample {
  phase: string;
  status: string;
  error: string;
  promptChars: number;
  ttftMs: number | null;
  totalMs: number | null;
  clientDecodeTps: number | null;
}

export interface Job {
  id: string;
  status: "running" | "stopping" | "completed" | "cancelled" | "failed";
  error: string;
  completed: number;
  total: number;
  events: { completed: number; total: number; phase: string; sample: Sample | null }[];
  resultFile: string | null;
}

export interface Payload {
  schemaVersion: string;
  kind: "llm-api";
  createdAt: string;
  runStatus?: "completed" | "cancelled";
  config: {
    suiteId?: string;
    provider: string;
    model: string;
    endpointHost: string;
    concurrency: number;
    clientRegion: string;
  };
  summary: LlmApiSummary;
  samples: Sample[];
}

export interface Result {
  payload: Payload;
  path: string;
}

export interface HistoryEntry {
  file: string;
  createdAt: string;
  config: Payload["config"];
  summary: LlmApiSummary;
  status: string;
}

export interface Plan {
  requests: number;
  promptChars: number;
  outputTokens: number;
}
