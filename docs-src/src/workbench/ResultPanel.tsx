import { useState } from "react";
import type { Result } from "./types";
import { errorMessage } from "./client";
import { formatEstimateConfidence } from "../utils/format";

function metric(value: number | null | undefined, unit: string): string {
  return value == null ? "—" : `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${unit}`;
}

export default function ResultPanel({ result }: { result: Result | null }) {
  const [message, setMessage] = useState("");
  if (!result) return <section className="wb-card wb-empty">
    <span className="wb-eyebrow">Results & collection</span>
    <h2>Your next measurement starts here.</h2>
    <p>Run a benchmark or open a saved result. The complete collection payload will appear here, ready to copy or download.</p>
  </section>;
  const { payload, path } = result;
  const summary = payload.summary;
  const text = JSON.stringify(payload, null, 2) + "\n";
  const standard = payload.config.suiteId === "api-standard-v2";
  const prefill = standard ? summary.effectivePrefillTps : summary.serverPrefillTps?.p50 ?? summary.effectivePrefillTps;
  const decode = standard ? summary.decodeTps : (summary.serverDecodeTps ?? summary.clientDecodeTps)?.p50;
  const cacheDescription = {
    uncached: "No cache hits reported across measured requests.",
    cached: "Cache hits reported; token rates include cached work.",
    mixed: "Mixed cache hits and misses; token rates include cached work.",
    unknown: "Cache reporting incomplete; zero hits cannot be confirmed.",
  }[summary.cacheStatus];

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setMessage("Full payload copied.");
    } catch (error) {
      setMessage(`Copy failed: ${errorMessage(error)}. You can select the payload below or download JSON.`);
    }
  }

  function download() {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = path.split(/[\\/]/).pop() ?? "llm-api-result.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <section className="wb-card" aria-label="Benchmark result">
    <div className="wb-section-head">
      <div><span className="wb-eyebrow">Results & collection</span><h2>{payload.config.model}</h2></div>
      <span className={`wb-badge ${payload.runStatus === "cancelled" ? "wb-amber" : ""}`}>{payload.runStatus === "cancelled" ? "Partial · stopped" : "Completed"}</span>
    </div>
    <p className="wb-muted">{payload.config.suiteId ?? "Legacy suite"} · {payload.config.provider} · {payload.config.endpointHost} · {new Date(payload.createdAt).toLocaleString()}</p>
    <p className={summary.cacheStatus === "cached" || summary.cacheStatus === "mixed" ? "wb-notice" : "wb-muted"}>Prompt cache: {cacheDescription}</p>
    {payload.runStatus === "cancelled" && <p className="wb-notice">This run stopped early. Its payload preserves completed samples; run the full workload before collecting comparable results.</p>}
    <div className="wb-metrics">
      <div><span>TTFT · p50</span><strong>{metric(summary.ttftMs?.p50, "ms")}</strong></div>
      <div><span>{standard ? "Decode estimate" : "Legacy decode · p50"}</span><strong>{metric(decode, "tok/s")}</strong><small>{standard ? formatEstimateConfidence(summary.estimates?.decode) : "Legacy method"}</small></div>
      <div><span>Prefill</span><strong>{metric(prefill, "tok/s")}</strong><small>{standard ? formatEstimateConfidence(summary.estimates?.prefill) : "Legacy method"}</small></div>
      <div><span>Measured samples</span><strong>{summary.successes} / {summary.successes + summary.failures}</strong><small>successful · warmups excluded</small></div>
    </div>
    {standard && !summary.ttftMs && <p className="wb-notice">No visible text arrived in the decode scan, so TTFT is unavailable. Generated-token counts can include reasoning tokens reported by the service.</p>}
    {summary.estimates && <div className="wb-quality" aria-label="Estimate quality">
      {(["decode", "prefill"] as const).map((name) => {
        const evidence = summary.estimates![name];
        return <p key={name} className={evidence.status === "ok" ? "wb-muted" : "wb-notice"}>
          <strong>{name === "decode" ? "Decode" : "Prefill"}: {formatEstimateConfidence(evidence)}</strong>
          {` · ${evidence.samples} samples`}
          {evidence.groups != null && ` · ${evidence.groups} lengths`}
          {evidence.tokenSpan != null && ` · ${evidence.tokenSpan.toLocaleString()} token span`}
          {evidence.fitR2 != null && ` · R² ${evidence.fitR2.toFixed(3)}`}
          {evidence.relativeSlopeSpread != null && ` · slope spread ${(evidence.relativeSlopeSpread * 100).toFixed(1)}%`}
          {evidence.reason && <><br />{evidence.reason}</>}
        </p>;
      })}
      <p className="wb-muted">Token rates are service-level length/latency estimates. Chunk timing and server timing extensions do not determine them.</p>
      <p className="wb-muted">Confidence labels summarize fit, repeat variability, and sample coverage; they are not accuracy probabilities. Low confidence retains the estimate. A rate is unavailable only when the observed data cannot identify a positive token-cost slope.</p>
    </div>}
    <div className="wb-payload-head">
      <div><h3>Collection payload</h3><p>The complete result JSON, compatible with the existing import workflow.</p></div>
      <div className="wb-actions"><button className="wb-primary" onClick={() => void copy()}>Copy payload</button><button onClick={download}>Download JSON</button></div>
    </div>
    <p role="status" className="wb-feedback">{message}</p>
    <details><summary>Inspect full payload</summary><textarea className="wb-payload" aria-label="Collection payload JSON" readOnly value={text} spellCheck={false} /></details>
    <label className="wb-saved-path">Saved locally<input aria-label="Saved result path" readOnly value={path} /></label>
    <p className="wb-muted">Collect with <code>scripts/manage-data.py import-api-payload</code> and the saved JSON file. Publishing remains a separate action.</p>
  </section>;
}
