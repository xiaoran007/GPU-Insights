import { useState } from "react";
import type { Result } from "./types";
import { errorMessage } from "./client";

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
  const decode = summary.serverDecodeTps ?? summary.clientDecodeTps;

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
    <p className="wb-muted">{payload.config.provider} · {payload.config.endpointHost} · concurrency {payload.config.concurrency} · {new Date(payload.createdAt).toLocaleString()}</p>
    {payload.runStatus === "cancelled" && <p className="wb-notice">This run stopped early. Its payload preserves completed samples; run the full workload before collecting comparable results.</p>}
    <div className="wb-metrics">
      <div><span>TTFT · p50</span><strong>{metric(summary.ttftMs?.p50, "ms")}</strong></div>
      <div><span>Decode · p50</span><strong>{metric(decode?.p50, "tok/s")}</strong><small>{summary.serverDecodeTps ? "Server reported" : "Client estimate"}</small></div>
      <div><span>Prefill</span><strong>{metric(summary.serverPrefillTps?.p50 ?? summary.effectivePrefillTps, "tok/s")}</strong><small>{summary.serverPrefillTps ? "Server reported" : "TTFT slope estimate"}</small></div>
      <div><span>Measured samples</span><strong>{summary.successes} / {summary.successes + summary.failures}</strong><small>successful · warmup excluded</small></div>
    </div>
    {payload.config.concurrency > 1 && <p className="wb-muted">Request throughput: {metric(summary.requestThroughputRps, "requests/s")} · concurrent mode does not estimate prefill.</p>}
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
