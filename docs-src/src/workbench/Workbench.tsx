import { useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { api, errorMessage } from "./client";
import ResultPanel from "./ResultPanel";
import type { Bootstrap, Config, Job } from "./types";
import { useWorkbench } from "./useWorkbench";

export default function WorkbenchRoot() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    api<Bootstrap>("bootstrap", "", undefined, controller.signal)
      .then(setBootstrap)
      .catch((error) => { if (!controller.signal.aborted) setError(errorMessage(error)); });
    return () => controller.abort();
  }, []);
  if (!bootstrap) return <main className="wb-loading"><h1>GPU Insights</h1><p role="status">{error || "Opening your local API workbench…"}</p>{error && <button onClick={() => window.location.reload()}>Reconnect</button>}</main>;
  return <Workbench bootstrap={bootstrap} />;
}

function Workbench({ bootstrap }: { bootstrap: Bootstrap }) {
  const { token } = bootstrap;
  const progressRef = useRef<HTMLElement>(null);
  const { job, setJob, history, result, setResult, loadResult, error, setError } = useWorkbench(token);
  const [config, setConfig] = useState(bootstrap.defaults);
  const [profiles, setProfiles] = useState(bootstrap.profiles);
  const [profileName, setProfileName] = useState("");
  const [key, setKey] = useState("");
  const [useSavedKey, setUseSavedKey] = useState(false);
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState("");
  const running = job?.status === "running" || job?.status === "stopping";
  const busy = pending || running;
  const provider = bootstrap.providers.find((item) => item.id === config.provider);
  const samples = job?.events.flatMap((event) => event.sample ? [event.sample] : []) ?? [];

  function update<K extends keyof Config>(name: K, value: Config[K]) {
    setConfig((previous) => ({ ...previous, [name]: value }));
  }

  async function action(task: () => Promise<void>) {
    setPending(true);
    setError("");
    setNotice("");
    try { await task(); } catch (error) { setError(errorMessage(error)); }
    finally { setPending(false); }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    void action(async () => {
      const next = await api<Job>("run", token, { config, apiKey: key || null, useSavedKey });
      setResult(null);
      setJob(next);
      setKey("");
      progressRef.current?.scrollIntoView({ block: "start" });
    });
  }

  return <main className="wb-shell">
    <header className="wb-header">
      <div className="wb-logo" aria-hidden="true">GI</div>
      <div><span className="wb-eyebrow">GPU Insights / Local workbench</span><h1>LLM API performance</h1></div>
      <span className="wb-local"><span /> Running on your computer</span>
    </header>
    <div className="wb-intro"><p>Configure a connection. Measure the service. Collect the payload.</p><span>Requests originate from this computer; network and queue time are included.</span></div>
    {error && <div className="wb-error" role="alert">{error}</div>}
    {notice && <p className="wb-feedback" role="status">{notice}</p>}
    <div className="wb-layout">
      <div className="wb-config-column">
        <form className="wb-card" onSubmit={submit}>
          <div className="wb-section-head"><div><span className="wb-eyebrow">01 / Configure</span><h2>Connection</h2></div></div>
          <fieldset disabled={busy}>
            <Field label="Saved connection">
              <select aria-label="Saved connection" defaultValue="" onChange={(event) => {
                const name = event.target.value;
                if (!name) return;
                setConfig(profiles[name]);
                setProfileName(name); setKey(""); setUseSavedKey(false);
              }}>
                <option value="">Choose a saved profile</option>
                {Object.keys(profiles).map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </Field>
            <Field label="Provider / protocol">
              <select value={config.provider} onChange={(event) => {
                const selected = bootstrap.providers.find((item) => item.id === event.target.value)!;
                setConfig((previous) => ({ ...previous, provider: selected.id, base_url: selected.baseUrl }));
                setKey(""); setUseSavedKey(false);
              }}>
                {bootstrap.providers.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
              </select>
            </Field>
            <Field label="Model"><input required placeholder="Model ID used by your endpoint" value={config.model} onChange={(event) => update("model", event.target.value)} /></Field>
            <Field label="Base URL" hint="Include /v1 when required by the endpoint."><input type="url" required placeholder="http://127.0.0.1:8000/v1" value={config.base_url} onChange={(event) => update("base_url", event.target.value)} /></Field>
            <Field label="Credentials">
              <select value={useSavedKey ? "saved" : "session"} onChange={(event) => { setUseSavedKey(event.target.value === "saved"); setKey(""); }}>
                <option value="session">Enter a session key / no key</option>
                <option value="saved">Environment / .env {provider?.keyConfigured ? "· configured" : "· no key configured"}</option>
              </select>
            </Field>
            {!useSavedKey && <Field label="Session API key" hint="Optional for local servers. Used for this run; never saved in profiles or payloads."><input type="password" autoComplete="off" spellCheck={false} value={key} onChange={(event) => setKey(event.target.value)} placeholder="Enter key if required" /></Field>}
            <Field label="Client region" hint="A comparison label, such as us-east or home-network."><input value={config.region} onChange={(event) => update("region", event.target.value)} placeholder="Where this computer is running" /></Field>
            <div className="wb-profile-save"><input aria-label="Profile name" placeholder="Connection profile name" maxLength={100} value={profileName} onChange={(event) => setProfileName(event.target.value)} /><button type="button" disabled={!profileName.trim()} onClick={() => void action(async () => {
              setProfiles(await api<Record<string, Config>>("profiles", token, { name: profileName, config }));
              setNotice("Connection profile saved without credentials.");
            })}>Save profile</button></div>
            <details className="wb-advanced"><summary>Connection compatibility</summary>
              <label className="wb-check"><input type="checkbox" checked={config.include_usage} onChange={(event) => update("include_usage", event.target.checked)} />Request stream usage (required for token-rate estimates)</label>
              <label className="wb-check"><input type="checkbox" checked={config.allow_insecure_http} onChange={(event) => update("allow_insecure_http", event.target.checked)} />Allow sending a key to a non-local HTTP endpoint</label>
            </details>
          </fieldset>
          <div className="wb-plan" aria-label="Standard suite">
            <span className="wb-eyebrow">{bootstrap.suite.id}</span>
            <strong>{bootstrap.suite.requests} requests · up to {bootstrap.suite.outputTokens.toLocaleString()} output tokens</strong>
            <span>{bootstrap.suite.warmups} excluded warmups · {bootstrap.suite.repetitions} repetitions at every length</span>
            <span>Input scan: {bootstrap.suite.inputSizes.join(" / ")} characters</span>
            <span>Output scan: {bootstrap.suite.outputLimits.join(" / ")} tokens</span>
            <span>Fixed workload · no requests until you click Run · not a price estimate</span>
          </div>
          <div className="wb-actions wb-run-actions"><button className="wb-primary" type="submit" disabled={busy}>{running ? "Benchmark running…" : "Run standard suite"}</button></div>
        </form>
        <section className="wb-card"><div className="wb-section-head"><div><span className="wb-eyebrow">Saved on this computer</span><h2>Run history <small>{history.length}</small></h2></div></div>
          <p className="wb-muted wb-history-path">{bootstrap.outputDir}</p>
          {history.length === 0 ? <p className="wb-muted">Completed and stopped runs will appear here.</p> : <div className="wb-history">{history.map((entry) => <button key={entry.file} onClick={() => void action(async () => loadResult(entry.file))}>
            <span><strong>{entry.config.model}</strong><small>{entry.config.provider} · {entry.config.clientRegion || "region unspecified"} · C{entry.config.concurrency}</small></span>
            <span><small>{new Date(entry.createdAt).toLocaleString()}</small><small>{entry.status} · {entry.summary.successes} successful</small></span>
          </button>)}</div>}
        </section>
      </div>
      <div className="wb-results-column">
        <section ref={progressRef} className="wb-card" aria-label="Run progress">
          <div className="wb-section-head"><div><span className="wb-eyebrow">02 / Measure</span><h2>Run progress</h2></div><span className={`wb-badge ${running ? "wb-amber" : ""}`}>{job?.status ?? "Ready"}</span></div>
          <div className="wb-progress-label"><span>{job ? `${job.completed} / ${job.total} requests finished` : "No active benchmark"}</span>{running && <button disabled={job?.status === "stopping"} onClick={() => void action(async () => setJob(await api<Job>("cancel", token, {})))}>{job?.status === "stopping" ? "Stopping…" : "Stop run"}</button>}</div>
          <progress max={job?.total ?? 1} value={job?.completed ?? 0} aria-label="Completed requests" />
          {job?.status === "stopping" && <p className="wb-notice">No new requests will start. Waiting for active requests to finish; completed samples will be saved.</p>}
          {job?.error && <p role="alert" className="wb-error">{job.error}</p>}
          {samples.length > 0 ? <div className="wb-sample-scroll"><table><thead><tr><th>Phase</th><th>Status</th><th>Input chars</th><th>TTFT</th><th>Total</th></tr></thead><tbody>{samples.map((sample, index) => <tr key={index}><td>{sample.phase}</td><td className={sample.status === "ok" ? "wb-ok" : "wb-failed"}>{sample.status}{sample.error && ` · ${sample.error}`}</td><td>{sample.promptChars.toLocaleString()}</td><td>{sample.ttftMs == null ? "—" : `${sample.ttftMs.toFixed(1)} ms`}</td><td>{sample.totalMs == null ? "—" : `${sample.totalMs.toFixed(1)} ms`}</td></tr>)}</tbody></table></div> : <p className="wb-muted">Warmup and per-request measurements appear here as they finish.</p>}
        </section>
        <ResultPanel key={result?.path ?? "empty"} result={result} />
        <footer className="wb-footer">Measured locally · estimates require stable length/latency evidence · payloads contain no API keys, prompts, or response text</footer>
      </div>
    </div>
  </main>;
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="wb-field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}

